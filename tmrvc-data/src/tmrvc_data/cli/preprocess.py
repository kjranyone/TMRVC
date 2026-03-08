"""``tmrvc-preprocess`` — UCLM full preprocessing pipeline.

Unified extraction of dual-stream tokens, voice state, and TTS alignment.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import numpy as np
import tqdm

from tmrvc_core.audio import compute_mel
from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.types import UCLMFeatureSet
from tmrvc_data.cache import FeatureCache
from tmrvc_data.dataset_adapters import get_adapter
from tmrvc_data.codec import UCLMCodecWrapper
from tmrvc_data.voice_state import SSLVoiceStateEstimator
from tmrvc_data.preprocessing import preprocess_audio, segment_utterance
from tmrvc_data.speaker import SpeakerEncoder

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-preprocess",
        description="Extract UCLM features from raw audio to cache.",
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--codec-checkpoint", type=Path, default=None)
    parser.add_argument("--language", default="ja", choices=["ja", "en", "zh", "ko"])
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Ratio of dataset to process (0.0-1.0)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--export-golden-fixture", action="store_true", help="Export text frontend parity data for Rust (Worker 06).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve adapter from datasets.yaml to handle speaker_map
    adapter_type: str | None = None
    language: str = args.language
    speaker_map_path: str | None = None
    datasets_yaml = Path("configs/datasets.yaml")
    if datasets_yaml.exists():
        import yaml

        with open(datasets_yaml, encoding="utf-8") as _f:
            _registry = yaml.safe_load(_f) or {}
        ds_cfg = (_registry.get("datasets") or {}).get(args.dataset) or {}
        adapter_type = ds_cfg.get("type")
        language = ds_cfg.get("language", args.language)
        speaker_map_path = ds_cfg.get("speaker_map")

    adapter = get_adapter(
        args.dataset,
        adapter_type=adapter_type,
        language=language,
        speaker_map_path=speaker_map_path,
    )
    cache = FeatureCache(args.cache_dir)

    # Load models
    logger.info("Loading UCLM extraction models on %s...", args.device)
    codec = UCLMCodecWrapper(args.codec_checkpoint, device=args.device)
    vs_estimator = SSLVoiceStateEstimator(device=args.device)
    spk_encoder = SpeakerEncoder(device=args.device)

    # ASR for TTS data
    from faster_whisper import WhisperModel

    compute_type = "float16" if args.device == "cuda" else "int8"
    # Use turbo model for faster transcription (8x faster than large-v3)
    whisper = WhisperModel(
        "large-v3-turbo", device=args.device, compute_type=compute_type
    )

    utterances = list(adapter.iter_utterances(args.raw_dir, args.split))
    logger.info("Found %d utterances in %s", len(utterances), args.dataset)

    # Random sampling if requested
    if args.sample_ratio < 1.0:
        n_sample = max(1, int(len(utterances) * args.sample_ratio))
        logger.info(
            "Sampling %d utterances (ratio=%.2f)...", n_sample, args.sample_ratio
        )
        utterances = random.sample(utterances, n_sample)

    processed = 0
    for utt in tqdm.tqdm(utterances, desc="Preprocessing"):
        if args.skip_existing and cache.exists(
            args.dataset, args.split, utt.speaker_id, utt.utterance_id
        ):
            continue

        try:
            # 0. Early Length Guard
            import soundfile as sf

            info = sf.info(str(utt.audio_path))
            if info.duration < 0.1 or info.duration > 30.0:
                logger.debug(
                    "Skipping %s due to duration (%.2fs)",
                    utt.utterance_id,
                    info.duration,
                )
                continue

            # 1. Load & Normalize
            waveform, sr = preprocess_audio(str(utt.audio_path), target_sr=SAMPLE_RATE)
            waveform_t = waveform.unsqueeze(0).to(args.device)

            # 2. Extract Dual-stream Tokens
            a_tokens, b_logits = codec.encode(waveform_t)
            b_tokens = b_logits.argmax(dim=-1)

            # 3. Extract Voice State (Physical + SSL)
            mel = compute_mel(waveform_t.squeeze(1)).to(args.device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=args.device)

            import torchaudio.transforms as T

            waveform_16k = T.Resample(SAMPLE_RATE, 16000).to(args.device)(
                waveform_t.squeeze(1)
            )
            vs_dict = vs_estimator(waveform_16k, waveform_t.squeeze(1), mel, f0)

            # 4. Transcribe & Align
            segments, _ = whisper.transcribe(
                str(utt.audio_path), language=args.language
            )
            text = "".join(seg.text for seg in segments).strip()

            # Worker 03: Run G2P to get phoneme_ids and suprasegmentals (accents/tones)
            from tmrvc_data.g2p import text_to_phonemes
            g2p_res = text_to_phonemes(text, language=args.language)
            phoneme_ids = g2p_res.phoneme_ids
            text_suprasegmentals = g2p_res.text_suprasegmentals # [L, D]

            # Worker 06: Export Golden Fixture for Python/Rust G2P parity
            if args.export_golden_fixture:
                fixture_path = args.cache_dir / "golden_fixtures" / f"{utt.utterance_id}_g2p.json"
                fixture_path.parent.mkdir(parents=True, exist_ok=True)
                import json
                with open(fixture_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "text": text,
                        "language": args.language,
                        "phoneme_ids": phoneme_ids.tolist(),
                        "text_suprasegmentals": text_suprasegmentals.tolist() if text_suprasegmentals is not None else None,
                    }, f, ensure_ascii=False, indent=2)

            spk_embed = spk_encoder.extract(waveform_t.squeeze(1))

            # 5. Frame Alignment Verification (CRITICAL: must match exactly)
            # All temporal features at codec rate MUST have T_target frames.
            # Any mismatch indicates a bug in extraction logic.
            T_target = a_tokens.shape[-1]
            T_mel = mel.shape[-1]

            # Verify mel frames match codec frames (MUST be exact)
            assert T_mel == T_target, (
                f"Frame mismatch: mel={T_mel}, codec={T_target}. "
                f"This indicates a bug in MelSpectrogram or codec implementation."
            )

            # Get voice state (should match mel frames)
            explicit_state = vs_dict["explicit_state"].detach().cpu()
            if explicit_state.dim() == 3:
                explicit_state = explicit_state.squeeze(0)  # [T, 8]
            T_explicit = explicit_state.shape[0]

            # Verify explicit_state frames match (MUST be exact)
            assert T_explicit == T_target, (
                f"Frame mismatch: explicit_state={T_explicit}, codec={T_target}. "
                f"This indicates a bug in VoiceStateEstimator implementation."
            )
            explicit_state = explicit_state.transpose(0, 1)  # [8, T]

            # SSL state at 50Hz needs interpolation to 100Hz
            ssl_state = vs_dict["ssl_state"].detach().cpu()
            if ssl_state.dim() == 3:
                ssl_state = ssl_state.squeeze(0)  # [T_ssl, 128]
            T_ssl = ssl_state.shape[0]

            # Interpolate SSL from 50Hz to 100Hz (T_target frames)
            # This is NOT padding - it's proper signal interpolation
            ssl_state = (
                torch.nn.functional.interpolate(
                    ssl_state.unsqueeze(0).transpose(1, 2),  # [1, 128, T_ssl]
                    size=T_target,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .squeeze(0)
            )  # [T_target, 128]
            ssl_state = ssl_state.transpose(0, 1)  # [128, T_target]

            # Align codec_tokens_b if needed (should already match)
            b_tokens_aligned = b_tokens.detach().cpu().squeeze(0)  # [4, T]
            T_b = b_tokens_aligned.shape[-1]
            assert T_b == T_target, (
                f"Frame mismatch: b_tokens={T_b}, codec={T_target}. "
                f"This indicates a bug in codec implementation."
            )

            # 6. Save Unified FeatureSet
            features = UCLMFeatureSet(
                codec_tokens_a=a_tokens.detach().cpu().squeeze(0),
                codec_tokens_b=b_tokens_aligned,
                voice_state_explicit=explicit_state,
                voice_state_ssl=ssl_state,
                spk_embed=spk_embed.detach().cpu().squeeze(0),
                phoneme_ids=phoneme_ids.detach() if phoneme_ids is not None else None,
                durations=durations.detach() if durations is not None else None,
                text_suprasegmentals=text_suprasegmentals.detach() if text_suprasegmentals is not None else None,
                text=text,
                utterance_id=utt.utterance_id,
                speaker_id=utt.speaker_id,
                n_frames=T_target,
                waveform=waveform.detach(),
            )
            cache.save(features, args.dataset, args.split)
            processed += 1

        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM for %s, skipping", utt.utterance_id)
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error("Failed to process %s: %s", utt.utterance_id, e)
        finally:
            # Frequent cleanup
            import gc

            if processed % 10 == 0:
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()

    # Write manifest
    manifest_dir = args.cache_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{args.dataset}_{args.split}.json"

    manifest = {
        "dataset": args.dataset,
        "n_utterances": processed,
        "language": args.language,
        "uclm_ready": True,
    }
    manifest_path.write_text(torch.json.dumps(manifest, indent=2)) if hasattr(
        torch, "json"
    ) else manifest_path.write_text(import_json_and_dump(manifest))

    logger.info("Done. Processed %d utterances. Manifest: %s", processed, manifest_path)


def import_json_and_dump(d):
    import json

    return json.dumps(d, indent=2)


if __name__ == "__main__":
    main()
