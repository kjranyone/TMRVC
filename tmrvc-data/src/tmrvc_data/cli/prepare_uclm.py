"""``tmrvc-prepare-uclm`` — Add UCLM-specific features to existing cache.

Extends existing feature cache with UCLM training files:
- codec_tokens.npy: [8, T] RVQ acoustic tokens (A_t)
- control_tokens.npy: [4, T] control tokens (B_t)
- explicit_state.npy: [T, 8] heuristic voice state parameters
- ssl_state.npy: [T, 128] WavLM SSL features

Requires either:
- waveform.npy in cache (saved with --save-waveform during tmrvc-preprocess)
- Or original audio files accessible via meta.json path

Usage::

    # From existing cache with waveform.npy
    tmrvc-prepare-uclm --cache-dir data/cache/vctk --device cuda

    # Specify audio directory for re-loading
    tmrvc-prepare-uclm --cache-dir data/cache/vctk --audio-dir data/raw/VCTK-Corpus --device cuda

    # Process specific dataset/split
    tmrvc-prepare-uclm --cache-dir data/cache --dataset vctk --split train --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio

from tmrvc_core.constants import HOP_LENGTH, SAMPLE_RATE
from tmrvc_data.cache import FeatureCache
from tmrvc_data.codec import EnCodecWrapper
from tmrvc_data.voice_state import VoiceStateEstimator, SSLVoiceStateEstimator

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-prepare-uclm",
        description="Add UCLM features to existing feature cache.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name (process all if not specified).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (default: train).",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Path to raw audio directory (for re-loading if waveform.npy missing).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip utterances that already have UCLM features.",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=0,
        help="Max utterances to process (0=all, for debugging).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def load_audio_from_cache(
    utt_dir: Path,
    audio_dir: Path | None = None,
) -> tuple[torch.Tensor, int] | tuple[None, None]:
    """Load audio waveform for an utterance.

    Tries:
    1. waveform.npy in cache
    2. Original audio file via meta.json audio_path

    Returns:
        (waveform, sample_rate) or (None, None) if not found.
    """
    wav_path = utt_dir / "waveform.npy"
    if wav_path.exists():
        waveform = np.load(wav_path)
        return torch.from_numpy(waveform), SAMPLE_RATE

    if audio_dir is not None:
        meta_path = utt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            audio_path = meta.get("audio_path")
            if audio_path:
                audio_path = Path(audio_path)
                if not audio_path.is_absolute():
                    audio_path = audio_dir / audio_path
                if audio_path.exists():
                    waveform, sr = torchaudio.load(audio_path)
                    return waveform, sr

    return None, None


def extract_uclm_features(
    waveform: torch.Tensor,
    sample_rate: int,
    mel: np.ndarray,
    f0: np.ndarray,
    codec: EnCodecWrapper,
    voice_state_estimator: VoiceStateEstimator,
    ssl_estimator: SSLVoiceStateEstimator | None,
    device: str,
) -> dict[str, np.ndarray]:
    """Extract UCLM-specific features from audio.

    Args:
        waveform: [1, T_samples] or [T_samples] audio tensor
        sample_rate: Sample rate of waveform
        mel: [80, T] mel spectrogram
        f0: [1, T] f0 contour
        codec: EnCodec wrapper
        voice_state_estimator: VoiceStateEstimator
        ssl_estimator: SSLVoiceStateEstimator (or None to skip SSL)
        device: Device for inference

    Returns:
        Dict with codec_tokens, control_tokens, explicit_state, ssl_state
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    T = mel.shape[1]

    with torch.no_grad():
        codec_tokens = codec.encode_simple(waveform, sample_rate=sample_rate)
        codec_tokens = codec_tokens.squeeze(0).cpu().numpy()

        if codec_tokens.shape[1] != T:
            try:
                import scipy.ndimage

                codec_tokens = scipy.ndimage.zoom(
                    codec_tokens, (1, T / codec_tokens.shape[1]), order=0
                )
                codec_tokens = codec_tokens.astype(np.int64)
            except ImportError:
                indices = np.linspace(0, codec_tokens.shape[1] - 1, T).astype(np.int64)
                codec_tokens = codec_tokens[:, indices]

    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
    f0_t = torch.from_numpy(f0).unsqueeze(0).to(device)

    with torch.no_grad():
        explicit_state = voice_state_estimator.estimate(mel_t, f0_t)
        explicit_state = explicit_state.squeeze(0).cpu().numpy()

    ssl_state = np.zeros((T, 128), dtype=np.float32)
    if ssl_estimator is not None:
        with torch.no_grad():
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000).to(
                    device
                )
                audio_16k = resampler(waveform.to(device))
            else:
                audio_16k = waveform.to(device)

            if sample_rate != SAMPLE_RATE:
                resampler_24k = torchaudio.transforms.Resample(
                    sample_rate, SAMPLE_RATE
                ).to(device)
                audio_24k = resampler_24k(waveform.to(device))
            else:
                audio_24k = waveform.to(device)

            ssl_out = ssl_estimator(audio_16k, audio_24k, mel_t, f0_t)
            ssl_state = ssl_out["ssl_state"].squeeze(0).cpu().numpy()

            if ssl_state.shape[0] != T:
                ssl_state = np.interp(
                    np.linspace(0, 1, T),
                    np.linspace(0, 1, ssl_state.shape[0]),
                    ssl_state.T,
                ).T.astype(np.float32)

    control_tokens = np.zeros((4, T), dtype=np.int64)

    return {
        "codec_tokens": codec_tokens,
        "control_tokens": control_tokens,
        "explicit_state": explicit_state,
        "ssl_state": ssl_state,
    }


def process_utterance(
    utt_dir: Path,
    audio_dir: Path | None,
    codec: EnCodecWrapper,
    voice_state_estimator: VoiceStateEstimator,
    ssl_estimator: SSLVoiceStateEstimator | None,
    device: str,
    skip_existing: bool,
) -> bool:
    """Process a single utterance.

    Returns:
        True if processed, False if skipped.
    """
    required = ["mel.npy", "f0.npy", "meta.json"]
    if not all((utt_dir / f).exists() for f in required):
        logger.debug("Missing required files in %s", utt_dir)
        return False

    if skip_existing and (utt_dir / "codec_tokens.npy").exists():
        return False

    waveform, sr = load_audio_from_cache(utt_dir, audio_dir)
    if waveform is None or sr is None:
        logger.debug("No audio available for %s", utt_dir)
        return False

    mel = np.load(utt_dir / "mel.npy")
    f0 = np.load(utt_dir / "f0.npy")

    features = extract_uclm_features(
        waveform, sr, mel, f0, codec, voice_state_estimator, ssl_estimator, device
    )

    np.save(utt_dir / "codec_tokens.npy", features["codec_tokens"])
    np.save(utt_dir / "control_tokens.npy", features["control_tokens"])
    np.save(utt_dir / "explicit_state.npy", features["explicit_state"])
    np.save(utt_dir / "ssl_state.npy", features["ssl_state"])

    with open(utt_dir / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    meta["has_uclm_features"] = True
    with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return True


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache = FeatureCache(args.cache_dir)

    logger.info("Loading models on %s ...", args.device)
    codec = EnCodecWrapper(device=args.device)
    voice_state_estimator = VoiceStateEstimator(device=args.device)

    ssl_estimator = None
    try:
        ssl_estimator = SSLVoiceStateEstimator(device=args.device)
        logger.info("SSL extractor loaded")
    except Exception as e:
        logger.warning("SSL extractor not available: %s. Using zeros.", e)

    datasets = [args.dataset] if args.dataset else []
    if not datasets:
        base = args.cache_dir
        if base.exists():
            for ds_dir in sorted(base.iterdir()):
                if ds_dir.is_dir() and (ds_dir / args.split).exists():
                    datasets.append(ds_dir.name)

    logger.info("Processing datasets: %s", datasets)

    processed = 0
    skipped = 0
    errors = 0

    for dataset in datasets:
        entries = cache.iter_entries(dataset, args.split)
        logger.info("Dataset %s: %d utterances", dataset, len(entries))

        for entry in entries:
            if 0 < args.max_utterances <= processed:
                break

            utt_dir = (
                args.cache_dir
                / dataset
                / args.split
                / entry["speaker_id"]
                / entry["utterance_id"]
            )

            try:
                if process_utterance(
                    utt_dir,
                    args.audio_dir,
                    codec,
                    voice_state_estimator,
                    ssl_estimator,
                    args.device,
                    args.skip_existing,
                ):
                    processed += 1
                else:
                    skipped += 1

                if processed % 100 == 0 and processed > 0:
                    logger.info(
                        "Processed %d (skipped %d, errors %d)",
                        processed,
                        skipped,
                        errors,
                    )

            except Exception:
                logger.error(
                    "Failed to process %s/%s",
                    entry["speaker_id"],
                    entry["utterance_id"],
                    exc_info=True,
                )
                errors += 1

    logger.info("Done. Processed=%d, Skipped=%d, Errors=%d", processed, skipped, errors)


if __name__ == "__main__":
    main()
