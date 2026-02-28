#!/usr/bin/env python3
"""Add codec_tokens and voice_state to existing cache.

Usage:
    uv run python scripts/add_codec_to_cache.py --speaker vctk_p225 --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_speaker(
    speaker_id: str,
    cache_dir: Path,
    raw_dir: Path,
    codec,
    vs_estimator,
    device: str,
) -> int:
    """Process all utterances for a speaker.

    Returns number of processed files.
    """
    import soundfile as sf
    from tmrvc_core.audio import compute_mel

    # Parse speaker ID (e.g., "vctk_p225" -> "p225")
    parts = speaker_id.split("_")
    if len(parts) == 2:
        raw_speaker = parts[1]  # p225
    else:
        raw_speaker = speaker_id

    speaker_cache_dir = cache_dir / speaker_id
    if not speaker_cache_dir.exists():
        logger.warning("Speaker cache not found: %s", speaker_cache_dir)
        return 0

    processed = 0
    skipped = 0

    for utt_dir in sorted(speaker_cache_dir.iterdir()):
        if not utt_dir.is_dir():
            continue

        # Check if already processed
        if (utt_dir / "codec_tokens.npy").exists():
            skipped += 1
            continue

        # Load meta
        meta_path = utt_dir / "meta.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        utt_id = meta.get("utterance_id", utt_dir.name)

        # Construct audio path
        # utt_id: "vctk_p225_002" -> "p225/p225_002_mic1.flac"
        utt_num = utt_id.split("_")[-1]
        audio_path = raw_dir / raw_speaker / f"{raw_speaker}_{utt_num}_mic1.flac"

        if not audio_path.exists():
            # Try .wav
            audio_path = raw_dir / raw_speaker / f"{raw_speaker}_{utt_num}_mic1.wav"

        if not audio_path.exists():
            logger.debug("Audio not found: %s", audio_path)
            continue

        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Normalize
            audio = audio / (np.abs(audio).max() + 1e-8)
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
            audio_t = audio_t.to(device)

            # Extract codec tokens
            with torch.no_grad():
                tokens = codec.encode_simple(audio_t)

            # Compute mel and voice state
            mel = compute_mel(audio_t.cpu())
            f0 = torch.zeros(1, 1, mel.shape[-1])

            with torch.no_grad():
                vs = vs_estimator.estimate(mel, f0)

            # Resample voice_state to match codec frame rate
            n_codec_frames = tokens.shape[-1]
            n_voice_frames = vs.shape[1]

            if n_voice_frames != n_codec_frames:
                vs = torch.nn.functional.interpolate(
                    vs.transpose(1, 2),
                    size=n_codec_frames,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            # Save
            np.save(utt_dir / "codec_tokens.npy", tokens[0].cpu().numpy())
            np.save(utt_dir / "voice_state.npy", vs[0].cpu().numpy())

            # Update meta
            meta["has_codec_tokens"] = True
            meta["n_codec_frames"] = int(n_codec_frames)
            with open(meta_path, "w") as f:
                json.dump(meta, f)

            processed += 1
            if processed % 10 == 0:
                logger.info("Processed %d files...", processed)

        except Exception as e:
            logger.warning("Failed to process %s: %s", utt_dir.name, e)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Add codec tokens to cache")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/vctk/train"))
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw/wav48_silence_trimmed")
    )
    parser.add_argument(
        "--speaker", type=str, required=True, help="Speaker ID (e.g., vctk_p225)"
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger.info("Loading models...")

    from tmrvc_data.codec import EnCodecWrapper
    from tmrvc_data.voice_state import VoiceStateEstimator

    codec = EnCodecWrapper(device=args.device)
    vs_estimator = VoiceStateEstimator(device=args.device)

    logger.info("Processing speaker: %s", args.speaker)

    processed = process_speaker(
        speaker_id=args.speaker,
        cache_dir=args.cache_dir,
        raw_dir=args.raw_dir,
        codec=codec,
        vs_estimator=vs_estimator,
        device=args.device,
    )

    logger.info("Done! Processed %d files", processed)


if __name__ == "__main__":
    main()
