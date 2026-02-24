#!/usr/bin/env python3
"""Preprocess emotion datasets for StyleEncoder training.

Parses emotion labels, extracts mel spectrograms, and writes cache entries
with emotion.json metadata for use with EmotionDataset.

Usage::

    python scripts/preprocess_emotion.py --dataset expresso --raw-dir data/raw/expresso --cache-dir data/cache
    python scripts/preprocess_emotion.py --dataset jvnv --raw-dir data/raw/jvnv --cache-dir data/cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", required=True,
        choices=["expresso", "jvnv", "emov_db", "ravdess"],
        help="Emotion dataset name.",
    )
    parser.add_argument("--raw-dir", required=True, type=Path, help="Raw dataset directory.")
    parser.add_argument("--cache-dir", required=True, type=Path, help="Feature cache directory.")
    parser.add_argument("--split", default="train", help="Split name (default: train).")
    parser.add_argument("--max-utterances", type=int, default=0, help="Limit utterances (0 = all).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from tmrvc_core.audio import compute_mel
    from tmrvc_core.constants import SAMPLE_RATE
    from tmrvc_core.dialogue_types import EMOTION_TO_ID
    from tmrvc_data.emotion_features import parse_dataset, RAVDESS_VAD

    raw_dir = args.raw_dir
    if not raw_dir.exists():
        logger.error("Raw directory not found: %s", raw_dir)
        sys.exit(1)

    logger.info("Parsing %s from %s", args.dataset, raw_dir)
    entries = parse_dataset(args.dataset, raw_dir)

    if args.max_utterances > 0:
        entries = entries[:args.max_utterances]

    logger.info("Found %d entries", len(entries))

    cache_dir = args.cache_dir / args.dataset / args.split
    processed = 0
    skipped = 0

    for i, entry in enumerate(entries):
        if not entry.audio_path.exists():
            skipped += 1
            continue

        # Create utterance directory
        utt_dir = cache_dir / entry.speaker_id / entry.utterance_id
        utt_dir.mkdir(parents=True, exist_ok=True)

        # Check if already processed
        if (utt_dir / "mel.npy").exists() and (utt_dir / "emotion.json").exists():
            processed += 1
            continue

        try:
            # Load audio
            audio, sr = sf.read(str(entry.audio_path), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # mono
            if sr != SAMPLE_RATE:
                import resampy
                audio = resampy.resample(audio, sr, SAMPLE_RATE)

            # Compute mel
            import torch
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            mel = compute_mel(audio_tensor).squeeze(0).numpy()  # [80, T]
            np.save(utt_dir / "mel.npy", mel)

            # Emotion metadata
            emotion_id = EMOTION_TO_ID.get(entry.emotion, EMOTION_TO_ID["neutral"])

            # VAD from entry fields
            vad = [entry.valence, entry.arousal, entry.dominance]

            emotion_meta = {
                "emotion_id": emotion_id,
                "emotion": entry.emotion,
                "vad": vad,
                "prosody": [0.0, 0.0, 0.0],  # TODO: compute from audio
            }
            with open(utt_dir / "emotion.json", "w", encoding="utf-8") as f:
                json.dump(emotion_meta, f, ensure_ascii=False)

            # Meta.json for cache compatibility
            meta = {
                "speaker_id": entry.speaker_id,
                "utterance_id": entry.utterance_id,
                "dataset": args.dataset,
                "emotion": entry.emotion,
            }
            with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)

            processed += 1
            if processed % 100 == 0:
                logger.info("Processed %d / %d", processed, len(entries))

        except Exception as e:
            logger.warning("Failed to process %s: %s", entry.utterance_id, e)
            skipped += 1

    logger.info("Done: %d processed, %d skipped out of %d total", processed, skipped, len(entries))


if __name__ == "__main__":
    main()
