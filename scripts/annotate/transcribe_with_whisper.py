#!/usr/bin/env python3
"""Transcribe audio files with Whisper and inject into cache meta.json.

Usage:
    uv run python scripts/transcribe_with_whisper.py \
        --cache-dir data/cache --dataset custom_speaker \
        --audio-dir data/wav --language ja --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import tqdm

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Transcribe with Whisper")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="ja", choices=["ja", "en", "zh"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from faster_whisper import WhisperModel

    logger.info("Loading Whisper model '%s' on %s...", args.model, args.device)
    compute_type = "int8" if args.device == "cpu" else "float16"
    model = WhisperModel(args.model, device=args.device, compute_type=compute_type)

    LANGUAGE_IDS = {"ja": 0, "en": 1, "zh": 2}
    lang_id = LANGUAGE_IDS[args.language]

    cache_root = args.cache_dir / args.dataset / args.split
    if not cache_root.exists():
        logger.error("Cache root not found: %s", cache_root)
        return

    meta_paths = sorted(cache_root.rglob("meta.json"))
    logger.info("Found %d cache entries", len(meta_paths))

    updated = 0
    errors = 0

    for meta_path in tqdm.tqdm(meta_paths, desc="Transcribing"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        utt_id = meta.get("utterance_id", "")

        if "text" in meta and meta["text"]:
            continue

        stem = utt_id.split("_")[-1]
        audio_path = args.audio_dir / f"{stem}.wav"

        if not audio_path.exists():
            logger.debug("Audio not found: %s", audio_path)
            errors += 1
            continue

        try:
            segments, _ = model.transcribe(str(audio_path), language=args.language)
            text = "".join(seg.text for seg in segments).strip()

            if text:
                meta["text"] = text
                meta["language_id"] = lang_id
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                updated += 1
        except Exception as e:
            logger.warning("Error transcribing %s: %s", utt_id, e)
            errors += 1

    logger.info("Done. Updated %d entries, %d errors", updated, errors)


if __name__ == "__main__":
    main()
