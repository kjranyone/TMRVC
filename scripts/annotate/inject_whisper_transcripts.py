#!/usr/bin/env python3
"""Inject Whisper transcripts from bulk voice preparation into cache meta.json.

Reads ``transcripts.txt`` files (``stem|text`` format) from speaker subdirectories
and injects ``text`` and ``language_id`` fields into the corresponding cache
``meta.json`` files.

Usage::

    python scripts/inject_whisper_transcripts.py \\
        --cache-dir data/cache --dataset galge \\
        --transcript-dir data/raw/galge_clean --language ja
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

LANGUAGE_IDS = {"ja": 0, "en": 1, "zh": 2}


def load_transcripts_dir(transcript_dir: Path) -> dict[str, str]:
    """Load all ``transcripts.txt`` files from speaker subdirectories.

    Each file has lines in ``stem|text`` format.

    Returns:
        Mapping from utterance stem to transcript text.
    """
    transcripts: dict[str, str] = {}
    transcript_files = sorted(transcript_dir.rglob("transcripts.txt"))
    if not transcript_files:
        logger.warning("No transcripts.txt found under %s", transcript_dir)
        return transcripts

    for tf in transcript_files:
        for line in tf.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            stem, text = line.split("|", 1)
            stem = stem.strip()
            text = text.strip()
            if stem and text:
                transcripts[stem] = text

    logger.info("Loaded %d transcripts from %d files", len(transcripts), len(transcript_files))
    return transcripts


def inject_to_cache(
    cache_dir: Path,
    dataset: str,
    split: str,
    transcripts: dict[str, str],
    language_id: int,
) -> int:
    """Inject transcripts into cache meta.json files.

    Walks ``cache_dir/{dataset}/{split}/**/meta.json`` and matches each
    utterance_id against the transcripts dict.

    Returns:
        Number of meta.json files updated.
    """
    cache_root = cache_dir / dataset / split
    if not cache_root.exists():
        logger.error("Cache root not found: %s", cache_root)
        return 0

    count = 0
    for meta_path in sorted(cache_root.rglob("meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        utt_id = meta.get("utterance_id", "")

        # Try matching: full utterance_id, or the stem portion after prefix
        text = None
        if utt_id in transcripts:
            text = transcripts[utt_id]
        else:
            # GenericAdapter prefixes utterance_id with dataset name + "_"
            # Try stripping common prefixes to find the stem
            for key in transcripts:
                if utt_id.endswith(key) or utt_id.endswith(f"_{key}"):
                    text = transcripts[key]
                    break

        if text is None:
            continue

        meta["text"] = text
        meta["language_id"] = language_id
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        count += 1

    return count


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Inject Whisper transcripts into cache meta.json.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True, help="Feature cache directory.")
    parser.add_argument("--dataset", required=True, help="Dataset name in cache.")
    parser.add_argument("--transcript-dir", type=Path, required=True, help="Directory with transcripts.txt files.")
    parser.add_argument("--split", default="train", help="Split name (default: train).")
    parser.add_argument("--language", default="ja", choices=list(LANGUAGE_IDS), help="Language for language_id.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    transcripts = load_transcripts_dir(args.transcript_dir)
    if not transcripts:
        logger.error("No transcripts loaded, exiting.")
        sys.exit(1)

    language_id = LANGUAGE_IDS[args.language]
    count = inject_to_cache(args.cache_dir, args.dataset, args.split, transcripts, language_id)
    logger.info("Injected %d transcripts (language_id=%d) into %s/%s/%s",
                count, language_id, args.cache_dir, args.dataset, args.split)


if __name__ == "__main__":
    main()
