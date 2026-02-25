#!/usr/bin/env python3
"""Inject text transcripts from raw datasets into cache meta.json.

Reads transcripts from raw VCTK and JVS directories and adds 'text' and
'language_id' fields to each utterance's meta.json in the feature cache.

Usage::

    python scripts/inject_text_to_cache.py --cache-dir data/cache --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def inject_vctk(cache_dir: Path, raw_dir: Path) -> int:
    """Inject VCTK English transcripts into cache.

    Raw structure: raw_dir/VCTK-Corpus/VCTK-Corpus/txt/{speaker}/{utt}.txt
    Cache structure: cache_dir/vctk/train/vctk_{speaker}/vctk_{speaker}_{utt}/meta.json
    """
    txt_root = raw_dir / "VCTK-Corpus" / "VCTK-Corpus" / "txt"
    cache_root = cache_dir / "vctk" / "train"

    if not txt_root.exists():
        logger.warning("VCTK txt dir not found: %s", txt_root)
        return 0

    count = 0
    for spk_dir in sorted(txt_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker = spk_dir.name  # e.g. "p225"
        cache_spk = cache_root / f"vctk_{speaker}"
        if not cache_spk.exists():
            continue

        for txt_file in sorted(spk_dir.glob("*.txt")):
            utt_id = txt_file.stem  # e.g. "p225_003"
            cache_utt = cache_spk / f"vctk_{utt_id}"
            meta_path = cache_utt / "meta.json"
            if not meta_path.exists():
                continue

            text = txt_file.read_text(encoding="utf-8").strip()
            if not text:
                continue

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["text"] = text
            meta["language_id"] = 1  # en
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            count += 1

    logger.info("VCTK: injected %d transcripts", count)
    return count


def inject_jvs(cache_dir: Path, raw_dir: Path) -> int:
    """Inject JVS Japanese transcripts into cache.

    Raw structure: raw_dir/jvs_corpus/jvs_ver1/{speaker}/{subset}/transcripts_utf8.txt
    Transcript format: CORPUS_ID:text (e.g. "BASIC5000_1356:母の死は...")
    Cache structure: cache_dir/jvs/train/jvs_{speaker}/jvs_{speaker}_{subset}_{corpus_id}/
    """
    jvs_root = raw_dir / "jvs_corpus" / "jvs_ver1"
    cache_root = cache_dir / "jvs" / "train"

    if not jvs_root.exists():
        logger.warning("JVS root dir not found: %s", jvs_root)
        return 0

    count = 0
    for spk_dir in sorted(jvs_root.iterdir()):
        if not spk_dir.is_dir() or not spk_dir.name.startswith("jvs"):
            continue
        speaker = spk_dir.name  # e.g. "jvs001"
        cache_spk = cache_root / f"jvs_{speaker}"
        if not cache_spk.exists():
            continue

        for subset_dir in sorted(spk_dir.iterdir()):
            if not subset_dir.is_dir():
                continue
            subset = subset_dir.name  # e.g. "nonpara30", "parallel100"
            transcript_path = subset_dir / "transcripts_utf8.txt"
            if not transcript_path.exists():
                continue

            # Parse transcript file
            transcripts: dict[str, str] = {}
            for line in transcript_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                corpus_id, text = line.split(":", 1)
                transcripts[corpus_id.strip()] = text.strip()

            # Match to cache entries
            for corpus_id, text in transcripts.items():
                cache_utt_name = f"jvs_{speaker}_{subset}_{corpus_id}"
                cache_utt = cache_spk / cache_utt_name
                meta_path = cache_utt / "meta.json"
                if not meta_path.exists():
                    continue

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["text"] = text
                meta["language_id"] = 0  # ja
                meta_path.write_text(
                    json.dumps(meta, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                count += 1

    logger.info("JVS: injected %d transcripts", count)
    return count


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Inject text into cache meta.json")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    total = 0
    total += inject_vctk(args.cache_dir, args.raw_dir)
    total += inject_jvs(args.cache_dir, args.raw_dir)
    logger.info("Total: %d transcripts injected", total)


if __name__ == "__main__":
    main()
