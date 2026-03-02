#!/usr/bin/env python3
"""Inject text transcripts from raw datasets into cache meta.json.

Reads transcripts from raw VCTK, JVS, and Tsukuyomi directories and adds
'text' and 'language_id' fields to each utterance's meta.json in the feature cache.
This ensures high-quality TTS training by using ground-truth text instead of ASR.

Usage::

    python scripts/annotate/inject_text_to_cache.py --cache-dir data/cache --raw-dir data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def inject_vctk(cache_dir: Path, raw_dir: Path) -> int:
    """Inject VCTK English transcripts into cache."""
    # VCTK structure can vary, try common ones
    txt_root = raw_dir / "vctk_txt"
    if not txt_root.exists():
        txt_root = raw_dir / "VCTK-Corpus" / "VCTK-Corpus" / "txt"
    
    cache_root = cache_dir / "vctk" / "train"

    if not txt_root.exists():
        logger.warning("VCTK txt dir not found at %s", txt_root)
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
    """Inject JVS Japanese transcripts into cache."""
    jvs_root = raw_dir / "jvs_corpus" / "jvs_ver1"
    cache_root = cache_dir / "jvs" / "train"

    if not jvs_root.exists():
        logger.warning("JVS root dir not found at %s", jvs_root)
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

            transcripts: dict[str, str] = {}
            for line in transcript_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                corpus_id, text = line.split(":", 1)
                transcripts[corpus_id.strip()] = text.strip()

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


def inject_tsukuyomi(cache_dir: Path, raw_dir: Path) -> int:
    """Inject Tsukuyomi-chan ground-truth text into cache."""
    # Official transcript path
    transcript_path = raw_dir / "tsukuyomi/corpus1/04 台本と補足資料/★台本テキスト/01 補足なし台本（JSUTコーパス・JVSコーパス版）.txt"
    # Tsukuyomi speaker folder name in cache is normalized to tsukuyomi_chan or tsukuyomi_tsukuyomi
    cache_root = cache_dir / "tsukuyomi" / "train"

    if not transcript_path.exists():
        logger.warning("Tsukuyomi transcript not found at %s", transcript_path)
        return 0

    if not cache_root.exists():
        logger.warning("Tsukuyomi cache root not found at %s", cache_root)
        return 0

    # ID format: VOICEACTRESS100_001
    transcripts = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            utt_id, text = line.split(":", 1)
            transcripts[utt_id.strip()] = text.strip()

    count = 0
    # TsukuyomiAdapter uses speaker_id="tsukuyomi_tsukuyomi" or similar
    for spk_dir in cache_root.iterdir():
        if not spk_dir.is_dir(): continue
        
        for utt_dir in spk_dir.iterdir():
            if not utt_dir.is_dir(): continue
            
            # Cache ID: tsukuyomi_VOICEACTRESS100_001
            raw_id = utt_dir.name.replace("tsukuyomi_", "")
            if raw_id in transcripts:
                meta_path = utt_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["text"] = transcripts[raw_id]
                    meta["language_id"] = 0  # ja
                    meta_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8"
                    )
                    count += 1
                    
    logger.info("Tsukuyomi: injected %d transcripts", count)
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
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    total = 0
    total += inject_vctk(args.cache_dir, args.raw_dir)
    total += inject_jvs(args.cache_dir, args.raw_dir)
    total += inject_tsukuyomi(args.cache_dir, args.raw_dir)
    logger.info("Total: %d transcripts injected", total)


if __name__ == "__main__":
    main()
