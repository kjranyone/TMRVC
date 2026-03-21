#!/usr/bin/env python3
"""Reorganize data/ for v4 training.

1. Delete ZIP files, v3 artifacts
2. Restructure moe_multispeaker_voices using speaker_map into per-speaker dirs
3. Reorganize raw/ into clean corpus structure
4. Delete old cache (will be rebuilt)

Usage:
    .venv/bin/python scripts/reorganize_data.py --dry-run   # preview
    .venv/bin/python scripts/reorganize_data.py --execute    # execute
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("reorg")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Preview only")
    p.add_argument("--execute", action="store_true", help="Actually execute")
    args = p.parse_args()

    if not args.dry_run and not args.execute:
        print("Specify --dry-run or --execute")
        sys.exit(1)

    dry = args.dry_run

    # =========================================================================
    # Step 1: Delete ZIP files
    # =========================================================================
    zips = list((DATA / "raw").glob("*.zip"))
    for z in zips:
        size_gb = z.stat().st_size / 1e9
        logger.info("[DELETE ZIP] %s (%.1f GB)%s", z.name, size_gb, " (dry)" if dry else "")
        if not dry:
            z.unlink()

    # =========================================================================
    # Step 2: Delete v3 artifacts
    # =========================================================================
    v3_dirs = [
        DATA / "alignments",
        DATA / "test_uclm_raw",
        DATA / "cache" / "smoke",
    ]
    for d in v3_dirs:
        if d.exists():
            size_mb = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
            logger.info("[DELETE V3] %s (%.0f MB)%s", d, size_mb, " (dry)" if dry else "")
            if not dry:
                shutil.rmtree(d)

    # =========================================================================
    # Step 3: Restructure moe_multispeaker_voices into per-speaker dirs
    # =========================================================================
    moe_dir = DATA / "moe_multispeaker_voices"
    moe_target = DATA / "raw" / "moe_voices"

    if moe_dir.exists():
        speaker_map_path = moe_dir / "_speaker_map.json"
        if speaker_map_path.exists():
            sm = json.loads(speaker_map_path.read_text())
            mapping = sm.get("mapping", {})
            n_noise = sm.get("n_noise", 0)
        else:
            mapping = {}

        # Count per speaker
        from collections import Counter
        spk_counts = Counter(mapping.values())
        logger.info("[MOE] %d files, %d speakers, %d noise",
                     len(mapping), len(spk_counts) - (1 if "spk_spk_noise" in spk_counts else 0), n_noise)

        if not dry:
            moe_target.mkdir(parents=True, exist_ok=True)

        # Move wavs into per-speaker subdirectories (symlink to save space, then delete original)
        moved = 0
        skipped_noise = 0
        for wav_name, spk_id in mapping.items():
            # Keys already include .wav extension
            src = moe_dir / wav_name
            if not src.exists():
                continue

            # Skip noise cluster
            if "noise" in spk_id:
                skipped_noise += 1
                continue

            # Clean speaker ID: "spk_0001" -> "moe_0001"
            clean_spk = spk_id.replace("spk_", "moe_")

            dst_dir = moe_target / clean_spk
            dst = dst_dir / src.name

            if not dry:
                dst_dir.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    # Move file
                    shutil.move(str(src), str(dst))
            moved += 1

        logger.info("[MOE] %d moved, %d noise skipped%s", moved, skipped_noise, " (dry)" if dry else "")

        # Delete the original moe directory after moving
        if not dry and moe_dir.exists():
            remaining = list(moe_dir.glob("*.wav"))
            if len(remaining) <= n_noise + 10:  # noise + margin
                logger.info("[DELETE] Removing %s (%d remaining noise wavs)", moe_dir, len(remaining))
                shutil.rmtree(moe_dir)
            else:
                logger.warning("[SKIP] %s still has %d wavs, not deleting", moe_dir, len(remaining))

    # =========================================================================
    # Step 4: Rename raw subdirs for clarity
    # =========================================================================
    renames = [
        (DATA / "raw" / "jvs_corpus", DATA / "raw" / "jvs"),
        (DATA / "raw" / "wav48_silence_trimmed", DATA / "raw" / "vctk"),
    ]
    for src, dst in renames:
        if src.exists() and not dst.exists():
            logger.info("[RENAME] %s -> %s%s", src.name, dst.name, " (dry)" if dry else "")
            if not dry:
                src.rename(dst)

    # Clean up empty/useless dirs in raw/
    cleanup_raw = [
        DATA / "raw" / "vctk_1pct",
        DATA / "raw" / "vctk_txt",
        DATA / "raw" / "speaker-info.txt",
        DATA / "raw" / "update.txt",
    ]
    for p in cleanup_raw:
        if p.exists():
            logger.info("[DELETE] %s%s", p, " (dry)" if dry else "")
            if not dry:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()

    # =========================================================================
    # Step 5: Delete old cache (will be rebuilt)
    # =========================================================================
    old_cache = DATA / "cache" / "v4full"
    if old_cache.exists():
        size_gb = sum(f.stat().st_size for f in old_cache.rglob("*") if f.is_file()) / 1e9
        logger.info("[DELETE CACHE] %s (%.1f GB)%s", old_cache, size_gb, " (dry)" if dry else "")
        if not dry:
            shutil.rmtree(old_cache)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("POST-REORGANIZATION STRUCTURE (expected):")
    logger.info("  data/raw/jvs/           — JVS 15k utterances (ja)")
    logger.info("  data/raw/vctk/          — VCTK 88k utterances (en, flac)")
    logger.info("  data/raw/tsukuyomi/     — つくよみ 304 utterances (ja)")
    logger.info("  data/raw/moe_voices/    — MOE 29k utterances, 5 speakers (ja)")
    logger.info("  data/cache/             — empty (rebuild needed)")
    logger.info("  data/curated_export/    — preserved")
    logger.info("  data/curation/          — preserved")
    logger.info("=" * 60)

    if dry:
        logger.info("DRY RUN — no changes made. Use --execute to apply.")


if __name__ == "__main__":
    main()
