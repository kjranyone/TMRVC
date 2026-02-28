#!/usr/bin/env python3
"""Build breath/pause event labels for cached utterances.

Scans the feature cache, extracts breath and pause events from mel/F0,
and saves ``events.json`` per utterance.

Usage::

    uv run python scripts/build_breath_event_cache.py --cache-dir data/cache --dataset vctk
    uv run python scripts/build_breath_event_cache.py --cache-dir data/cache --dataset vctk,jvs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from tmrvc_data.cache import FeatureCache
from tmrvc_data.events import extract_events, save_events

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract breath/pause events and save to feature cache.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name(s), comma-separated.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing events.json.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache = FeatureCache(args.cache_dir)
    datasets = [d.strip() for d in args.dataset.split(",")]

    total_processed = 0
    total_events = 0
    total_breaths = 0
    total_pauses = 0

    for ds_name in datasets:
        entries = cache.iter_entries(ds_name, args.split)
        logger.info("Dataset '%s': %d entries", ds_name, len(entries))

        for i, entry in enumerate(entries):
            utt_dir = cache._utt_dir(
                ds_name, args.split, entry["speaker_id"], entry["utterance_id"],
            )

            if not args.overwrite and (utt_dir / "events.json").exists():
                continue

            mel = np.load(utt_dir / "mel.npy")
            f0 = np.load(utt_dir / "f0.npy")

            events = extract_events(mel, f0)
            save_events(events, utt_dir)

            n_breath = sum(1 for e in events if e["type"] == "breath")
            n_pause = sum(1 for e in events if e["type"] == "pause")
            total_events += len(events)
            total_breaths += n_breath
            total_pauses += n_pause
            total_processed += 1

            if args.verbose and (i + 1) % 500 == 0:
                logger.debug("  [%d/%d] %s/%s: %d events",
                             i + 1, len(entries), entry["speaker_id"],
                             entry["utterance_id"], len(events))

        logger.info("  Processed %d/%d entries for '%s'", total_processed, len(entries), ds_name)

    logger.info("")
    logger.info("=== Summary ===")
    logger.info("Processed: %d utterances", total_processed)
    logger.info("Events: %d total (%d breaths, %d pauses)", total_events, total_breaths, total_pauses)
    logger.info("Avg events/utterance: %.1f", total_events / max(1, total_processed))


if __name__ == "__main__":
    main()
