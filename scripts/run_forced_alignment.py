#!/usr/bin/env python3
"""Run forced alignment on cached datasets and save phoneme_ids/durations.

Processes each utterance in the feature cache:
1. Loads text from meta.json
2. Converts text → phonemes via G2P
3. Saves phoneme_ids.npy and durations.npy alongside existing features

For datasets without MFA alignment, uses a heuristic equal-duration split.
For datasets with MFA TextGrid files, loads actual alignments.

Usage::

    python scripts/run_forced_alignment.py --cache-dir data/cache --dataset jsut --language ja
    python scripts/run_forced_alignment.py --cache-dir data/cache --dataset ljspeech --language en
    python scripts/run_forced_alignment.py --cache-dir data/cache --dataset vctk --language en --textgrid-dir data/alignments/vctk
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run forced alignment and save phoneme_ids/durations to cache.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Feature cache directory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name in cache (e.g. 'jsut', 'ljspeech').",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (default: train).",
    )
    parser.add_argument(
        "--language",
        required=True,
        choices=["ja", "en"],
        help="Language for G2P.",
    )
    parser.add_argument(
        "--textgrid-dir",
        type=Path,
        default=None,
        help="Directory of MFA TextGrid files. If provided, uses actual alignments.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing phoneme_ids/durations.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    return parser


def _equal_duration_split(n_phonemes: int, n_frames: int) -> np.ndarray:
    """Heuristic: divide frames equally across phonemes."""
    if n_phonemes <= 0:
        return np.array([], dtype=np.int64)
    base = n_frames // n_phonemes
    remainder = n_frames % n_phonemes
    durations = np.full(n_phonemes, base, dtype=np.int64)
    durations[:remainder] += 1
    return durations


def process_utterance_g2p(
    utt_dir: Path,
    language: str,
    overwrite: bool = False,
) -> bool:
    """Process a single utterance: G2P → phoneme_ids + heuristic durations.

    Returns True if processed, False if skipped.
    """
    phone_path = utt_dir / "phoneme_ids.npy"
    dur_path = utt_dir / "durations.npy"

    if phone_path.exists() and dur_path.exists() and not overwrite:
        return False

    meta_path = utt_dir / "meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    text = meta.get("text", "")
    if not text:
        return False

    n_frames = meta.get("n_frames", 0)
    if n_frames <= 0:
        return False

    from tmrvc_data.g2p import text_to_phonemes, LANG_JA, LANG_EN

    result = text_to_phonemes(text, language=language)
    phoneme_ids = result.phoneme_ids.numpy()
    durations = _equal_duration_split(len(phoneme_ids), n_frames)

    np.save(phone_path, phoneme_ids)
    np.save(dur_path, durations)

    # Update meta with language_id
    meta["language_id"] = result.language_id
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return True


def process_utterance_textgrid(
    utt_dir: Path,
    textgrid_path: Path,
    language: str,
    overwrite: bool = False,
) -> bool:
    """Process a single utterance using MFA TextGrid alignment.

    Returns True if processed, False if skipped.
    """
    phone_path = utt_dir / "phoneme_ids.npy"
    dur_path = utt_dir / "durations.npy"

    if phone_path.exists() and dur_path.exists() and not overwrite:
        return False

    if not textgrid_path.exists():
        return False

    meta_path = utt_dir / "meta.json"
    if not meta_path.exists():
        return False

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    n_frames = meta.get("n_frames", 0)

    from tmrvc_data.alignment import load_textgrid_durations
    from tmrvc_data.g2p import BOS_ID, EOS_ID, PHONE2ID, UNK_ID, LANG_JA, LANG_EN

    alignment = load_textgrid_durations(textgrid_path, total_frames=n_frames)

    # Convert MFA phonemes to our vocabulary IDs
    # Wrap with BOS/EOS
    ids = [BOS_ID]
    durs = [0]  # BOS has 0 duration
    for phone, dur in zip(alignment.phonemes, alignment.durations):
        ids.append(PHONE2ID.get(phone, UNK_ID))
        durs.append(int(dur))
    ids.append(EOS_ID)
    durs.append(0)  # EOS has 0 duration

    phoneme_ids = np.array(ids, dtype=np.int64)
    durations = np.array(durs, dtype=np.int64)

    np.save(phone_path, phoneme_ids)
    np.save(dur_path, durations)

    # Update meta
    lang_id = LANG_JA if language == "ja" else LANG_EN
    meta["language_id"] = lang_id
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return True


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from tmrvc_data.cache import FeatureCache

    cache = FeatureCache(args.cache_dir)
    entries = cache.iter_entries(args.dataset, args.split)

    if not entries:
        logger.error("No entries found for dataset '%s' split '%s'", args.dataset, args.split)
        return

    logger.info("Processing %d entries for '%s/%s'", len(entries), args.dataset, args.split)

    processed = 0
    skipped = 0

    for entry in entries:
        utt_dir = cache._utt_dir(
            args.dataset, args.split, entry["speaker_id"], entry["utterance_id"],
        )

        if args.textgrid_dir:
            # Try to find matching TextGrid
            tg_path = args.textgrid_dir / entry["speaker_id"] / f"{entry['utterance_id']}.TextGrid"
            if not tg_path.exists():
                tg_path = args.textgrid_dir / f"{entry['utterance_id']}.TextGrid"
            ok = process_utterance_textgrid(
                utt_dir, tg_path, args.language, args.overwrite,
            )
        else:
            ok = process_utterance_g2p(utt_dir, args.language, args.overwrite)

        if ok:
            processed += 1
        else:
            skipped += 1

    logger.info("Done: %d processed, %d skipped", processed, skipped)


if __name__ == "__main__":
    main()
