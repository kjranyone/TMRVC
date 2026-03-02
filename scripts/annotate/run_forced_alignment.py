#!/usr/bin/env python3
"""Run forced alignment on cached datasets and save text unit IDs/durations.

Processes each utterance in the feature cache:
1. Loads text from meta.json
2. Converts text → phonemes via G2P
3. Saves phoneme_ids.npy and durations.npy alongside existing features

MFA Integration:
Loads actual alignments from Montreal Forced Aligner (MFA) TextGrid files.
Maintains strict frame parity with UCLM v2 (100Hz).

Usage::

    # Strict MFA mode (Recommended)
    python scripts/annotate/run_forced_alignment.py \
        --cache-dir data/cache --dataset vctk --language en \
        --textgrid-dir data/alignments/vctk

    # Heuristic mode (Fallback only)
    python scripts/annotate/run_forced_alignment.py \
        --cache-dir data/cache --dataset ljspeech --language en \
        --allow-heuristic
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

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
        choices=["ja", "en", "zh", "ko"],
        help="Language for text frontend.",
    )
    parser.add_argument(
        "--frontend",
        choices=["phoneme", "tokenizer"],
        default="phoneme",
        help="Text frontend mode (default: phoneme).",
    )
    parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Allow equal-duration heuristic if TextGrid is missing (NOT recommended).",
    )
    parser.add_argument(
        "--textgrid-dir",
        type=Path,
        default=None,
        help="Directory of MFA TextGrid files. Required for accurate TTS.",
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
    allow_heuristic: bool = False,
) -> bool:
    """Process a single utterance: G2P → phoneme_ids + heuristic durations."""
    if not allow_heuristic:
        return False

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

    from tmrvc_data.g2p import text_to_phonemes

    result = text_to_phonemes(text, language=language)
    phoneme_ids = result.phoneme_ids.numpy()
    durations = _equal_duration_split(len(phoneme_ids), n_frames)

    np.save(phone_path, phoneme_ids)
    np.save(dur_path, durations)

    # Update meta
    meta["language_id"] = result.language_id
    meta["text_frontend"] = "phoneme"
    meta["alignment_type"] = "heuristic"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return True


def process_utterance_textgrid(
    utt_dir: Path,
    textgrid_path: Path,
    language: str,
    overwrite: bool = False,
) -> bool:
    """Process a single utterance using MFA TextGrid alignment."""
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
    if n_frames <= 0:
        return False

    from tmrvc_data.alignment import load_textgrid_durations
    from tmrvc_data.g2p import BOS_ID, EOS_ID, PHONE2ID, UNK_ID

    try:
        alignment = load_textgrid_durations(textgrid_path, total_frames=n_frames)
    except Exception as e:
        logger.error("Failed to parse TextGrid %s: %s", textgrid_path, e)
        return False

    # Convert MFA phonemes to our vocabulary IDs
    # Wrap with BOS/EOS
    ids = [BOS_ID]
    durs = [0]  # BOS/EOS have 0 duration in our UCLM v2 spec
    for phone, dur in zip(alignment.phonemes, alignment.durations):
        # MFA sometimes uses different labels for silence
        if phone in ("sil", "sp", ""):
            p_id = PHONE2ID.get("<sil>", UNK_ID)
        else:
            p_id = PHONE2ID.get(phone, UNK_ID)
            if p_id == UNK_ID:
                logger.warning("Unknown phoneme '%s' in %s", phone, textgrid_path.name)
        
        ids.append(p_id)
        durs.append(int(dur))
    
    ids.append(EOS_ID)
    durs.append(0)

    phoneme_ids = np.array(ids, dtype=np.int64)
    durations = np.array(durs, dtype=np.int64)

    # Final sanity check: durations.sum() must equal n_frames
    if durations.sum() != n_frames:
        logger.error("Frame parity error in %s: sum=%d, expected=%d", 
                     utt_dir.name, durations.sum(), n_frames)
        return False

    np.save(phone_path, phoneme_ids)
    np.save(dur_path, durations)

    # Update meta
    lang_map = {"ja": 0, "en": 1, "zh": 2, "ko": 3}
    meta["language_id"] = lang_map.get(language, 0)
    meta["text_frontend"] = "phoneme"
    meta["alignment_type"] = "mfa"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

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

        ok = False
        if args.textgrid_dir:
            # Try to find matching TextGrid: speaker/utt.TextGrid or utt.TextGrid
            tg_path = args.textgrid_dir / entry["speaker_id"] / f"{entry['utterance_id']}.TextGrid"
            if not tg_path.exists():
                tg_path = args.textgrid_dir / f"{entry['utterance_id']}.TextGrid"
            
            if tg_path.exists():
                ok = process_utterance_textgrid(utt_dir, tg_path, args.language, args.overwrite)
            elif args.allow_heuristic:
                ok = process_utterance_g2p(utt_dir, args.language, args.overwrite, allow_heuristic=True)
            else:
                logger.error("TextGrid missing for %s. Use --allow-heuristic if needed.", entry["utterance_id"])
        else:
            if not args.allow_heuristic:
                logger.error("No --textgrid-dir provided and --allow-heuristic is disabled. Cannot process.")
                break
            ok = process_utterance_g2p(utt_dir, args.language, args.overwrite, allow_heuristic=True)

        if ok:
            processed += 1
        else:
            skipped += 1

    logger.info("Done: %d processed, %d skipped", processed, skipped)


if __name__ == "__main__":
    main()
