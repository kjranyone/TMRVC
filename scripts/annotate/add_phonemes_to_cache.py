#!/usr/bin/env python3
"""Add phoneme_ids and durations to existing cache.

For JVS dataset: reads lab files (monophone alignments).
For VCTK dataset: uses G2P and uniform duration.

Usage:
    uv run python scripts/add_phonemes_to_cache.py \
        --cache-dir data/cache \
        --dataset jvs \
        --raw-dir data/raw/jvs_corpus/jvs_ver1
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from tmrvc_data.g2p import text_to_phonemes, PHONE2ID

logger = logging.getLogger(__name__)


def parse_lab_file(lab_path: Path) -> tuple[list[str], list[float]]:
    """Parse HTK-style lab file.

    Returns:
        phonemes: List of phoneme symbols
        durations: List of durations in frames (at 100fps)
    """
    phonemes = []
    durations = []

    with open(lab_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                start = float(parts[0])
                end = float(parts[1])
                phone = parts[2]

                # Skip silence at edges
                if phone == "sil":
                    continue

                phonemes.append(phone)

                # Convert seconds to frames (100fps = 10ms hop)
                duration_frames = int((end - start) * 100)
                durations.append(duration_frames)

    return phonemes, durations


def add_phonemes_jvs(
    cache_dir: Path,
    raw_dir: Path,
    overwrite: bool = False,
) -> int:
    """Add phoneme_ids and durations to JVS cache."""
    count = 0

    # Find all speaker directories in cache
    for speaker_dir in sorted(cache_dir.glob("jvs_*")):
        if not speaker_dir.is_dir():
            continue

        # Extract speaker ID (jvs001, jvs002, etc.)
        speaker_id = speaker_dir.name

        # Find corresponding raw directory
        raw_speaker_dir = raw_dir / speaker_id / "parallel100"
        if not raw_speaker_dir.exists():
            logger.warning("Raw speaker dir not found: %s", raw_speaker_dir)
            continue

        # Read transcripts
        transcripts = {}
        transcript_file = raw_speaker_dir / "transcripts_utf8.txt"
        if transcript_file.exists():
            with open(transcript_file) as f:
                for line in f:
                    if ":" in line:
                        utt_id, text = line.strip().split(":", 1)
                        transcripts[utt_id] = text

        # Process each utterance in cache
        for utt_dir in sorted(speaker_dir.iterdir()):
            if not utt_dir.is_dir():
                continue

            utt_id = utt_dir.name
            meta_path = utt_dir / "meta.json"
            phoneme_path = utt_dir / "phoneme_ids.npy"
            durations_path = utt_dir / "durations.npy"

            # Skip if already exists
            if not overwrite and phoneme_path.exists() and durations_path.exists():
                continue

            # Read meta.json to get source info
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                original_utt_id = meta.get("original_utt_id", utt_id)
            else:
                original_utt_id = utt_id

            # Try to read lab file
            lab_path = raw_speaker_dir / "lab" / "mon" / f"{original_utt_id}.lab"

            if lab_path.exists():
                phonemes, durations = parse_lab_file(lab_path)
                phoneme_ids = [PHONE2ID.get(p, 1) for p in phonemes]
            else:
                # Fall back to G2P
                text = transcripts.get(original_utt_id, "")
                if not text:
                    logger.warning("No transcript for %s", utt_id)
                    continue
                result = text_to_phonemes(text, language="ja")
                phoneme_ids = result.phoneme_ids.tolist()
                # Uniform duration
                durations = [10] * len(phoneme_ids)

            # Save
            np.save(phoneme_path, np.array(phoneme_ids, dtype=np.int64))
            np.save(durations_path, np.array(durations, dtype=np.int64))

            # Update meta.json
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {}
            meta["n_phonemes"] = len(phoneme_ids)
            meta["has_phonemes"] = True
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            count += 1
            if count % 100 == 0:
                logger.info("Processed %d utterances", count)

    return count


def add_phonemes_vctk(
    cache_dir: Path,
    raw_dir: Path,
    overwrite: bool = False,
) -> int:
    """Add phoneme_ids and durations to VCTK cache."""
    count = 0

    # Read VCTK transcripts
    transcript_file = raw_dir / "txt" / "p225" / "p225_001.txt"
    if not transcript_file.exists():
        logger.warning("VCTK transcript structure not found")
        return 0

    # Build transcript lookup
    transcripts = {}
    for speaker_dir in (raw_dir / "txt").iterdir():
        if speaker_dir.is_dir():
            for txt_file in speaker_dir.glob("*.txt"):
                utt_id = txt_file.stem
                with open(txt_file) as f:
                    transcripts[utt_id] = f.read().strip()

    # Process cache - check both train/ and root
    cache_root = cache_dir / "train" if (cache_dir / "train").exists() else cache_dir

    for speaker_dir in sorted(cache_root.glob("vctk_p*")):
        if not speaker_dir.is_dir():
            continue

        for utt_dir in sorted(speaker_dir.iterdir()):
            if not utt_dir.is_dir():
                continue

            utt_id = utt_dir.name
            meta_path = utt_dir / "meta.json"
            phoneme_path = utt_dir / "phoneme_ids.npy"
            durations_path = utt_dir / "durations.npy"

            if not overwrite and phoneme_path.exists():
                continue

            # Get transcript - VCTK utterance ID format is vctk_pXXX_YYY
            # Text files are named pXXX_YYY.txt
            text_key = utt_id.replace("vctk_", "")
            text = transcripts.get(text_key)
            if not text:
                logger.debug("No transcript for %s", utt_id)
                continue

            # Phonemize
            result = text_to_phonemes(text, language="en")
            phoneme_ids = result.phoneme_ids.tolist()

            # Read audio length from meta
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                n_frames = meta.get("n_frames", 100)
            else:
                n_frames = 100

            # Uniform duration
            duration_per_phone = max(1, n_frames // len(phoneme_ids))
            durations = [duration_per_phone] * len(phoneme_ids)

            # Adjust last phone to match total (ensure non-negative)
            remaining = n_frames - sum(durations[:-1])
            durations[-1] = max(1, remaining)

            # Save
            np.save(phoneme_path, np.array(phoneme_ids, dtype=np.int64))
            np.save(durations_path, np.array(durations, dtype=np.int64))

            # Update meta
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
            else:
                meta = {}
            meta["n_phonemes"] = len(phoneme_ids)
            meta["has_phonemes"] = True
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            count += 1
            if count % 100 == 0:
                logger.info("Processed %d utterances", count)

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=str, default="data/cache")
    parser.add_argument("--dataset", type=str, required=True, choices=["jvs", "vctk"])
    parser.add_argument("--raw-dir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    cache_dir = Path(args.cache_dir)
    raw_dir = Path(args.raw_dir)

    count = 0
    if args.dataset == "jvs":
        count = add_phonemes_jvs(cache_dir, raw_dir, args.overwrite)
    elif args.dataset == "vctk":
        count = add_phonemes_vctk(cache_dir, raw_dir, args.overwrite)

    logger.info("Added phonemes to %d utterances", count)


if __name__ == "__main__":
    main()
