#!/usr/bin/env python3
"""Add phoneme annotations to dataset cache.

Converts text to phoneme_ids using OpenJTalk and estimates durations
by evenly distributing frames across phonemes.

Usage:
    uv run python scripts/add_phoneme_annotations.py \
        --cache-dir data/cache \
        --dataset custom_speaker
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import tqdm

logger = logging.getLogger(__name__)

# Phoneme vocabulary (compatible with TTS frontend)
# Based on OpenJTalk phoneme set
PHONEME_VOCAB = [
    "<pad>",
    "<unk>",
    "<sos>",
    "<eos>",
    # Vowels
    "a",
    "i",
    "u",
    "e",
    "o",
    "a:",
    "i:",
    "u:",
    "e:",
    "o:",
    # Consonants
    "k",
    "ky",
    "kw",
    "s",
    "sh",
    "sy",
    "sw",
    "t",
    "ts",
    "ty",
    "ch",
    "n",
    "ny",
    "nw",
    "h",
    "hy",
    "hw",
    "f",
    "fy",
    "fw",
    "m",
    "my",
    "mw",
    "y",
    "yw",
    "r",
    "ry",
    "rw",
    "w",
    "g",
    "gy",
    "gw",
    "z",
    "zy",
    "zw",
    "j",
    "jy",
    "d",
    "dy",
    "dw",
    "b",
    "by",
    "bw",
    "p",
    "py",
    "pw",
    "N",  # moraic nasal
    "q",  # glottal stop
    # Pause/silence
    "pau",
    "sil",
    # English (for mixed text)
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "ay",
    "eh",
    "er",
    "ey",
    "ih",
    "iy",
    "ow",
    "oy",
    "uh",
    "uw",
    "b",
    "ch",
    "d",
    "dh",
    "f",
    "g",
    "hh",
    "jh",
    "k",
    "l",
    "m",
    "n",
    "ng",
    "p",
    "r",
    "s",
    "sh",
    "t",
    "th",
    "v",
    "w",
    "y",
    "z",
    "zh",
]

PHONEME_TO_ID = {p: i for i, p in enumerate(PHONEME_VOCAB)}
PAD_ID = 0
UNK_ID = 1


def text_to_phonemes(text: str) -> list[str]:
    """Convert Japanese text to phoneme sequence using OpenJTalk."""
    try:
        import openjtalk

        # Get fullcontext labels
        labels = openjtalk.extract_fullcontext(text)

        phonemes = []
        for label in labels:
            # Extract phoneme from fullcontext label
            # Format: "xx^xx-pau+xx=xx/A:..." or "xx^xx-k+xx=xx/A:..."
            parts = label.split("-")
            if len(parts) >= 2:
                phoneme_part = parts[1].split("+")[0]
                # Map to simplified phoneme
                phoneme = map_to_simple_phoneme(phoneme_part)
                if phoneme:
                    phonemes.append(phoneme)

        return phonemes if phonemes else ["sil", "pau", "sil"]

    except Exception as e:
        logger.warning("OpenJTalk failed for '%s': %s", text[:30], e)
        # Fallback: simple character-based mapping
        return simple_text_to_phonemes(text)


def map_to_simple_phoneme(label_phoneme: str) -> str | None:
    """Map OpenJTalk phoneme label to simple phoneme."""
    # Common mappings
    mapping = {
        "pau": "pau",
        "sil": "sil",
        "cl": "q",
        "a": "a",
        "i": "i",
        "u": "u",
        "e": "e",
        "o": "o",
        "A": "a:",
        "I": "i:",
        "U": "u:",
        "E": "e:",
        "O": "o:",
        "k": "k",
        "ky": "ky",
        "kw": "kw",
        "s": "s",
        "sh": "sh",
        "sy": "sy",
        "t": "t",
        "ts": "ts",
        "ty": "ty",
        "ch": "ch",
        "n": "n",
        "ny": "ny",
        "h": "h",
        "hy": "hy",
        "f": "f",
        "fy": "fy",
        "m": "m",
        "my": "my",
        "y": "y",
        "r": "r",
        "ry": "ry",
        "w": "w",
        "g": "g",
        "gy": "gy",
        "gw": "gw",
        "z": "z",
        "zy": "zy",
        "j": "j",
        "jy": "jy",
        "d": "d",
        "dy": "dy",
        "b": "b",
        "by": "by",
        "p": "p",
        "py": "py",
        "N": "N",
    }

    return mapping.get(label_phoneme)


def simple_text_to_phonemes(text: str) -> list[str]:
    """Fallback: simple character-based phoneme mapping."""
    phonemes = ["sil"]

    for char in text:
        if char in "あア":
            phonemes.append("a")
        elif char in "いイ":
            phonemes.append("i")
        elif char in "うウ":
            phonemes.append("u")
        elif char in "えエ":
            phonemes.append("e")
        elif char in "おオ":
            phonemes.append("o")
        elif char in "かカ":
            phonemes.append("k")
            phonemes.append("a")
        elif char in "きキ":
            phonemes.append("k")
            phonemes.append("i")
        elif char in "くク":
            phonemes.append("k")
            phonemes.append("u")
        elif char in "けケ":
            phonemes.append("k")
            phonemes.append("e")
        elif char in "こコ":
            phonemes.append("k")
            phonemes.append("o")
        elif char in "がガ":
            phonemes.append("g")
            phonemes.append("a")
        elif char in "ぎギ":
            phonemes.append("g")
            phonemes.append("i")
        elif char in "ぐグ":
            phonemes.append("g")
            phonemes.append("u")
        elif char in "げゲ":
            phonemes.append("g")
            phonemes.append("e")
        elif char in "ごゴ":
            phonemes.append("g")
            phonemes.append("o")
        elif char in "さサ":
            phonemes.append("s")
            phonemes.append("a")
        elif char in "しシ":
            phonemes.append("sh")
            phonemes.append("i")
        elif char in "すス":
            phonemes.append("s")
            phonemes.append("u")
        elif char in "せセ":
            phonemes.append("s")
            phonemes.append("e")
        elif char in "そソ":
            phonemes.append("s")
            phonemes.append("o")
        elif char in "ざザ":
            phonemes.append("z")
            phonemes.append("a")
        elif char in "じジ":
            phonemes.append("j")
            phonemes.append("i")
        elif char in "ずズ":
            phonemes.append("z")
            phonemes.append("u")
        elif char in "ぜゼ":
            phonemes.append("z")
            phonemes.append("e")
        elif char in "ぞゾ":
            phonemes.append("z")
            phonemes.append("o")
        elif char in "たタ":
            phonemes.append("t")
            phonemes.append("a")
        elif char in "ちチ":
            phonemes.append("ch")
            phonemes.append("i")
        elif char in "つツ":
            phonemes.append("ts")
            phonemes.append("u")
        elif char in "てテ":
            phonemes.append("t")
            phonemes.append("e")
        elif char in "とト":
            phonemes.append("t")
            phonemes.append("o")
        elif char in "だダ":
            phonemes.append("d")
            phonemes.append("a")
        elif char in "ぢヂ":
            phonemes.append("j")
            phonemes.append("i")
        elif char in "づヅ":
            phonemes.append("z")
            phonemes.append("u")
        elif char in "でデ":
            phonemes.append("d")
            phonemes.append("e")
        elif char in "どド":
            phonemes.append("d")
            phonemes.append("o")
        elif char in "なナ":
            phonemes.append("n")
            phonemes.append("a")
        elif char in "にニ":
            phonemes.append("ny")
            phonemes.append("i")
        elif char in "ぬヌ":
            phonemes.append("n")
            phonemes.append("u")
        elif char in "ねネ":
            phonemes.append("n")
            phonemes.append("e")
        elif char in "のノ":
            phonemes.append("n")
            phonemes.append("o")
        elif char in "はハ":
            phonemes.append("h")
            phonemes.append("a")
        elif char in "ひヒ":
            phonemes.append("hy")
            phonemes.append("i")
        elif char in "ふフ":
            phonemes.append("f")
            phonemes.append("u")
        elif char in "へヘ":
            phonemes.append("h")
            phonemes.append("e")
        elif char in "ほホ":
            phonemes.append("h")
            phonemes.append("o")
        elif char in "ばバ":
            phonemes.append("b")
            phonemes.append("a")
        elif char in "びビ":
            phonemes.append("b")
            phonemes.append("i")
        elif char in "ぶブ":
            phonemes.append("b")
            phonemes.append("u")
        elif char in "べベ":
            phonemes.append("b")
            phonemes.append("e")
        elif char in "ぼボ":
            phonemes.append("b")
            phonemes.append("o")
        elif char in "ぱパ":
            phonemes.append("p")
            phonemes.append("a")
        elif char in "ぴピ":
            phonemes.append("p")
            phonemes.append("i")
        elif char in "ぷプ":
            phonemes.append("p")
            phonemes.append("u")
        elif char in "ぺペ":
            phonemes.append("p")
            phonemes.append("e")
        elif char in "ぽポ":
            phonemes.append("p")
            phonemes.append("o")
        elif char in "まマ":
            phonemes.append("m")
            phonemes.append("a")
        elif char in "みミ":
            phonemes.append("my")
            phonemes.append("i")
        elif char in "むム":
            phonemes.append("m")
            phonemes.append("u")
        elif char in "めメ":
            phonemes.append("m")
            phonemes.append("e")
        elif char in "もモ":
            phonemes.append("m")
            phonemes.append("o")
        elif char in "やヤ":
            phonemes.append("y")
            phonemes.append("a")
        elif char in "ゆユ":
            phonemes.append("y")
            phonemes.append("u")
        elif char in "よヨ":
            phonemes.append("y")
            phonemes.append("o")
        elif char in "らラ":
            phonemes.append("r")
            phonemes.append("a")
        elif char in "りリ":
            phonemes.append("r")
            phonemes.append("i")
        elif char in "るル":
            phonemes.append("r")
            phonemes.append("u")
        elif char in "れレ":
            phonemes.append("r")
            phonemes.append("e")
        elif char in "ろロ":
            phonemes.append("r")
            phonemes.append("o")
        elif char in "わワ":
            phonemes.append("w")
            phonemes.append("a")
        elif char in "をヲ":
            phonemes.append("o")
        elif char in "んン":
            phonemes.append("N")
        elif char in "、。！？":
            phonemes.append("pau")
        elif char == " ":
            pass  # Skip spaces

    phonemes.append("sil")
    return phonemes


def phonemes_to_ids(phonemes: list[str]) -> list[int]:
    """Convert phoneme list to IDs."""
    return [PHONEME_TO_ID.get(p, UNK_ID) for p in phonemes]


def estimate_durations(n_phonemes: int, n_frames: int) -> list[int]:
    """Estimate durations by distributing frames across phonemes.

    Simple heuristic: evenly distribute with remainder going to last phoneme.
    """
    if n_phonemes == 0:
        return []

    base_duration = n_frames // n_phonemes
    remainder = n_frames % n_phonemes

    durations = [base_duration] * n_phonemes
    # Add remainder to middle phonemes (vowels usually longer)
    for i in range(remainder):
        idx = (n_phonemes // 2) + i
        if idx < n_phonemes:
            durations[idx] += 1

    return durations


def main():
    parser = argparse.ArgumentParser(description="Add phoneme annotations")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache_root = args.cache_dir / args.dataset / args.split
    if not cache_root.exists():
        logger.error("Cache root not found: %s", cache_root)
        return 1

    meta_files = list(cache_root.rglob("meta.json"))
    logger.info("Found %d meta files", len(meta_files))

    updated = 0
    errors = 0

    for meta_path in tqdm.tqdm(meta_files, desc="Annotating"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))

            if args.skip_existing and "phoneme_ids" in meta:
                continue

            text = meta.get("text", "")
            if not text:
                errors += 1
                continue

            n_frames = meta.get("n_frames", 0)

            # Convert text to phonemes
            phonemes = text_to_phonemes(text)
            phoneme_ids = phonemes_to_ids(phonemes)

            # Estimate durations
            durations = estimate_durations(len(phoneme_ids), n_frames)

            # Save
            meta["phonemes"] = phonemes
            meta["phoneme_ids"] = phoneme_ids
            meta["duration"] = durations
            meta["n_phonemes"] = len(phoneme_ids)

            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            updated += 1

        except Exception as e:
            logger.warning("Error: %s: %s", meta_path, e)
            errors += 1

    logger.info("Done. Updated %d, errors %d", updated, errors)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
