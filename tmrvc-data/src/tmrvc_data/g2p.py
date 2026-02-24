"""Grapheme-to-phoneme frontend for Japanese and English.

Converts text to a unified IPA-based phoneme sequence with language ID.
Japanese uses pyopenjtalk, English uses phonemizer (espeak-ng backend).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

# --- Unified phoneme vocabulary ---
# Based on IPA with pause/breath tokens.  Index 0 = <pad>, 1 = <unk>.

_SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<sil>", "<breath>"]

# Japanese phonemes (pyopenjtalk fullcontext → phoneme)
_JA_VOWELS = ["a", "i", "u", "e", "o"]
_JA_CONSONANTS = [
    "k", "s", "t", "n", "h", "m", "y", "r", "w", "g", "z", "d", "b", "p",
    "ky", "sy", "ty", "ny", "hy", "my", "ry", "gy", "zy", "dy", "by", "py",
    "ts", "ch", "sh", "f", "j", "v",
]
_JA_SPECIAL = ["N", "cl", "pau"]  # moraic nasal, geminate, pause

# English phonemes (ARPAbet-style via espeak / CMU)
_EN_VOWELS = [
    "iː", "ɪ", "eɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "oʊ", "ʊ", "uː",
    "ʌ", "ə", "ɜː", "aɪ", "aʊ", "ɔɪ",
]
_EN_CONSONANTS = [
    "p", "b", "t", "d", "k", "g", "f", "v", "θ", "ð", "s", "z",
    "ʃ", "ʒ", "h", "m", "n", "ŋ", "l", "r", "w", "j",
    "tʃ", "dʒ",
]

# Shared suprasegmentals
_PROSODY = ["ˈ", "ˌ", "ː", "̃"]

# Build full vocabulary
_ALL_PHONES = (
    _SPECIAL
    + sorted(set(
        _JA_VOWELS + _JA_CONSONANTS + _JA_SPECIAL
        + _EN_VOWELS + _EN_CONSONANTS + _PROSODY
    ))
)

# Deduplicate while preserving order
_seen: set[str] = set()
PHONEME_LIST: list[str] = []
for _p in _ALL_PHONES:
    if _p not in _seen:
        PHONEME_LIST.append(_p)
        _seen.add(_p)

PHONE2ID: dict[str, int] = {p: i for i, p in enumerate(PHONEME_LIST)}
ID2PHONE: dict[int, str] = {i: p for i, p in enumerate(PHONEME_LIST)}

PAD_ID = PHONE2ID["<pad>"]
UNK_ID = PHONE2ID["<unk>"]
BOS_ID = PHONE2ID["<bos>"]
EOS_ID = PHONE2ID["<eos>"]
SIL_ID = PHONE2ID["<sil>"]

# Language IDs
LANG_JA = 0
LANG_EN = 1
LANG_ZH = 2
LANG_OTHER = 3


@dataclass
class G2PResult:
    """Result of grapheme-to-phoneme conversion."""

    phoneme_ids: torch.Tensor  # [L] int64
    phonemes: list[str]  # Human-readable phoneme list
    language_id: int


def _g2p_japanese(text: str) -> list[str]:
    """Convert Japanese text to phoneme list using pyopenjtalk."""
    try:
        import pyopenjtalk
    except ImportError:
        raise ImportError(
            "pyopenjtalk is required for Japanese G2P. "
            "Install with: pip install pyopenjtalk"
        )

    # Get fullcontext labels
    labels = pyopenjtalk.extract_fullcontext(text)
    phonemes: list[str] = []

    for label in labels:
        # Extract phoneme from fullcontext label
        # Format: p1^p2-p3+p4=p5/.../...
        match = re.match(r"[^-]*-([^+]+)\+", label)
        if match:
            phone = match.group(1)
            if phone == "xx":
                continue
            if phone == "pau":
                phonemes.append("pau")
            elif phone == "cl":
                phonemes.append("cl")
            else:
                phonemes.append(phone)

    return phonemes


def _g2p_english(text: str) -> list[str]:
    """Convert English text to phoneme list using phonemizer."""
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
    except ImportError:
        raise ImportError(
            "phonemizer is required for English G2P. "
            "Install with: pip install phonemizer"
        )

    # Use espeak-ng backend with IPA output
    result = phonemize(
        text,
        language="en-us",
        backend="espeak",
        separator=Separator(phone=" ", word=" <sil> ", syllable=""),
        strip=True,
        preserve_punctuation=False,
    )

    phonemes = [p for p in result.split() if p]
    return phonemes


def text_to_phonemes(
    text: str,
    language: str = "ja",
) -> G2PResult:
    """Convert text to phoneme IDs.

    Args:
        text: Input text string.
        language: Language code ("ja", "en").

    Returns:
        G2PResult with phoneme IDs tensor and metadata.
    """
    if language == "ja":
        phonemes = _g2p_japanese(text)
        lang_id = LANG_JA
    elif language == "en":
        phonemes = _g2p_english(text)
        lang_id = LANG_EN
    else:
        raise ValueError(f"Unsupported language: {language}")

    # Add BOS/EOS
    phonemes = ["<bos>"] + phonemes + ["<eos>"]

    # Convert to IDs
    ids = [PHONE2ID.get(p, UNK_ID) for p in phonemes]

    return G2PResult(
        phoneme_ids=torch.tensor(ids, dtype=torch.long),
        phonemes=phonemes,
        language_id=lang_id,
    )
