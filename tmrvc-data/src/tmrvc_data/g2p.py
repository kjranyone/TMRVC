"""Grapheme-to-phoneme frontend for Japanese, English, Chinese, and Korean.

Converts text to a unified IPA-based phoneme sequence with language ID.

Backend policy:
- Japanese: prefer ``pyopenjtalk`` -> fallback ``phonemizer`` (espeak ja) ->
  final grapheme fallback (degraded quality, no hard failure).
- English/Chinese/Korean: use ``phonemizer`` (espeak backend).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_PAUSE_CHARS = set(" \t\r\n、。，．,.!?！？…・「」『』（）()[]{}<>")

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

# Mandarin Chinese IPA (common subset used by espeak outputs)
_ZH_VOWELS = [
    "a", "ɑ", "e", "ə", "ɤ", "i", "u", "y", "o", "ɚ",
]
_ZH_CONSONANTS = [
    "p", "pʰ", "t", "tʰ", "k", "kʰ", "m", "n", "ŋ", "f",
    "s", "ʂ", "ɕ", "ʐ", "ɻ", "l", "tɕ", "tɕʰ", "ts", "tsʰ", "ʈʂ", "ʈʂʰ",
]
_ZH_TONES = ["˥", "˦", "˧", "˨", "˩", "1", "2", "3", "4", "5"]

# Korean IPA (common subset used by espeak outputs)
_KO_VOWELS = [
    "a", "e", "i", "o", "u", "ɯ", "ʌ", "ɛ", "ø", "y", "ɐ",
]
_KO_CONSONANTS = [
    "p", "pʰ", "p͈", "t", "tʰ", "t͈", "k", "kʰ", "k͈",
    "s", "s͈", "h", "m", "n", "ŋ", "l", "ɾ", "tɕ", "tɕʰ", "tɕ͈", "j", "w",
]

# Shared suprasegmentals
_PROSODY = ["ˈ", "ˌ", "ː", "̃"]

# Build full vocabulary
_ALL_PHONES = (
    _SPECIAL
    + sorted(set(
        _JA_VOWELS + _JA_CONSONANTS + _JA_SPECIAL
        + _EN_VOWELS + _EN_CONSONANTS
        + _ZH_VOWELS + _ZH_CONSONANTS + _ZH_TONES
        + _KO_VOWELS + _KO_CONSONANTS
        + _PROSODY
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
LANG_KO = 3


@dataclass
class G2PResult:
    """Result of grapheme-to-phoneme conversion."""

    phoneme_ids: torch.Tensor  # [L] int64
    phonemes: list[str]  # Human-readable phoneme list
    language_id: int


def _g2p_japanese(text: str) -> list[str]:
    """Convert Japanese text to phoneme list.

    Preference order:
    1) pyopenjtalk fullcontext
    2) phonemizer espeak ja backend
    3) grapheme fallback (degraded, but avoids runtime crash)
    """
    try:
        import pyopenjtalk
    except ImportError:
        logger.info("pyopenjtalk is unavailable; trying phonemizer fallback for Japanese")
    else:
        try:
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
            if phonemes:
                return phonemes
            logger.warning("pyopenjtalk produced no phonemes; trying phonemizer fallback")
        except Exception as e:  # pragma: no cover - backend-dependent
            logger.warning("pyopenjtalk failed; trying phonemizer fallback: %s", e)

    try:
        phonemes = _g2p_phonemizer(text, ["ja", "ja-jp"])
        if phonemes:
            return phonemes
    except Exception as e:  # pragma: no cover - backend-dependent
        logger.warning("phonemizer Japanese fallback failed; using grapheme fallback: %s", e)

    return _g2p_grapheme_fallback(text)


def _g2p_phonemizer(text: str, language_options: list[str]) -> list[str]:
    """Convert text to phoneme list using phonemizer/espeak.

    Tries language codes in order and returns the first successful result.
    """
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
    except ImportError:
        raise ImportError(
            "phonemizer is required for this G2P backend. "
            "Install with: pip install phonemizer"
        )

    last_error: Exception | None = None
    for lang in language_options:
        try:
            result = phonemize(
                text,
                language=lang,
                backend="espeak",
                separator=Separator(phone=" ", word=" <sil> ", syllable=""),
                strip=True,
                preserve_punctuation=False,
            )
            phonemes = [p for p in result.split() if p]
            if phonemes:
                return phonemes
        except Exception as e:  # pragma: no cover - backend-dependent
            last_error = e
            logger.debug("phonemizer failed for language=%s: %s", lang, e)

    if last_error is not None:
        raise RuntimeError(
            f"phonemizer failed for languages={language_options}: {last_error}"
        )
    return []


def _g2p_grapheme_fallback(text: str) -> list[str]:
    """Best-effort fallback when no G2P backend is available.

    This preserves runtime availability at reduced quality.
    """
    if not text:
        return ["<sil>"]

    phones: list[str] = []
    for ch in text:
        if ch in _PAUSE_CHARS:
            phones.append("<sil>")
            continue
        lo = ch.lower()
        if lo in PHONE2ID:
            phones.append(lo)
        elif ch in PHONE2ID:
            phones.append(ch)
        else:
            # Use a stable known phoneme instead of <unk>-only collapse.
            phones.append("a")

    # Collapse duplicate silences to keep durations reasonable.
    dedup: list[str] = []
    for p in phones:
        if p == "<sil>" and dedup and dedup[-1] == "<sil>":
            continue
        dedup.append(p)
    return dedup if dedup else ["<sil>"]


def _g2p_english(text: str) -> list[str]:
    """Convert English text to phoneme list using phonemizer."""
    return _g2p_phonemizer(text, ["en-us", "en"])


def _g2p_chinese(text: str) -> list[str]:
    """Convert Chinese text to phoneme list using phonemizer."""
    return _g2p_phonemizer(text, ["cmn", "zh", "zh-cn"])


def _g2p_korean(text: str) -> list[str]:
    """Convert Korean text to phoneme list using phonemizer."""
    return _g2p_phonemizer(text, ["ko", "ko-kr"])


def text_to_phonemes(
    text: str,
    language: str = "ja",
) -> G2PResult:
    """Convert text to phoneme IDs.

    Args:
        text: Input text string.
        language: Language code ("ja", "en", "zh", "ko").

    Returns:
        G2PResult with phoneme IDs tensor and metadata.
    """
    if language == "ja":
        phonemes = _g2p_japanese(text)
        lang_id = LANG_JA
    elif language == "en":
        phonemes = _g2p_english(text)
        lang_id = LANG_EN
    elif language == "zh":
        phonemes = _g2p_chinese(text)
        lang_id = LANG_ZH
    elif language == "ko":
        phonemes = _g2p_korean(text)
        lang_id = LANG_KO
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
