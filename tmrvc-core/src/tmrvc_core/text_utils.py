"""Text utilities for sentence segmentation and style inference."""

from __future__ import annotations

import re
from dataclasses import dataclass


# Abbreviations that should NOT trigger sentence splits
_EN_ABBREVS = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Jr", "Sr", "St",
    "Inc", "Ltd", "Corp", "Co", "vs", "etc", "al", "approx",
    "dept", "est", "vol", "No",
})

_EN_ABBREV_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _EN_ABBREVS) + r")\.\s*$"
)


def segment_sentences(text: str, language: str = "ja") -> list[str]:
    """Split text into sentence-level segments.

    Splitting rules:
    - ja/zh: split on punctuation marks (period/exclamation/question).
      Does NOT split on punctuation inside quotation brackets.
    - en/ko: split on .!? followed by whitespace and uppercase letter.
      Common abbreviations (Mr., Dr., etc.) are preserved.
    - Each segment is stripped; empty segments are removed.
    - Segments shorter than 2 characters are merged into the previous one.

    Args:
        text: Input text.
        language: Language code.

    Returns:
        List of non-empty sentences (at least 1 element).
    """
    text = text.strip()
    if not text:
        return [text] if text else [""]

    if language in ("ja", "zh"):
        segments = _segment_ja_zh(text)
    else:
        segments = _segment_en(text)

    # Merge very short segments into the previous one
    merged: list[str] = []
    for seg in segments:
        if merged and len(seg) < 2:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    return merged if merged else [text]


def _segment_ja_zh(text: str) -> list[str]:
    """Segment Japanese/Chinese text by sentence-ending punctuation.

    Preserves punctuation inside bracket pairs.
    """
    result: list[str] = []
    current: list[str] = []
    depth = 0  # bracket nesting depth

    for ch in text:
        current.append(ch)
        if ch in "\u300c\u300e\uff08\u3010\uff3b":  # opening brackets
            depth += 1
        elif ch in "\u300d\u300f\uff09\u3011\uff3d":  # closing brackets
            depth = max(0, depth - 1)
        elif ch in "\u3002\uff01\uff1f\n" and depth == 0:
            seg = "".join(current).strip()
            if seg:
                result.append(seg)
            current = []

    # Flush remainder
    if current:
        seg = "".join(current).strip()
        if seg:
            result.append(seg)

    return result


def _segment_en(text: str) -> list[str]:
    """Segment English/Korean text by sentence-ending punctuation."""
    # Split on .!? followed by whitespace and uppercase letter,
    # but not after known abbreviations.
    parts: list[str] = []
    last = 0

    for m in re.finditer(r"[.!?]\s+(?=[A-Z])", text):
        candidate = text[last:m.end()]
        # Check if this is an abbreviation
        prefix = text[last:m.start() + 1]
        if _EN_ABBREV_PATTERN.search(prefix):
            continue
        parts.append(candidate.strip())
        last = m.end()

    # Flush remainder
    remainder = text[last:].strip()
    if remainder:
        parts.append(remainder)

    return parts


# ---------------------------------------------------------------------------
# Per-sentence style inference (rule-based)
# ---------------------------------------------------------------------------

# Keyword → (emotion, arousal_delta, valence_delta) mappings
_JA_STYLE_KEYWORDS: dict[str, tuple[str, float, float]] = {
    # Positive
    "嬉しい": ("happy", 0.3, 0.5),
    "楽しい": ("happy", 0.3, 0.4),
    "ありがとう": ("happy", 0.1, 0.3),
    "すごい": ("excited", 0.5, 0.4),
    "やった": ("excited", 0.5, 0.5),
    "素敵": ("happy", 0.2, 0.4),
    "大好き": ("happy", 0.3, 0.6),
    "最高": ("excited", 0.4, 0.5),
    # Negative
    "悲しい": ("sad", -0.2, -0.5),
    "つらい": ("sad", -0.1, -0.4),
    "寂しい": ("sad", -0.2, -0.3),
    "ごめん": ("sad", -0.1, -0.2),
    "残念": ("sad", -0.1, -0.3),
    # Anger
    "怒": ("angry", 0.5, -0.4),
    "ふざけ": ("angry", 0.4, -0.3),
    "許さない": ("angry", 0.5, -0.5),
    # Fear/Surprise
    "怖い": ("fearful", 0.3, -0.3),
    "びっくり": ("surprised", 0.4, 0.1),
    "驚": ("surprised", 0.4, 0.1),
    "まさか": ("surprised", 0.3, 0.0),
    # Whisper
    "ひそひそ": ("whisper", -0.4, 0.0),
    "内緒": ("whisper", -0.3, 0.0),
    "秘密": ("whisper", -0.3, 0.0),
}

_EN_STYLE_KEYWORDS: dict[str, tuple[str, float, float]] = {
    "happy": ("happy", 0.3, 0.5),
    "wonderful": ("happy", 0.3, 0.5),
    "amazing": ("excited", 0.5, 0.4),
    "awesome": ("excited", 0.4, 0.4),
    "love": ("happy", 0.2, 0.5),
    "thank": ("happy", 0.1, 0.3),
    "great": ("happy", 0.2, 0.3),
    "sad": ("sad", -0.2, -0.5),
    "sorry": ("sad", -0.1, -0.2),
    "unfortunately": ("sad", -0.1, -0.3),
    "angry": ("angry", 0.5, -0.4),
    "furious": ("angry", 0.6, -0.5),
    "scared": ("fearful", 0.3, -0.3),
    "afraid": ("fearful", 0.2, -0.3),
    "wow": ("surprised", 0.4, 0.2),
    "surprise": ("surprised", 0.4, 0.1),
    "whisper": ("whisper", -0.4, 0.0),
    "quiet": ("whisper", -0.3, 0.0),
}


def infer_sentence_style(
    sentence: str,
    language: str,
    base_style: object,
) -> object:
    """Infer emotion style for a sentence based on its text content.

    Analyzes punctuation and keywords to adjust the base style.
    Returns a new StyleParams with per-sentence overrides.

    Args:
        sentence: A single sentence.
        language: Language code.
        base_style: Base StyleParams to modify.

    Returns:
        A new StyleParams instance with inferred adjustments.
    """
    from tmrvc_core.dialogue_types import StyleParams
    if not isinstance(base_style, StyleParams):
        return base_style

    # Start from base
    emotion = base_style.emotion
    arousal = base_style.arousal
    valence = base_style.valence
    energy = base_style.energy
    pitch_range = base_style.pitch_range

    # Punctuation-based adjustments
    exclamation_count = sentence.count("!") + sentence.count("\uff01")
    question_count = sentence.count("?") + sentence.count("\uff1f")

    if exclamation_count >= 2:
        arousal = min(1.0, arousal + 0.4)
        energy = min(1.0, energy + 0.3)
        if emotion == "neutral":
            emotion = "excited"
    elif exclamation_count == 1:
        arousal = min(1.0, arousal + 0.2)
        energy = min(1.0, energy + 0.1)

    if question_count > 0:
        pitch_range = min(1.0, pitch_range + 0.2)
        arousal = min(1.0, arousal + 0.1)

    # Keyword-based adjustments
    keywords = _JA_STYLE_KEYWORDS if language in ("ja", "zh") else _EN_STYLE_KEYWORDS
    text_lower = sentence.lower()
    best_match: tuple[str, float, float] | None = None
    best_priority = 0.0

    for kw, (emo, ar_delta, val_delta) in keywords.items():
        if kw in text_lower or kw in sentence:
            priority = abs(ar_delta) + abs(val_delta)
            if priority > best_priority:
                best_priority = priority
                best_match = (emo, ar_delta, val_delta)

    if best_match is not None:
        kw_emotion, ar_delta, val_delta = best_match
        if emotion == "neutral" or best_priority > 0.5:
            emotion = kw_emotion
        arousal = max(-1.0, min(1.0, arousal + ar_delta))
        valence = max(-1.0, min(1.0, valence + val_delta))

    # Ellipsis → bored/tender, lower energy
    if "..." in sentence or "\u2026" in sentence:
        energy = max(-1.0, energy - 0.2)
        if emotion == "neutral":
            emotion = "tender"

    return StyleParams(
        emotion=emotion,
        valence=valence,
        arousal=arousal,
        dominance=base_style.dominance,
        speech_rate=base_style.speech_rate,
        energy=energy,
        pitch_range=pitch_range,
        reasoning=f"auto-inferred from: {sentence[:30]}",
    )
