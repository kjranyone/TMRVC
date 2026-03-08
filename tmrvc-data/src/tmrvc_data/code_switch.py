"""Code-switch detection for multilingual text.

Detects language switches within text using Unicode script analysis and
simple heuristics. Useful for:
    - Identifying mixed-language utterances in training data
    - Computing code_switch_coverage for dataset reports
    - Guiding G2P backend selection at language boundaries
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class CodeSwitchSpan:
    """A detected language span within text."""

    start_char: int
    end_char: int
    language: str
    confidence: float


# ---------------------------------------------------------------------------
# Unicode script detection
# ---------------------------------------------------------------------------

# CJK Unified Ideographs ranges
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0x2A700, 0x2B73F),  # CJK Unified Ideographs Extension C
    (0x2B740, 0x2B81F),  # CJK Unified Ideographs Extension D
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
]

# Hiragana / Katakana ranges
_HIRAGANA_RANGE = (0x3040, 0x309F)
_KATAKANA_RANGE = (0x30A0, 0x30FF)
_KATAKANA_EXT_RANGE = (0x31F0, 0x31FF)
_HALFWIDTH_KATAKANA_RANGE = (0xFF65, 0xFF9F)

# Hangul ranges
_HANGUL_RANGES = [
    (0xAC00, 0xD7AF),   # Hangul Syllables
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x3130, 0x318F),   # Hangul Compatibility Jamo
    (0xA960, 0xA97F),   # Hangul Jamo Extended-A
    (0xD7B0, 0xD7FF),   # Hangul Jamo Extended-B
]

# Latin (English and other Latin-script languages)
_LATIN_RANGES = [
    (0x0041, 0x005A),   # A-Z
    (0x0061, 0x007A),   # a-z
    (0x00C0, 0x00FF),   # Latin Extended (accented chars)
    (0x0100, 0x024F),   # Latin Extended-A/B
]


def _in_ranges(cp: int, ranges: list[tuple[int, int]]) -> bool:
    """Check if codepoint falls within any of the given ranges."""
    for lo, hi in ranges:
        if lo <= cp <= hi:
            return True
    return False


def _classify_char(ch: str) -> str | None:
    """Classify a single character's script/language.

    Returns:
        Language code ("ja", "zh", "ko", "en") or None for
        punctuation/whitespace/unclassifiable characters.
    """
    cp = ord(ch)

    # Japanese-specific scripts (Hiragana/Katakana are unambiguously Japanese)
    if _HIRAGANA_RANGE[0] <= cp <= _HIRAGANA_RANGE[1]:
        return "ja"
    if _KATAKANA_RANGE[0] <= cp <= _KATAKANA_RANGE[1]:
        return "ja"
    if _KATAKANA_EXT_RANGE[0] <= cp <= _KATAKANA_EXT_RANGE[1]:
        return "ja"
    if _HALFWIDTH_KATAKANA_RANGE[0] <= cp <= _HALFWIDTH_KATAKANA_RANGE[1]:
        return "ja"

    # Korean Hangul
    if _in_ranges(cp, _HANGUL_RANGES):
        return "ko"

    # CJK Ideographs (shared between zh/ja/ko - we default to "zh"
    # but context from surrounding characters can override this)
    if _in_ranges(cp, _CJK_RANGES):
        return "cjk"  # ambiguous, resolved later

    # Latin script -> "en" (could be other Latin-script languages,
    # but for TMRVC's ja/en/zh/ko scope this is sufficient)
    if _in_ranges(cp, _LATIN_RANGES):
        return "en"

    # Numbers, punctuation, whitespace -> no language assignment
    return None


def _resolve_cjk_ambiguity(
    char_langs: list[str | None],
    text: str,
    primary_language: str,
) -> list[str | None]:
    """Resolve "cjk" labels to a specific language (zh, ja, ko).

    Heuristic rules:
    1. If surrounded by Japanese-specific chars (hiragana/katakana), assign "ja".
    2. If surrounded by Hangul, assign "ko" (rare in practice).
    3. Otherwise, assign based on primary_language if it's zh/ja/ko.
    4. Final fallback: "zh" (CJK ideographs are most commonly Chinese).
    """
    resolved = list(char_langs)
    n = len(resolved)

    for i in range(n):
        if resolved[i] != "cjk":
            continue

        # Look at neighboring classified characters
        prev_lang = None
        next_lang = None
        for j in range(i - 1, -1, -1):
            if resolved[j] is not None and resolved[j] != "cjk":
                prev_lang = resolved[j]
                break
        for j in range(i + 1, n):
            if resolved[j] is not None and resolved[j] != "cjk":
                next_lang = resolved[j]
                break

        # Rule 1: surrounded by Japanese
        if prev_lang == "ja" or next_lang == "ja":
            resolved[i] = "ja"
        # Rule 2: surrounded by Korean
        elif prev_lang == "ko" or next_lang == "ko":
            resolved[i] = "ko"
        # Rule 3: use primary language if CJK-capable
        elif primary_language in ("zh", "ja", "ko"):
            resolved[i] = primary_language
        # Rule 4: default to Chinese
        else:
            resolved[i] = "zh"

    return resolved


def detect_code_switch_spans(
    text: str,
    primary_language: str = "ja",
) -> list[dict]:
    """Detect language switches within text.

    Identifies contiguous spans of each language and returns them as a list
    of dictionaries.

    Args:
        text: Input text string to analyse.
        primary_language: Expected primary language of the text ("ja", "en", "zh", "ko").
            Used to disambiguate CJK ideographs.

    Returns:
        List of dicts, each with keys:
            - start_char: int, start character index (inclusive)
            - end_char: int, end character index (exclusive)
            - language: str, detected language code
            - confidence: float, detection confidence (0.0 to 1.0)

        Empty list if text is empty or contains only punctuation/whitespace.

    Examples:
        >>> detect_code_switch_spans("Hello World", "en")
        [{'start_char': 0, 'end_char': 11, 'language': 'en', 'confidence': 1.0}]

        >>> spans = detect_code_switch_spans("これはtestです", "ja")
        >>> len(spans)
        3
    """
    if not text:
        return []

    # Step 1: classify each character
    char_langs = [_classify_char(ch) for ch in text]

    # Step 2: resolve CJK ambiguity
    char_langs = _resolve_cjk_ambiguity(char_langs, text, primary_language)

    # Step 3: merge contiguous spans of the same language
    spans: list[dict] = []
    current_lang: str | None = None
    current_start = 0
    classified_count = 0
    total_chars = 0

    for i, lang in enumerate(char_langs):
        if lang is None:
            # Skip unclassifiable characters (punctuation, whitespace)
            # They inherit the language of the previous span
            total_chars += 1
            continue

        total_chars += 1
        classified_count += 1

        if lang != current_lang:
            # Close previous span
            if current_lang is not None:
                spans.append({
                    "start_char": current_start,
                    "end_char": i,
                    "language": current_lang,
                    "confidence": 1.0,  # adjusted below
                })
            current_lang = lang
            current_start = i

    # Close the last span
    if current_lang is not None:
        spans.append({
            "start_char": current_start,
            "end_char": len(text),
            "language": current_lang,
            "confidence": 1.0,
        })

    # Step 4: adjust confidence based on span length and context
    for span in spans:
        span_text = text[span["start_char"]:span["end_char"]]
        # Short spans (1-2 chars) get lower confidence
        lang_chars = sum(1 for ch in span_text if _classify_char(ch) is not None)
        if lang_chars <= 1:
            span["confidence"] = 0.5
        elif lang_chars <= 3:
            span["confidence"] = 0.7
        else:
            span["confidence"] = 1.0

        # CJK in non-CJK primary context gets slightly lower confidence
        if span["language"] in ("zh", "ja", "ko") and span["language"] != primary_language:
            span["confidence"] = min(span["confidence"], 0.8)

    return spans


def has_code_switch(text: str, primary_language: str = "ja") -> bool:
    """Quick check: does the text contain any language switches?

    Args:
        text: Input text.
        primary_language: Expected primary language.

    Returns:
        True if 2+ distinct languages are detected in the text.
    """
    spans = detect_code_switch_spans(text, primary_language)
    languages = {s["language"] for s in spans}
    return len(languages) >= 2


def code_switch_ratio(texts: list[str], primary_language: str = "ja") -> float:
    """Compute the ratio of texts containing code-switches.

    Args:
        texts: List of text strings.
        primary_language: Expected primary language.

    Returns:
        Float in [0, 1] representing fraction of texts with code-switching.
        Returns 0.0 for empty input.
    """
    if not texts:
        return 0.0
    switched = sum(1 for t in texts if has_code_switch(t, primary_language))
    return switched / len(texts)
