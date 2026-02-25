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


# ---------------------------------------------------------------------------
# Inline stage-direction analysis
# ---------------------------------------------------------------------------

_STAGE_BLOCK_PATTERN = re.compile(
    r"\[[^\[\]\n]{1,160}\]"
    r"|\([^\(\)\n]{1,160}\)"
    r"|（[^（）\n]{1,160}）"
    r"|【[^【】\n]{1,160}】"
    r"|<[^<>\n]{1,160}>"
    r"|＜[^＜＞\n]{1,160}＞"
)

_NON_SPOKEN_CHARS = set(
    " \t\r\n"
    + ".,!?;:。！？、…"
    + "\"'`"
    + "「」『』（）()[]【】<>＜＞"
)


@dataclass(frozen=True)
class InlineStageAnalysis:
    """Inline stage-direction parse result for acting-aware TTS."""

    spoken_text: str
    stage_directions: list[str]
    style_overlay: object | None
    speed_scale: float = 1.0
    sentence_pause_ms_delta: int = 0
    leading_silence_ms: int = 0
    trailing_silence_ms: int = 0


@dataclass(frozen=True)
class _StageCueRule:
    tag: str
    keywords: tuple[str, ...]
    emotion: str | None = None
    delta_valence: float = 0.0
    delta_arousal: float = 0.0
    delta_energy: float = 0.0
    delta_speech_rate: float = 0.0
    delta_pitch_range: float = 0.0
    speed_scale: float = 1.0
    pause_ms: int = 0
    priority: float = 0.0


_STAGE_CUE_RULES: tuple[_StageCueRule, ...] = (
    _StageCueRule(
        tag="whisper",
        keywords=("whisper", "囁", "ささや", "속삭", "低声", "耳元"),
        emotion="whisper",
        delta_valence=0.1,
        delta_arousal=-0.25,
        delta_energy=-0.45,
        delta_speech_rate=-0.20,
        delta_pitch_range=-0.10,
        speed_scale=0.88,
        pause_ms=120,
        priority=0.9,
    ),
    _StageCueRule(
        tag="long_breath",
        keywords=(
            "long breath", "deep breath", "long inhale", "long exhale",
            "深呼吸", "長い息", "긴 숨", "长呼吸",
        ),
        delta_arousal=-0.20,
        delta_energy=-0.30,
        delta_speech_rate=-0.25,
        speed_scale=0.82,
        pause_ms=260,
        priority=0.7,
    ),
    _StageCueRule(
        tag="breath",
        keywords=("breath", "inhale", "exhale", "息", "吐息", "呼吸", "숨", "气息"),
        delta_arousal=-0.10,
        delta_energy=-0.18,
        delta_speech_rate=-0.12,
        speed_scale=0.93,
        pause_ms=120,
        priority=0.5,
    ),
    _StageCueRule(
        tag="breathy_acting",
        keywords=("moan", "呻吟", "喘ぎ", "신음"),
        emotion="tender",
        delta_valence=0.05,
        delta_arousal=0.12,
        delta_energy=-0.10,
        delta_speech_rate=-0.15,
        speed_scale=0.90,
        pause_ms=80,
        priority=0.85,
    ),
    _StageCueRule(
        tag="pause",
        keywords=("pause", "silence", "間", "沈黙", "停顿", "정적"),
        delta_speech_rate=-0.12,
        speed_scale=0.92,
        pause_ms=180,
        priority=0.6,
    ),
    _StageCueRule(
        tag="tremble",
        keywords=("tremble", "shiver", "震", "ふる", "떨", "颤"),
        emotion="fearful",
        delta_valence=-0.10,
        delta_arousal=0.18,
        delta_energy=-0.05,
        delta_pitch_range=0.22,
        speed_scale=0.96,
        pause_ms=40,
        priority=0.8,
    ),
    _StageCueRule(
        tag="cry",
        keywords=("cry", "sob", "涙", "泣", "울", "哭"),
        emotion="sad",
        delta_valence=-0.40,
        delta_arousal=-0.08,
        delta_energy=-0.22,
        delta_speech_rate=-0.12,
        speed_scale=0.90,
        pause_ms=100,
        priority=0.85,
    ),
    _StageCueRule(
        tag="shout",
        keywords=("shout", "yell", "叫", "大声", "怒鳴", "소리치", "喊"),
        emotion="excited",
        delta_valence=-0.05,
        delta_arousal=0.45,
        delta_energy=0.42,
        delta_speech_rate=0.12,
        delta_pitch_range=0.18,
        speed_scale=1.10,
        pause_ms=0,
        priority=0.9,
    ),
)


def _clamp_stage(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _has_spoken_content(fragment: str) -> bool:
    for ch in fragment:
        if ch in _NON_SPOKEN_CHARS:
            continue
        return True
    return False


def _extract_stage_blocks(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Extract inline stage blocks and return (spoken_text, [(direction, location)])."""
    blocks: list[tuple[str, str]] = []
    spoken_parts: list[str] = []
    cursor = 0

    for match in _STAGE_BLOCK_PATTERN.finditer(text):
        spoken_parts.append(text[cursor:match.start()])
        cursor = match.end()

        block = match.group(0)
        direction = block[1:-1].strip()
        if not direction:
            continue

        before_has = _has_spoken_content(text[:match.start()])
        after_has = _has_spoken_content(text[match.end():])
        if not before_has and after_has:
            location = "prefix"
        elif before_has and not after_has:
            location = "suffix"
        elif before_has and after_has:
            location = "middle"
        else:
            location = "standalone"
        blocks.append((direction, location))

    spoken_parts.append(text[cursor:])
    spoken_text = re.sub(r"\s+", " ", "".join(spoken_parts)).strip()
    return spoken_text, blocks


def _rule_matches(direction: str, rule: _StageCueRule) -> bool:
    lower = direction.lower()
    for keyword in rule.keywords:
        if keyword.isascii():
            if keyword.lower() in lower:
                return True
        elif keyword in direction:
            return True
    return False


def analyze_inline_stage_directions(
    text: str,
    language: str = "ja",
) -> InlineStageAnalysis:
    """Parse inline stage directions and derive acting control signals.

    Supported direction blocks:
    - ``(...)`` / ``（...）``
    - ``[...]`` / ``【...】``
    - ``<...>`` / ``＜...＞``
    """
    from tmrvc_core.dialogue_types import StyleParams

    _ = language  # reserved for future language-specific weighting
    spoken_text, blocks = _extract_stage_blocks(text)
    stage_texts = [direction for direction, _loc in blocks]

    if not blocks:
        clean = text.strip()
        return InlineStageAnalysis(
            spoken_text=clean if clean else text,
            stage_directions=[],
            style_overlay=None,
        )

    emotion = "neutral"
    emotion_priority = 0.0
    valence = 0.0
    arousal = 0.0
    energy = 0.0
    speech_rate = 0.0
    pitch_range = 0.0
    speed_scale = 1.0
    sentence_pause_ms_delta = 0
    leading_silence_ms = 0
    trailing_silence_ms = 0
    tags: list[str] = []

    for direction, location in blocks:
        for rule in _STAGE_CUE_RULES:
            if not _rule_matches(direction, rule):
                continue

            if rule.emotion is not None and rule.priority >= emotion_priority:
                emotion = rule.emotion
                emotion_priority = rule.priority

            valence += rule.delta_valence
            arousal += rule.delta_arousal
            energy += rule.delta_energy
            speech_rate += rule.delta_speech_rate
            pitch_range += rule.delta_pitch_range
            speed_scale *= rule.speed_scale

            if rule.pause_ms > 0:
                if location == "prefix":
                    leading_silence_ms += rule.pause_ms
                elif location == "suffix":
                    trailing_silence_ms += rule.pause_ms
                else:
                    sentence_pause_ms_delta += rule.pause_ms

            if rule.tag not in tags:
                tags.append(rule.tag)

    if not tags:
        fallback_text = spoken_text if spoken_text else text.strip()
        return InlineStageAnalysis(
            spoken_text=fallback_text if fallback_text else text,
            stage_directions=stage_texts,
            style_overlay=None,
        )

    style_overlay = StyleParams(
        emotion=emotion,
        valence=_clamp_stage(valence, -1.0, 1.0),
        arousal=_clamp_stage(arousal, -1.0, 1.0),
        dominance=0.0,
        speech_rate=_clamp_stage(speech_rate, -1.0, 1.0),
        energy=_clamp_stage(energy, -1.0, 1.0),
        pitch_range=_clamp_stage(pitch_range, -1.0, 1.0),
        reasoning=f"inline_stage:{','.join(tags)}",
    )

    fallback_text = spoken_text if spoken_text else text.strip()
    return InlineStageAnalysis(
        spoken_text=fallback_text if fallback_text else text,
        stage_directions=stage_texts,
        style_overlay=style_overlay,
        speed_scale=_clamp_stage(speed_scale, 0.75, 1.25),
        sentence_pause_ms_delta=int(_clamp_stage(float(sentence_pause_ms_delta), -120.0, 600.0)),
        leading_silence_ms=int(_clamp_stage(float(leading_silence_ms), 0.0, 1600.0)),
        trailing_silence_ms=int(_clamp_stage(float(trailing_silence_ms), 0.0, 1600.0)),
    )
