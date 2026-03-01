"""Text utilities for sentence segmentation and style inference (UCLM v2)."""

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
    text = text.strip()
    if not text:
        return [text] if text else [""]

    if language in ("ja", "zh"):
        segments = _segment_ja_zh(text)
    else:
        segments = _segment_en(text)

    merged: list[str] = []
    for seg in segments:
        if merged and len(seg) < 2:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    return merged if merged else [text]


def _segment_ja_zh(text: str) -> list[str]:
    result: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in text:
        current.append(ch)
        if ch in "\u300c\u300e\uff08\u3010\uff3b": depth += 1
        elif ch in "\u300d\u300f\uff09\u3011\uff3d": depth = max(0, depth - 1)
        elif ch in "\u3002\uff01\uff1f\n" and depth == 0:
            seg = "".join(current).strip()
            if seg: result.append(seg)
            current = []
    if current:
        seg = "".join(current).strip()
        if seg: result.append(seg)
    return result


def _segment_en(text: str) -> list[str]:
    parts: list[str] = []
    last = 0
    for m in re.finditer(r"[.!?]\s+(?=[A-Z])", text):
        candidate = text[last:m.end()]
        prefix = text[last:m.start() + 1]
        if _EN_ABBREV_PATTERN.search(prefix): continue
        parts.append(candidate.strip())
        last = m.end()
    remainder = text[last:].strip()
    if remainder: parts.append(remainder)
    return parts


# ---------------------------------------------------------------------------
# Per-sentence style inference (rule-based)
# ---------------------------------------------------------------------------

_JA_STYLE_KEYWORDS: dict[str, tuple[str, float, float]] = {
    "嬉しい": ("happy", 0.3, 0.5),
    "楽しい": ("happy", 0.3, 0.4),
    "ありがとう": ("happy", 0.1, 0.3),
    "すごい": ("excited", 0.5, 0.4),
    "やった": ("excited", 0.5, 0.5),
    "悲しい": ("sad", -0.2, -0.5),
    "怒": ("angry", 0.5, -0.4),
    "怖い": ("fearful", 0.3, -0.3),
    "びっくり": ("surprised", 0.4, 0.1),
    "ひそひそ": ("whisper", -0.4, 0.0),
}

_EN_STYLE_KEYWORDS: dict[str, tuple[str, float, float]] = {
    "happy": ("happy", 0.3, 0.5),
    "amazing": ("excited", 0.5, 0.4),
    "sad": ("sad", -0.2, -0.5),
    "angry": ("angry", 0.5, -0.4),
    "wow": ("surprised", 0.4, 0.2),
    "whisper": ("whisper", -0.4, 0.0),
}


def infer_sentence_style(
    sentence: str,
    language: str,
    base_style: object | None = None,
) -> object:
    from tmrvc_core.dialogue_types import StyleParams
    if base_style is None:
        base_style = StyleParams.neutral()
    
    if not isinstance(base_style, StyleParams):
        return base_style

    emotion = base_style.emotion
    arousal = base_style.arousal
    valence = base_style.valence
    energy = base_style.energy

    exclamation_count = sentence.count("!") + sentence.count("\uff01")
    if exclamation_count >= 2:
        arousal = min(1.0, arousal + 0.4)
        energy = min(1.0, energy + 0.3)
        if emotion == "neutral": emotion = "excited"
    elif exclamation_count == 1:
        arousal = min(1.0, arousal + 0.2)
        energy = min(1.0, energy + 0.1)

    keywords = _JA_STYLE_KEYWORDS if language in ("ja", "zh") else _EN_STYLE_KEYWORDS
    text_lower = sentence.lower()
    for kw, (emo, ar_delta, val_delta) in keywords.items():
        if kw in text_lower or kw in sentence:
            emotion = emo
            arousal = max(-1.0, min(1.0, arousal + ar_delta))
            valence = max(-1.0, min(1.0, valence + val_delta))
            break

    return StyleParams(
        emotion=emotion,
        valence=valence,
        arousal=arousal,
        speech_rate=base_style.speech_rate,
        energy=energy,
        reasoning=f"auto-inferred: {sentence[:30]}",
    )


# ---------------------------------------------------------------------------
# Inline stage-direction analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InlineStageAnalysis:
    spoken_text: str
    stage_directions: list[str]
    style_overlay: object | None
    speed_scale: float = 1.0
    sentence_pause_ms_delta: int = 0
    leading_silence_ms: int = 0
    trailing_silence_ms: int = 0


_STAGE_BLOCK_PATTERN = re.compile(r"\[[^\]\n]+\]|\([^\)\n]+\)|（[^）\n]+）|【[^】\n]+】|<[^>\n]+>|＜[^＞\n]+＞")

def analyze_inline_stage_directions(text: str, language: str = "ja") -> InlineStageAnalysis:
    from tmrvc_core.dialogue_types import StyleParams
    
    # Very simple extraction for now to fix the core crash
    blocks = _STAGE_BLOCK_PATTERN.findall(text)
    spoken_text = _STAGE_BLOCK_PATTERN.sub("", text).strip()
    
    if not blocks:
        return InlineStageAnalysis(spoken_text=spoken_text or text, stage_directions=[], style_overlay=None)

    # Basic mapping: [whisper] -> breathiness up, voicing down
    style_overlay = StyleParams.neutral()
    leading_silence = 0
    
    for b in blocks:
        b = b[1:-1].lower()
        if "whisper" in b or "囁" in b:
            style_overlay.emotion = "whisper"
            style_overlay.breathiness = 0.8
            style_overlay.voicing = 0.2
        elif "happy" in b:
            style_overlay.emotion = "happy"
            style_overlay.valence = 0.6
        elif "ms" in b:
            try: leading_silence = int(re.search(r"\d+", b).group())
            except: pass
        
        # Handle key=value pairs (e.g., tension=0.8)
        kv_match = re.findall(r"([a-z_]+)\s*=\s*([0-9.]+)", b)
        for k, v in kv_match:
            if hasattr(style_overlay, k):
                try: setattr(style_overlay, k, float(v))
                except: pass

    return InlineStageAnalysis(
        spoken_text=spoken_text or text,
        stage_directions=blocks,
        style_overlay=style_overlay,
        leading_silence_ms=leading_silence
    )
