"""Dialogue and character types for context-aware TTS.

`StyleParams` in this module is a legacy convenience layer used by existing
heuristics, script parsing, and context predictors. It is not the canonical
v3 `voice_state` registry; the canonical 8-D physical contract lives in
`tmrvc_core.voice_state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

EMOTION_CATEGORIES: list[str] = [
    "happy",
    "sad",
    "angry",
    "fearful",
    "surprised",
    "disgusted",
    "neutral",
    "bored",
    "excited",
    "tender",
    "sarcastic",
    "whisper",
]

EMOTION_TO_ID: dict[str, int] = {e: i for i, e in enumerate(EMOTION_CATEGORIES)}
SUPPORTED_LANGUAGES: tuple[str, ...] = ("ja", "en", "zh", "ko")


@dataclass
class StyleParams:
    """Legacy convenience style parameters for compatibility paths.

    `to_vector()` returns the legacy 8-D style vector used by older style
    predictors and script tooling. It should not be treated as the canonical
    v3 physical `voice_state` registry for sign-off purposes.
    """

    emotion: str = "neutral"
    breathiness: float = 0.0
    tension: float = 0.0
    arousal: float = 0.0
    valence: float = 0.0
    roughness: float = 0.0
    voicing: float = 1.0
    energy: float = 0.0
    speech_rate: float = 1.0
    pitch_range: float = 0.0  # Kept for backward compatibility, mapped to arousal/energy
    reasoning: str = ""

    def to_vector(self) -> list[float]:
        """Convert to the legacy 8-dim style vector."""
        vec = [0.0] * 8

        vec[0] = _clamp(self.breathiness, 0.0, 1.0)
        vec[1] = _clamp(self.tension, 0.0, 1.0)
        vec[2] = _clamp(self.arousal, 0.0, 1.0)
        vec[3] = _clamp(self.valence, -1.0, 1.0)
        vec[4] = _clamp(self.roughness, 0.0, 1.0)
        vec[5] = _clamp(self.voicing, 0.0, 1.0)
        vec[6] = _clamp(self.energy, 0.0, 1.0)
        vec[7] = _clamp(self.speech_rate, 0.5, 2.0)

        return vec

    @classmethod
    def from_dict(cls, d: dict) -> StyleParams:
        """Create from a dict (e.g. JSON response from LLM)."""
        return cls(
            emotion=d.get("emotion", "neutral"),
            breathiness=d.get("breathiness", 0.0),
            tension=d.get("tension", 0.0),
            arousal=d.get("arousal", 0.0),
            valence=d.get("valence", 0.0),
            roughness=d.get("roughness", 0.0),
            voicing=d.get("voicing", 1.0),
            energy=d.get("energy", 0.0),
            speech_rate=d.get("speech_rate", 1.0),
            pitch_range=d.get("pitch_range", 0.0),
            reasoning=d.get("reasoning", ""),
        )

    @classmethod
    def neutral(cls) -> StyleParams:
        """Return default neutral style for compatibility paths."""
        return cls()


@dataclass
class CharacterProfile:
    """Character profile for context-aware TTS.

    Stores persona information used by the ContextStylePredictor
    to infer appropriate emotion and style for each utterance.
    """

    name: str
    personality: str = ""
    voice_description: str = ""
    default_style: StyleParams = field(default_factory=StyleParams.neutral)
    speaker_file: Path | None = None
    language: str = "ja"

    def __post_init__(self) -> None:
        if self.language not in SUPPORTED_LANGUAGES:
            supported = ", ".join(SUPPORTED_LANGUAGES)
            raise ValueError(
                f"Unsupported language: {self.language}. Supported languages: {supported}"
            )


@dataclass
class DialogueTurn:
    """A single turn in a conversation."""

    speaker: str
    text: str
    emotion: str | None = None
    timestamp: float | None = None


@dataclass
class ScriptEntry:
    """A single entry in a TTS script/scenario."""

    speaker: str
    text: str
    hint: str | None = None
    style_override: StyleParams | None = None
    speed: float = 1.0


@dataclass
class Script:
    """A TTS script/scenario with characters and dialogue."""

    title: str = ""
    situation: str = ""
    characters: dict[str, CharacterProfile] = field(default_factory=dict)
    entries: list[ScriptEntry] = field(default_factory=list)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
