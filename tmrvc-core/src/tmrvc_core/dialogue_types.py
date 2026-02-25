"""Dialogue and character types for context-aware TTS.

Used by the ContextStylePredictor (Phase 4) and VTuber integration (Phase 5)
to represent characters, conversation history, and style parameters.
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
    """Parsed style parameters for TTS emotion conditioning.

    Maps to emotion_style[32d] vector:
    - [0:3]  VAD (Valence, Arousal, Dominance)
    - [3:6]  VAD uncertainty
    - [6:9]  Speech rate, Energy, Pitch range
    - [9:21] 12 emotion category softmax
    - [21:29] Learned latent
    - [29:32] Reserved
    """

    emotion: str = "neutral"
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    speech_rate: float = 0.0
    energy: float = 0.0
    pitch_range: float = 0.0
    reasoning: str = ""

    def to_vector(self) -> list[float]:
        """Convert to 32-dim emotion_style vector.

        Returns:
            List of 32 float values.
        """
        vec = [0.0] * 32

        # VAD [0:3]
        vec[0] = _clamp(self.valence, -1.0, 1.0)
        vec[1] = _clamp(self.arousal, -1.0, 1.0)
        vec[2] = _clamp(self.dominance, -1.0, 1.0)

        # VAD uncertainty [3:6] — default 0 (confident)
        # Prosody [6:9]
        vec[6] = _clamp(self.speech_rate, -1.0, 1.0)
        vec[7] = _clamp(self.energy, -1.0, 1.0)
        vec[8] = _clamp(self.pitch_range, -1.0, 1.0)

        # Emotion category one-hot [9:21]
        eid = EMOTION_TO_ID.get(self.emotion, EMOTION_TO_ID["neutral"])
        vec[9 + eid] = 1.0

        return vec

    @classmethod
    def from_dict(cls, d: dict) -> StyleParams:
        """Create from a dict (e.g. JSON response from LLM)."""
        return cls(
            emotion=d.get("emotion", "neutral"),
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.0),
            dominance=d.get("dominance", 0.0),
            speech_rate=d.get("speech_rate", 0.0),
            energy=d.get("energy", 0.0),
            pitch_range=d.get("pitch_range", 0.0),
            reasoning=d.get("reasoning", ""),
        )

    @classmethod
    def neutral(cls) -> StyleParams:
        """Return default neutral style."""
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
