"""Dialogue and character types for context-aware TTS.

`StyleParams` mirrors the canonical 12-D physical control contract defined
in `tmrvc_core.voice_state`. Its `to_vector()` returns the 12-D physical
voice-state vector in canonical index order.
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
    """Style parameters for script parsing and context prediction.

    Fields mirror the canonical 12-D physical voice-state registry
    (``tmrvc_core.voice_state``).  ``to_vector()`` returns the 12-D
    physical vector in canonical index order.

    Non-physical metadata (``emotion``, ``reasoning``) are kept for
    script-level semantics but are *not* part of the numeric vector.
    """

    emotion: str = "neutral"

    # -- 12-D physical controls (canonical index order) --
    # idx 0: pitch_level
    pitch_level: float = 0.5
    # idx 1: pitch_range
    pitch_range: float = 0.3
    # idx 2: energy_level  (legacy alias: energy)
    energy: float = 0.5
    # idx 3: pressedness  (legacy alias: tension)
    tension: float = 0.35
    # idx 4: spectral_tilt
    spectral_tilt: float = 0.5
    # idx 5: breathiness
    breathiness: float = 0.2
    # idx 6: voice_irregularity  (legacy alias: roughness)
    roughness: float = 0.15
    # idx 7: openness
    openness: float = 0.5
    # idx 8: aperiodicity
    aperiodicity: float = 0.2
    # idx 9: formant_shift
    formant_shift: float = 0.5
    # idx 10: vocal_effort
    vocal_effort: float = 0.4
    # idx 11: creak
    creak: float = 0.1

    # -- Non-physical semantic/control fields --
    arousal: float = 0.0
    valence: float = 0.0
    voicing: float = 1.0
    speech_rate: float = 1.0
    reasoning: str = ""

    def to_vector(self) -> list[float]:
        """Return the 12-D physical voice-state vector (canonical order)."""
        return [
            _clamp(self.pitch_level, 0.0, 1.0),       # 0  pitch_level
            _clamp(self.pitch_range, 0.0, 1.0),        # 1  pitch_range
            _clamp(self.energy, 0.0, 1.0),             # 2  energy_level
            _clamp(self.tension, 0.0, 1.0),            # 3  pressedness
            _clamp(self.spectral_tilt, 0.0, 1.0),      # 4  spectral_tilt
            _clamp(self.breathiness, 0.0, 1.0),        # 5  breathiness
            _clamp(self.roughness, 0.0, 1.0),          # 6  voice_irregularity
            _clamp(self.openness, 0.0, 1.0),           # 7  openness
            _clamp(self.aperiodicity, 0.0, 1.0),       # 8  aperiodicity
            _clamp(self.formant_shift, 0.0, 1.0),      # 9  formant_shift
            _clamp(self.vocal_effort, 0.0, 1.0),       # 10 vocal_effort
            _clamp(self.creak, 0.0, 1.0),              # 11 creak
        ]

    @classmethod
    def from_dict(cls, d: dict) -> StyleParams:
        """Create from a dict (e.g. JSON response from LLM)."""
        return cls(
            emotion=d.get("emotion", "neutral"),
            pitch_level=d.get("pitch_level", 0.5),
            pitch_range=d.get("pitch_range", 0.3),
            energy=d.get("energy", 0.5),
            tension=d.get("tension", 0.35),
            spectral_tilt=d.get("spectral_tilt", 0.5),
            breathiness=d.get("breathiness", 0.2),
            roughness=d.get("roughness", 0.15),
            openness=d.get("openness", 0.5),
            aperiodicity=d.get("aperiodicity", 0.2),
            formant_shift=d.get("formant_shift", 0.5),
            vocal_effort=d.get("vocal_effort", 0.4),
            creak=d.get("creak", 0.1),
            arousal=d.get("arousal", 0.0),
            valence=d.get("valence", 0.0),
            voicing=d.get("voicing", 1.0),
            speech_rate=d.get("speech_rate", 1.0),
            reasoning=d.get("reasoning", ""),
        )

    @classmethod
    def neutral(cls) -> StyleParams:
        """Return default neutral style (all physical dims at registry defaults)."""
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
