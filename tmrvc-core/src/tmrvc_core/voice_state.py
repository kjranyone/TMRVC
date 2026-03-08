"""Canonical 8-D physical voice_state registry.

Defines the frozen semantic meaning, units, and ranges for the explicit physical
control path in TMRVC UCLM v3. Also provides compatibility helpers for legacy
style surfaces that still expose non-canonical controls.
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class VoiceStateDimension:
    index: int
    id: str
    name: str
    physical_interpretation: str
    unit: str
    min_val: float
    max_val: float
    default_val: float
    is_frame_local: bool
    proxy_observable: str


# ---------------------------------------------------------------------------
# Canonical 8-D Registry (Frozen)
# ---------------------------------------------------------------------------
# Reusing index k with different semantics across workers is forbidden.

VOICE_STATE_REGISTRY: Dict[int, VoiceStateDimension] = {
    0: VoiceStateDimension(
        index=0,
        id="pitch_level",
        name="Pitch Level",
        physical_interpretation="Fundamental frequency (F0) level",
        unit="log-Hz normalized",
        min_val=0.0,
        max_val=1.0,
        default_val=0.5,
        is_frame_local=True,
        proxy_observable="f0",
    ),
    1: VoiceStateDimension(
        index=1,
        id="pitch_range",
        name="Pitch Range",
        physical_interpretation="Local melodic variation derived from F0 spread",
        unit="normalized std",
        min_val=0.0,
        max_val=1.0,
        default_val=0.3,
        is_frame_local=False,
        proxy_observable="f0_std",
    ),
    2: VoiceStateDimension(
        index=2,
        id="energy_level",
        name="Energy Level",
        physical_interpretation="Root Mean Square (RMS) energy",
        unit="dB normalized",
        min_val=0.0,
        max_val=1.0,
        default_val=0.5,
        is_frame_local=True,
        proxy_observable="rms",
    ),
    3: VoiceStateDimension(
        index=3,
        id="pressedness",
        name="Pressedness",
        physical_interpretation="Phonation compression / glottal adduction proxy",
        unit="normalized composite",
        min_val=0.0,
        max_val=1.0,
        default_val=0.35,
        is_frame_local=False,
        proxy_observable="cpp_h1h2",
    ),
    4: VoiceStateDimension(
        index=4,
        id="spectral_tilt",
        name="Spectral Tilt",
        physical_interpretation="Spectral slope / brightness of the vocal source",
        unit="normalized slope",
        min_val=0.0,
        max_val=1.0,
        default_val=0.5,
        is_frame_local=True,
        proxy_observable="spectral_tilt",
    ),
    5: VoiceStateDimension(
        index=5,
        id="breathiness",
        name="Breathiness",
        physical_interpretation="Inverse harmonics-to-noise ratio",
        unit="normalized inverse HNR",
        min_val=0.0,
        max_val=1.0,
        default_val=0.2,
        is_frame_local=True,
        proxy_observable="hnr",
    ),
    6: VoiceStateDimension(
        index=6,
        id="voice_irregularity",
        name="Voice Irregularity",
        physical_interpretation="Jitter + shimmer composite",
        unit="normalized composite",
        min_val=0.0,
        max_val=1.0,
        default_val=0.15,
        is_frame_local=True,
        proxy_observable="jitter_shimmer",
    ),
    7: VoiceStateDimension(
        index=7,
        id="openness",
        name="Openness",
        physical_interpretation="Vocal-tract opening / articulation openness proxy",
        unit="normalized openness",
        min_val=0.0,
        max_val=1.0,
        default_val=0.5,
        is_frame_local=False,
        proxy_observable="f1_proxy",
    ),
}

LEGACY_VOICE_STATE_LABELS: Dict[str, str] = {
    "pitch": "pitch_level",
    "energy": "energy_level",
    "speech_rate": "pressedness",
    "breathiness": "breathiness",
    "tension": "spectral_tilt",
    "brightness": "voice_irregularity",
    "pause_bias": "openness",
    "emphasis": "pitch_range",
}

CANONICAL_VOICE_STATE_IDS: tuple[str, ...] = tuple(
    VOICE_STATE_REGISTRY[i].id for i in range(len(VOICE_STATE_REGISTRY))
)
CANONICAL_VOICE_STATE_LABELS: tuple[str, ...] = tuple(
    VOICE_STATE_REGISTRY[i].name for i in range(len(VOICE_STATE_REGISTRY))
)
CANONICAL_VOICE_STATE_DEFAULTS: tuple[float, ...] = tuple(
    VOICE_STATE_REGISTRY[i].default_val for i in range(len(VOICE_STATE_REGISTRY))
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def canonical_voice_state_dict(values: Sequence[float]) -> Dict[str, float]:
    if len(values) != len(CANONICAL_VOICE_STATE_IDS):
        raise ValueError(
            f"Expected {len(CANONICAL_VOICE_STATE_IDS)} voice-state values, got {len(values)}"
        )
    return {
        CANONICAL_VOICE_STATE_IDS[i]: float(values[i])
        for i in range(len(CANONICAL_VOICE_STATE_IDS))
    }


def legacy_style_to_canonical_voice_state(style: object | None) -> List[float]:
    """Approximate canonical 8-D voice_state from legacy compatibility style fields.

    This is intentionally conservative: only dimensions that have a plausible
    legacy analogue are transferred; the rest fall back to canonical defaults.
    """
    values = list(CANONICAL_VOICE_STATE_DEFAULTS)
    if style is None:
        return values

    pitch_range = getattr(style, "pitch_range", values[1])
    if pitch_range == 0.0:
        pitch_range = getattr(style, "arousal", values[1])

    speech_rate = getattr(style, "speech_rate", 1.0)
    pressedness = (float(speech_rate) - 0.5) / 1.5

    values[0] = _clamp01(getattr(style, "arousal", values[0]))
    values[1] = _clamp01(pitch_range)
    values[2] = _clamp01(getattr(style, "energy", values[2]))
    values[3] = _clamp01(pressedness)
    values[4] = _clamp01(getattr(style, "tension", values[4]))
    values[5] = _clamp01(getattr(style, "breathiness", values[5]))
    values[6] = _clamp01(getattr(style, "roughness", values[6]))
    values[7] = CANONICAL_VOICE_STATE_DEFAULTS[7]
    return values


def get_voice_state_dimension_names() -> List[str]:
    return [VOICE_STATE_REGISTRY[i].name for i in range(len(VOICE_STATE_REGISTRY))]
