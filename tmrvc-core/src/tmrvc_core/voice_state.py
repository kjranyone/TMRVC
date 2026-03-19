"""Canonical 12-D physical voice_state registry.

Defines the frozen semantic meaning, units, and ranges for the explicit physical
control path in TMRVC UCLM.
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
# Canonical 12-D Registry (Frozen)
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
    8: VoiceStateDimension(
        index=8,
        id="aperiodicity",
        name="Aperiodicity",
        physical_interpretation="Aperiodic energy ratio in voiced segments",
        unit="normalized ratio",
        min_val=0.0,
        max_val=1.0,
        default_val=0.2,
        is_frame_local=True,
        proxy_observable="aperiodicity_ratio",
    ),
    9: VoiceStateDimension(
        index=9,
        id="formant_shift",
        name="Formant Shift",
        physical_interpretation="Vocal-tract length proxy via formant frequency shift",
        unit="normalized shift",
        min_val=0.0,
        max_val=1.0,
        default_val=0.5,
        is_frame_local=False,
        proxy_observable="vtln_warp",
    ),
    10: VoiceStateDimension(
        index=10,
        id="vocal_effort",
        name="Vocal Effort",
        physical_interpretation="Overall phonatory effort independent of loudness",
        unit="normalized effort",
        min_val=0.0,
        max_val=1.0,
        default_val=0.4,
        is_frame_local=True,
        proxy_observable="effort_composite",
    ),
    11: VoiceStateDimension(
        index=11,
        id="creak",
        name="Creak",
        physical_interpretation="Subharmonic / vocal fry presence",
        unit="normalized presence",
        min_val=0.0,
        max_val=1.0,
        default_val=0.1,
        is_frame_local=True,
        proxy_observable="subharmonic_ratio",
    ),
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

# Basic panel: most-used 6 controls for non-expert users
BASIC_PANEL_IDS: tuple[str, ...] = (
    "pitch_level", "energy_level", "breathiness",
    "pressedness", "spectral_tilt", "vocal_effort",
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


def get_voice_state_dimension_names() -> List[str]:
    return [VOICE_STATE_REGISTRY[i].name for i in range(len(VOICE_STATE_REGISTRY))]
