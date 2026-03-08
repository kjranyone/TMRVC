"""tmrvc-core: shared constants, mel computation, type definitions."""

from tmrvc_core.constants import (
    D_MODEL,
    D_SPEAKER,
    HOP_LENGTH,
    MEL_FMAX,
    MEL_FMIN,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from tmrvc_core.device import get_device, pin_memory_for_device
from tmrvc_core.types import (
    CFG_PRESERVED_FIELDS,
    CFG_ZEROED_FIELDS,
    PointerState,
    SpeakerProfile,
    VoiceStateSupervision,
)

__all__ = [
    "CFG_PRESERVED_FIELDS",
    "CFG_ZEROED_FIELDS",
    "D_MODEL",
    "D_SPEAKER",
    "HOP_LENGTH",
    "MEL_FMAX",
    "MEL_FMIN",
    "N_FFT",
    "N_MELS",
    "SAMPLE_RATE",
    "WINDOW_LENGTH",
    "PointerState",
    "SpeakerProfile",
    "VoiceStateSupervision",
    "get_device",
    "pin_memory_for_device",
]
