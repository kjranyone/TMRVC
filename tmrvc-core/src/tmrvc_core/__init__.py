"""tmrvc-core: shared constants, mel computation, type definitions."""

from tmrvc_core.constants import (
    D_CONTENT,
    D_CONTENT_VEC,
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

__all__ = [
    "D_CONTENT",
    "D_CONTENT_VEC",
    "D_SPEAKER",
    "HOP_LENGTH",
    "MEL_FMAX",
    "MEL_FMIN",
    "N_FFT",
    "N_MELS",
    "SAMPLE_RATE",
    "WINDOW_LENGTH",
    "get_device",
    "pin_memory_for_device",
]
