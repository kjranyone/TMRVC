"""Device selection utilities with XPU / CUDA / CPU fallback."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto") -> torch.device:
    """Select the best available device.

    Priority: XPU > CUDA > CPU (when *preferred* is ``"auto"``).
    If a specific device is requested but unavailable, falls back to CPU.

    Parameters
    ----------
    preferred : str
        ``"auto"`` for automatic detection, or ``"xpu"``/``"cuda"``/``"cpu"``
        to request a specific backend.
    """
    preferred = preferred.lower()

    if preferred == "auto":
        if _xpu_available():
            logger.info("Using XPU device (Intel Arc)")
            return torch.device("xpu")
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return torch.device("cuda")
        logger.info("Using CPU device")
        return torch.device("cpu")

    if preferred == "xpu":
        if _xpu_available():
            return torch.device("xpu")
        logger.warning("XPU requested but not available, falling back to CPU")
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")

    return torch.device("cpu")


def pin_memory_for_device(device: torch.device | str) -> bool:
    """Return whether DataLoader should use ``pin_memory`` for *device*.

    Only CUDA benefits from pinned memory; XPU and CPU do not.
    """
    return str(device).startswith("cuda")


def _xpu_available() -> bool:
    """Check if Intel XPU is available.

    PyTorch >= 2.10 has native XPU support; older versions need IPEX.
    """
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
    except Exception:
        pass
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401

        return torch.xpu.is_available()
    except (ImportError, AttributeError):
        return False
