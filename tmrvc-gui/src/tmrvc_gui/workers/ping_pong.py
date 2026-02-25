"""Double-buffered (ping-pong) state tensor management."""

from __future__ import annotations

import numpy as np


class PingPongState:
    """Double-buffered (ping-pong) state tensor pair.

    Maintains two numpy arrays of identical shape and alternates
    which one is used as model input vs. output on each frame,
    avoiding in-place overwrites.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of each state buffer (e.g. ``(1, 256, 28)``).
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.buffer_a: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.buffer_b: np.ndarray = np.zeros(shape, dtype=np.float32)
        self._current: int = 0  # 0 => A is input, 1 => B is input

    @property
    def input(self) -> np.ndarray:
        """Return the current input state buffer."""
        return self.buffer_a if self._current == 0 else self.buffer_b

    @property
    def output(self) -> np.ndarray:
        """Return the current output state buffer."""
        return self.buffer_b if self._current == 0 else self.buffer_a

    def swap(self) -> None:
        """Swap input and output roles (toggle ping-pong)."""
        self._current ^= 1

    def reset(self) -> None:
        """Zero-initialise both buffers (silence state)."""
        self.buffer_a[:] = 0.0
        self.buffer_b[:] = 0.0
        self._current = 0
