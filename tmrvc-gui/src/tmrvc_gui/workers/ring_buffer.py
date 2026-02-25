"""Lock-free SPSC ring buffer backed by a numpy array."""

from __future__ import annotations

import numpy as np


# Ring buffer capacity (samples)
RING_BUFFER_CAPACITY: int = 4096


class RingBuffer:
    """Lock-free single-producer single-consumer ring buffer.

    Backed by a contiguous :class:`numpy.ndarray` of ``float32`` values.
    The read and write positions advance monotonically and wrap using
    modular arithmetic.

    Parameters
    ----------
    capacity : int
        Maximum number of samples the buffer can hold.  Must be a
        power of two for efficient modular wrapping.

    Notes
    -----
    This Python implementation is **not** truly lock-free in the C++
    sense (the GIL serialises access), but it follows the same API
    contract as the C++ ``FixedRingBuffer`` to keep the code portable.
    """

    def __init__(self, capacity: int = RING_BUFFER_CAPACITY) -> None:
        # Round up to the next power of two for efficient masking
        self._capacity: int = 1 << (capacity - 1).bit_length()
        self._mask: int = self._capacity - 1
        self._buffer: np.ndarray = np.zeros(self._capacity, dtype=np.float32)
        self._read_pos: int = 0
        self._write_pos: int = 0

    @property
    def capacity(self) -> int:
        """Return the usable capacity (always one less than allocation)."""
        return self._capacity

    def available(self) -> int:
        """Return the number of samples available for reading."""
        return self._write_pos - self._read_pos

    def free_space(self) -> int:
        """Return the number of samples that can be written."""
        return self._capacity - self.available()

    def write(self, data: np.ndarray) -> int:
        """Write *data* into the buffer.

        Parameters
        ----------
        data : np.ndarray
            1-D ``float32`` array of samples to write.

        Returns
        -------
        int
            Number of samples actually written (may be less than
            ``len(data)`` if the buffer is nearly full).
        """
        n = min(len(data), self.free_space())
        if n == 0:
            return 0

        start = self._write_pos & self._mask
        end = start + n

        if end <= self._capacity:
            self._buffer[start:end] = data[:n]
        else:
            first = self._capacity - start
            self._buffer[start:] = data[:first]
            self._buffer[: n - first] = data[first:n]

        self._write_pos += n
        return n

    def read(self, count: int) -> np.ndarray:
        """Read and consume up to *count* samples from the buffer.

        Parameters
        ----------
        count : int
            Maximum number of samples to read.

        Returns
        -------
        np.ndarray
            1-D ``float32`` array of consumed samples.  May be shorter
            than *count* if fewer samples are available.
        """
        n = min(count, self.available())
        if n == 0:
            return np.empty(0, dtype=np.float32)

        start = self._read_pos & self._mask
        end = start + n

        if end <= self._capacity:
            out = self._buffer[start:end].copy()
        else:
            first = self._capacity - start
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: n - first]

        self._read_pos += n
        return out

    def peek(self, count: int, offset: int = 0) -> np.ndarray:
        """Read *count* samples starting at *offset* without consuming.

        Parameters
        ----------
        count : int
            Number of samples to peek.
        offset : int
            Offset from the current read position.

        Returns
        -------
        np.ndarray
            1-D ``float32`` array.
        """
        avail = self.available()
        if offset + count > avail:
            count = max(0, avail - offset)
        if count == 0:
            return np.empty(0, dtype=np.float32)

        start = (self._read_pos + offset) & self._mask
        end = start + count

        if end <= self._capacity:
            return self._buffer[start:end].copy()
        else:
            first = self._capacity - start
            out = np.empty(count, dtype=np.float32)
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: count - first]
            return out

    def reset(self) -> None:
        """Reset the buffer to empty, zeroing all data."""
        self._read_pos = 0
        self._write_pos = 0
        self._buffer[:] = 0.0
