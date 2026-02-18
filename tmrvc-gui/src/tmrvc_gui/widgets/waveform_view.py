"""WaveformView widget - Audio waveform display backed by pyqtgraph."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class WaveformView(QWidget):
    """Display a static or rolling audio waveform.

    Uses :class:`pyqtgraph.PlotWidget` for efficient rendering.  The X
    axis is in seconds and the Y axis shows sample amplitude.

    Args:
        title: Plot title.
        duration_sec: Length of visible window for rolling display.
        sample_rate: Sample rate of the audio data in Hz.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        title: str = "",
        duration_sec: float = 5.0,
        sample_rate: int = 24000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._duration_sec = duration_sec
        self._sample_rate = sample_rate
        self._max_samples = int(duration_sec * sample_rate)

        # Rolling buffer for append_data.
        self._buffer: np.ndarray = np.array([], dtype=np.float32)

        # --- Plot widget ---
        self._plot_widget = pg.PlotWidget(title=title if title else None)
        self._plot_widget.setLabel("bottom", "Time", units="s")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setYRange(-1.0, 1.0)

        self._curve: pg.PlotDataItem = self._plot_widget.plot(
            [], [], pen=pg.mkPen(color="#4fc3f7", width=1)
        )

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_data(self, samples: np.ndarray) -> None:
        """Display a static waveform.

        Replaces any existing data (including the rolling buffer).

        Args:
            samples: 1-D array of audio samples.
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self._buffer = samples

        n = len(samples)
        t = np.arange(n, dtype=np.float32) / self._sample_rate
        self._curve.setData(t, samples)
        self._plot_widget.setXRange(0, t[-1] if n > 0 else self._duration_sec)

    def append_data(self, samples: np.ndarray) -> None:
        """Append audio samples for a rolling waveform display.

        Only the last *duration_sec* seconds of data are retained.

        Args:
            samples: 1-D array of new audio samples to append.
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        self._buffer = np.concatenate([self._buffer, samples])

        # Trim to the maximum visible window.
        if len(self._buffer) > self._max_samples:
            self._buffer = self._buffer[-self._max_samples:]

        n = len(self._buffer)
        t = np.arange(n, dtype=np.float32) / self._sample_rate
        self._curve.setData(t, self._buffer)
        self._plot_widget.setXRange(0, self._duration_sec)

    def clear(self) -> None:
        """Clear the waveform display and internal buffer."""
        self._buffer = np.array([], dtype=np.float32)
        self._curve.setData([], [])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """The sample rate used for time-axis calculation."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value
        self._max_samples = int(self._duration_sec * value)

    @property
    def duration_sec(self) -> float:
        """The visible window duration in seconds."""
        return self._duration_sec

    @duration_sec.setter
    def duration_sec(self, value: float) -> None:
        self._duration_sec = value
        self._max_samples = int(value * self._sample_rate)
