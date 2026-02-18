"""MetricPlot widget - Real-time metric plotting backed by pyqtgraph."""

from __future__ import annotations

from typing import Dict, List

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget

# Colour palette cycled for successive series.
_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


class MetricPlot(QWidget):
    """Wrapper around :class:`pyqtgraph.PlotWidget` for real-time metric plots.

    Each *series* is an independent line on the same axes.  When the number
    of points in a series exceeds *max_points*, the data is downsampled by
    keeping every other point so that memory usage stays bounded.

    Args:
        title: Plot title displayed at the top.
        x_label: Label for the X axis.
        y_label: Label for the Y axis.
        max_points: Maximum data points per series before downsampling.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        title: str,
        x_label: str,
        y_label: str,
        max_points: int = 10000,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._max_points = max_points

        # Data storage: series_name -> {"x": [...], "y": [...]}
        self._data: Dict[str, Dict[str, List[float]]] = {}
        # PlotDataItem handles keyed by series name.
        self._curves: Dict[str, pg.PlotDataItem] = {}
        # Colour index counter.
        self._color_index: int = 0

        # --- Layout ---
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget(title=title)
        self._plot_widget.setLabel("bottom", x_label)
        self._plot_widget.setLabel("left", y_label)
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_widget)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_point(self, x: float, y: float, series: str = "default") -> None:
        """Append a single data point to *series*.

        If the series does not yet exist it is created automatically with
        the next colour from the palette.

        Args:
            x: X-axis value (e.g. step or time).
            y: Y-axis value (e.g. loss).
            series: Name of the data series.
        """
        if series not in self._data:
            self._create_series(series)

        store = self._data[series]
        store["x"].append(x)
        store["y"].append(y)

        # Downsample when exceeding max_points.
        if len(store["x"]) > self._max_points:
            store["x"] = store["x"][::2]
            store["y"] = store["y"][::2]

        self._curves[series].setData(store["x"], store["y"])

    def clear_series(self, series: str) -> None:
        """Remove all data points from the named *series*.

        The series curve remains on the plot (empty) so that subsequent
        calls to :meth:`add_point` reuse the same colour.

        Args:
            series: Name of the series to clear.
        """
        if series in self._data:
            self._data[series]["x"].clear()
            self._data[series]["y"].clear()
            self._curves[series].setData([], [])

    def clear_all(self) -> None:
        """Remove all series and their data from the plot."""
        for name in list(self._data.keys()):
            self._plot_widget.removeItem(self._curves[name])
        self._data.clear()
        self._curves.clear()
        self._color_index = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_series(self, name: str) -> None:
        """Register a new series with the next palette colour."""
        color = _PALETTE[self._color_index % len(_PALETTE)]
        self._color_index += 1

        pen = pg.mkPen(color=color, width=2)
        curve = self._plot_widget.plot([], [], pen=pen, name=name)

        self._data[name] = {"x": [], "y": []}
        self._curves[name] = curve
