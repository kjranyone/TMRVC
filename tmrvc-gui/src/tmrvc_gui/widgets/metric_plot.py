"""MetricPlotWidget - Real-time metric plotting backed by pyqtgraph."""

from __future__ import annotations

from typing import Dict, List

import pyqtgraph as pg
from PySide6.QtWidgets import QVBoxLayout, QWidget


class MetricPlotWidget(QWidget):
    """Real-time line-chart widget for training metrics.

    Usage::

        plot = MetricPlotWidget()
        plot.add_metric("loss_total", color="white")
        plot.add_metric("loss_a", color="cyan")
        # during training loop:
        plot.add_data_point("loss_total", 0.42)
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._step_counters: Dict[str, int] = {}
        self._data: Dict[str, Dict[str, List[float]]] = {}
        self._curves: Dict[str, pg.PlotDataItem] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setLabel("bottom", "Step")
        self._plot_widget.setLabel("left", "Value")
        self._plot_widget.addLegend()
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self._plot_widget)

    def add_metric(self, name: str, color: str = "white") -> None:
        """Register a named metric series with the given colour."""
        if name in self._curves:
            return
        pen = pg.mkPen(color=color, width=2)
        curve = self._plot_widget.plot([], [], pen=pen, name=name)
        self._data[name] = {"x": [], "y": []}
        self._curves[name] = curve
        self._step_counters[name] = 0

    def add_data_point(self, name: str, value: float) -> None:
        """Append a data point to the named series (x auto-increments)."""
        if name not in self._data:
            self.add_metric(name)

        self._step_counters[name] += 1
        store = self._data[name]
        store["x"].append(float(self._step_counters[name]))
        store["y"].append(float(value))

        # Downsample when exceeding 10k points.
        if len(store["x"]) > 10000:
            store["x"] = store["x"][::2]
            store["y"] = store["y"][::2]

        self._curves[name].setData(store["x"], store["y"])

    def clear_all(self) -> None:
        """Remove all series and their data."""
        for name in list(self._curves):
            self._plot_widget.removeItem(self._curves[name])
        self._data.clear()
        self._curves.clear()
        self._step_counters.clear()
