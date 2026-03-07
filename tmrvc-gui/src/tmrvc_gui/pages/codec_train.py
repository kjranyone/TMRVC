"""Codec Train page: monitor Emotion-Aware Codec training."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tmrvc_gui.widgets.metric_plot import MetricPlotWidget


class CodecTrainPage(QWidget):
    """Monitor Emotion-Aware Codec training (Token Spec)."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        status_group = QGroupBox("Codec Training Status")
        status_layout = QHBoxLayout(status_group)

        info_form = QFormLayout()
        self.step_label = QLabel("0")
        self.epoch_label = QLabel("0")
        info_form.addRow("Step:", self.step_label)
        info_form.addRow("Epoch:", self.epoch_label)
        status_layout.addLayout(info_form)

        ctrl_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        status_layout.addLayout(ctrl_layout)

        layout.addWidget(status_group)

        # Plot
        metrics_group = QGroupBox("Reconstruction & Token Losses")
        metrics_layout = QVBoxLayout(metrics_group)
        self.loss_plot = MetricPlotWidget()
        self.loss_plot.add_metric("loss_total", color="white")
        self.loss_plot.add_metric("loss_stft", color="cyan")
        self.loss_plot.add_metric("loss_vq", color="magenta")
        self.loss_plot.add_metric("loss_control", color="yellow") # B_t reconstruction
        
        metrics_layout.addWidget(self.loss_plot)
        layout.addWidget(metrics_group)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(100)
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log_edit)

    def update_metrics(self, metrics: dict[str, float]) -> None:
        self.step_label.setText(str(int(metrics.get("step", 0))))
        self.epoch_label.setText(str(int(metrics.get("epoch", 0))))
        
        self.loss_plot.add_data_point("loss_total", metrics.get("loss", 0))
        self.loss_plot.add_data_point("loss_stft", metrics.get("loss_stft", 0))
        self.loss_plot.add_data_point("loss_vq", metrics.get("loss_vq", 0))
        self.loss_plot.add_data_point("loss_control", metrics.get("loss_control", 0))
