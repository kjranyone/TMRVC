"""UCLM Training page: monitor dual-stream token model training."""

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


class TokenTrainPage(QWidget):
    """Monitor and control UCLM (dual-stream) training."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Status & Control ---
        status_group = QGroupBox("UCLM Training Status")
        status_layout = QHBoxLayout(status_group)

        info_form = QFormLayout()
        self.step_label = QLabel("0")
        self.epoch_label = QLabel("0")
        self.mode_label = QLabel("Idle") # TTS or VC
        info_form.addRow("Step:", self.step_label)
        info_form.addRow("Epoch:", self.epoch_label)
        info_form.addRow("Current Mode:", self.mode_label)
        status_layout.addLayout(info_form)

        ctrl_layout = QVBoxLayout()
        self.btn_start = QPushButton("Start Training")
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        status_layout.addLayout(ctrl_layout)

        layout.addWidget(status_group)

        # --- Metrics Plot ---
        metrics_group = QGroupBox("Loss Metrics (Dual-Stream)")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.loss_plot = MetricPlotWidget()
        # Track A_t, B_t, and Total loss
        self.loss_plot.add_metric("loss_total", color="white")
        self.loss_plot.add_metric("loss_a", color="cyan") # Acoustic
        self.loss_plot.add_metric("loss_b", color="yellow") # Control
        self.loss_plot.add_metric("loss_vq", color="magenta") # IB Bottleneck
        self.loss_plot.add_metric("loss_adv", color="red") # Disentanglement
        
        metrics_layout.addWidget(self.loss_plot)
        layout.addWidget(metrics_group)

        # --- Training Progress ---
        progress_group = QGroupBox("Epoch Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.epoch_progress = QProgressBar()
        progress_layout.addWidget(self.epoch_progress)
        layout.addWidget(progress_group)

        # --- Log ---
        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group)

    def update_metrics(self, metrics: dict[str, float]) -> None:
        """Update labels and plots with latest training metrics."""
        self.step_label.setText(str(int(metrics.get("step", 0))))
        self.epoch_label.setText(str(int(metrics.get("epoch", 0))))
        
        mode = "TTS" if metrics.get("mode", 0) > 0.5 else "VC"
        self.mode_label.setText(mode)

        self.loss_plot.add_data_point("loss_total", metrics.get("loss", 0))
        self.loss_plot.add_data_point("loss_a", metrics.get("loss_a", 0))
        self.loss_plot.add_data_point("loss_b", metrics.get("loss_b", 0))
        if "loss_vq" in metrics:
            self.loss_plot.add_data_point("loss_vq", metrics["loss_vq"])
        if "loss_adv" in metrics:
            self.loss_plot.add_data_point("loss_adv", metrics["loss_adv"])

    def append_log(self, text: str) -> None:
        self.log_edit.append(text)
