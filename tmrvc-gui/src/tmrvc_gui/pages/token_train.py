"""Token Model training page for Mamba-based streaming token prediction."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _make_plot_placeholder(title: str) -> QFrame:
    frame = QFrame()
    frame.setFrameShape(QFrame.Shape.StyledPanel)
    frame.setMinimumSize(200, 120)
    frame_layout = QVBoxLayout(frame)
    label = QLabel(f"Plot: {title}")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    frame_layout.addWidget(label)
    return frame


class TokenTrainPage(QWidget):
    """Token Model training configuration and monitoring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()

        config_group = QGroupBox("Training Configuration")
        config_form = QFormLayout(config_group)

        self._codec_ckpt = QLineEdit()
        self._codec_ckpt.setPlaceholderText("checkpoints/codec/codec_final.pt")
        codec_btn = QPushButton("Browse...")
        codec_btn.clicked.connect(self._browse_codec)
        codec_row = QHBoxLayout()
        codec_row.addWidget(self._codec_ckpt)
        codec_row.addWidget(codec_btn)
        config_form.addRow("Codec Checkpoint:", codec_row)

        self._cache_dir = QLineEdit()
        self._cache_dir.setPlaceholderText("data/cache")
        self._cache_dir.setText("data/cache")
        cache_btn = QPushButton("Browse...")
        cache_btn.clicked.connect(self._browse_cache_dir)
        cache_row = QHBoxLayout()
        cache_row.addWidget(self._cache_dir)
        cache_row.addWidget(cache_btn)
        config_form.addRow("Cache Dir:", cache_row)

        self._output_dir = QLineEdit()
        self._output_dir.setPlaceholderText("checkpoints/token_model")
        self._output_dir.setText("checkpoints/token_model")
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        output_row.addWidget(output_btn)
        config_form.addRow("Output Dir:", output_row)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 128)
        self._batch_size.setValue(32)
        config_form.addRow("Batch Size:", self._batch_size)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-6, 1e-2)
        self._lr.setDecimals(6)
        self._lr.setValue(1e-4)
        self._lr.setSingleStep(1e-5)
        config_form.addRow("Learning Rate:", self._lr)

        self._steps = QSpinBox()
        self._steps.setRange(1000, 1000000)
        self._steps.setValue(200000)
        self._steps.setSingleStep(10000)
        config_form.addRow("Steps:", self._steps)

        self._device = QComboBox()
        self._device.addItems(["cuda", "xpu", "cpu"])
        config_form.addRow("Device:", self._device)

        top_row.addWidget(config_group)

        model_group = QGroupBox("Model Architecture")
        model_form = QFormLayout(model_group)

        self._model_type = QComboBox()
        self._model_type.addItems(["mamba", "transformer"])
        model_form.addRow("Model Type:", self._model_type)

        self._d_model = QSpinBox()
        self._d_model.setRange(64, 512)
        self._d_model.setValue(256)
        self._d_model.setSingleStep(64)
        model_form.addRow("Hidden Dimension:", self._d_model)

        self._n_layers = QSpinBox()
        self._n_layers.setRange(2, 24)
        self._n_layers.setValue(6)
        model_form.addRow("Layers:", self._n_layers)

        self._context_length = QSpinBox()
        self._context_length.setRange(1, 50)
        self._context_length.setValue(10)
        model_form.addRow("Context Length (frames):", self._context_length)

        self._temperature = QDoubleSpinBox()
        self._temperature.setRange(0.1, 2.0)
        self._temperature.setValue(1.0)
        self._temperature.setSingleStep(0.1)
        model_form.addRow("Sampling Temperature:", self._temperature)

        top_row.addWidget(model_group)
        layout.addLayout(top_row)

        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QHBoxLayout(metrics_group)
        metrics_layout.addWidget(_make_plot_placeholder("Token Loss"))
        metrics_layout.addWidget(_make_plot_placeholder("Accuracy"))
        metrics_layout.addWidget(_make_plot_placeholder("Perplexity"))
        layout.addWidget(metrics_group)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Training")
        self._start_btn.setObjectName("primary")
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        btn_row.addStretch()
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        log_group = QGroupBox("Training Log")
        log_layout = QVBoxLayout(log_group)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setObjectName("terminal")
        log_layout.addWidget(self._log_view)
        layout.addWidget(log_group)

    def _browse_codec(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Codec Checkpoint", "", "PyTorch Checkpoint (*.pt)"
        )
        if path:
            self._codec_ckpt.setText(path)

    def _browse_cache_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Cache Directory")
        if path:
            self._cache_dir.setText(path)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self._output_dir.setText(path)

    def get_config(self) -> dict:
        return {
            "codec_checkpoint": self._codec_ckpt.text(),
            "cache_dir": self._cache_dir.text(),
            "output_dir": self._output_dir.text(),
            "batch_size": self._batch_size.value(),
            "lr": self._lr.value(),
            "steps": self._steps.value(),
            "device": self._device.currentText(),
            "model_type": self._model_type.currentText(),
            "d_model": self._d_model.value(),
            "n_layers": self._n_layers.value(),
            "context_length": self._context_length.value(),
            "temperature": self._temperature.value(),
        }
