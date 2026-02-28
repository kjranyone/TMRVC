"""Codec training page for Streaming Neural Audio Codec."""

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


class CodecTrainPage(QWidget):
    """Streaming Codec training configuration and monitoring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()

        config_group = QGroupBox("Codec Configuration")
        config_form = QFormLayout(config_group)

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
        self._output_dir.setPlaceholderText("checkpoints/codec")
        self._output_dir.setText("checkpoints/codec")
        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._browse_output_dir)
        output_row = QHBoxLayout()
        output_row.addWidget(self._output_dir)
        output_row.addWidget(output_btn)
        config_form.addRow("Output Dir:", output_row)

        self._batch_size = QSpinBox()
        self._batch_size.setRange(1, 128)
        self._batch_size.setValue(16)
        config_form.addRow("Batch Size:", self._batch_size)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-6, 1e-2)
        self._lr.setDecimals(6)
        self._lr.setValue(3e-4)
        self._lr.setSingleStep(1e-5)
        config_form.addRow("Learning Rate:", self._lr)

        self._steps = QSpinBox()
        self._steps.setRange(1000, 1000000)
        self._steps.setValue(100000)
        self._steps.setSingleStep(10000)
        config_form.addRow("Steps:", self._steps)

        self._device = QComboBox()
        self._device.addItems(["cuda", "xpu", "cpu"])
        config_form.addRow("Device:", self._device)

        top_row.addWidget(config_group)

        codec_group = QGroupBox("Codec Parameters")
        codec_form = QFormLayout(codec_group)

        self._frame_size = QSpinBox()
        self._frame_size.setRange(160, 960)
        self._frame_size.setValue(480)
        self._frame_size.setSingleStep(80)
        codec_form.addRow("Frame Size (samples):", self._frame_size)

        self._n_codebooks = QSpinBox()
        self._n_codebooks.setRange(1, 8)
        self._n_codebooks.setValue(4)
        codec_form.addRow("RVQ Codebooks:", self._n_codebooks)

        self._codebook_size = QSpinBox()
        self._codebook_size.setRange(256, 4096)
        self._codebook_size.setValue(1024)
        self._codebook_size.setSingleStep(256)
        codec_form.addRow("Codebook Size:", self._codebook_size)

        top_row.addWidget(codec_group)
        layout.addLayout(top_row)

        loss_group = QGroupBox("Loss Weights")
        loss_form = QFormLayout(loss_group)

        self._lambda_rec = QDoubleSpinBox()
        self._lambda_rec.setRange(0.0, 10.0)
        self._lambda_rec.setValue(1.0)
        loss_form.addRow("λ Reconstruction:", self._lambda_rec)

        self._lambda_adv = QDoubleSpinBox()
        self._lambda_adv.setRange(0.0, 10.0)
        self._lambda_adv.setValue(1.0)
        loss_form.addRow("λ Adversarial:", self._lambda_adv)

        self._lambda_stft = QDoubleSpinBox()
        self._lambda_stft.setRange(0.0, 10.0)
        self._lambda_stft.setValue(1.0)
        loss_form.addRow("λ STFT:", self._lambda_stft)

        self._lambda_commit = QDoubleSpinBox()
        self._lambda_commit.setRange(0.0, 1.0)
        self._lambda_commit.setValue(0.25)
        self._lambda_commit.setSingleStep(0.05)
        loss_form.addRow("λ Commitment:", self._lambda_commit)

        layout.addWidget(loss_group)

        metrics_group = QGroupBox("Training Metrics")
        metrics_layout = QHBoxLayout(metrics_group)
        metrics_layout.addWidget(_make_plot_placeholder("Generator Loss"))
        metrics_layout.addWidget(_make_plot_placeholder("Discriminator Loss"))
        metrics_layout.addWidget(_make_plot_placeholder("STFT Loss"))
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
            "cache_dir": self._cache_dir.text(),
            "output_dir": self._output_dir.text(),
            "batch_size": self._batch_size.value(),
            "lr": self._lr.value(),
            "steps": self._steps.value(),
            "device": self._device.currentText(),
            "frame_size": self._frame_size.value(),
            "n_codebooks": self._n_codebooks.value(),
            "codebook_size": self._codebook_size.value(),
            "lambda_rec": self._lambda_rec.value(),
            "lambda_adv": self._lambda_adv.value(),
            "lambda_stft": self._lambda_stft.value(),
            "lambda_commit": self._lambda_commit.value(),
        }
