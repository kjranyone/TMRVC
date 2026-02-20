"""Real-time voice conversion demo page with audio I/O and monitoring."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from tmrvc_core.constants import HOP_LENGTH, SAMPLE_RATE

logger = logging.getLogger(__name__)


class RealtimeDemoPage(QWidget):
    """Real-time voice conversion demo with audio device selection and monitoring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._engine = None
        self._setup_ui()
        self._connect_signals()
        self._populate_audio_devices()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()

        # --- Audio device configuration ---
        device_group = QGroupBox("Audio Configuration")
        device_form = QFormLayout(device_group)

        self.input_device_combo = QComboBox()
        device_form.addRow("Input device:", self.input_device_combo)

        self.output_device_combo = QComboBox()
        device_form.addRow("Output device:", self.output_device_combo)

        self.buffer_size_combo = QComboBox()
        self.buffer_size_combo.addItems(["64", "128", "256", "512", "1024"])
        self.buffer_size_combo.setCurrentIndex(2)  # default 256
        device_form.addRow("Buffer size:", self.buffer_size_combo)

        top_row.addWidget(device_group)

        # --- Model configuration ---
        model_group = QGroupBox("Model Configuration")
        model_form = QFormLayout(model_group)

        model_dir_row = QHBoxLayout()
        self.model_dir_combo = QComboBox()
        self.model_dir_combo.setEditable(True)
        self.model_dir_combo.setPlaceholderText("Path to ONNX model directory")
        model_dir_row.addWidget(self.model_dir_combo)
        self.btn_browse_model = QPushButton("Browse...")
        model_dir_row.addWidget(self.btn_browse_model)
        model_form.addRow("ONNX models:", model_dir_row)

        speaker_row = QHBoxLayout()
        self.speaker_file_combo = QComboBox()
        self.speaker_file_combo.setEditable(True)
        self.speaker_file_combo.setPlaceholderText("Path to .tmrvc_speaker file")
        speaker_row.addWidget(self.speaker_file_combo)
        self.btn_browse_speaker = QPushButton("Browse...")
        speaker_row.addWidget(self.btn_browse_speaker)
        model_form.addRow("Speaker file:", speaker_row)

        top_row.addWidget(model_group)
        layout.addLayout(top_row)

        # --- Start / Stop ---
        ctrl_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_start.setMinimumWidth(120)
        ctrl_row.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setMinimumWidth(120)
        self.btn_stop.setEnabled(False)
        ctrl_row.addWidget(self.btn_stop)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # --- Monitoring ---
        monitor_group = QGroupBox("Monitoring")
        monitor_layout = QHBoxLayout(monitor_group)

        # Level meters
        meters_col = QVBoxLayout()

        meters_col.addWidget(QLabel("Input Level"))
        self.input_meter = QProgressBar()
        self.input_meter.setRange(0, 100)
        self.input_meter.setValue(0)
        self.input_meter.setTextVisible(False)
        meters_col.addWidget(self.input_meter)

        meters_col.addWidget(QLabel("Output Level"))
        self.output_meter = QProgressBar()
        self.output_meter.setRange(0, 100)
        self.output_meter.setValue(0)
        self.output_meter.setTextVisible(False)
        meters_col.addWidget(self.output_meter)

        monitor_layout.addLayout(meters_col)

        # Stats
        stats_col = QFormLayout()

        self.inference_time_label = QLabel("-- ms")
        stats_col.addRow("Inference time:", self.inference_time_label)

        self.latency_label = QLabel("-- ms")
        stats_col.addRow("Total latency:", self.latency_label)

        self.buffer_status_label = QLabel("Idle")
        stats_col.addRow("Buffer status:", self.buffer_status_label)

        monitor_layout.addLayout(stats_col)

        # Sliders
        sliders_col = QFormLayout()

        self.dry_wet_slider = QSlider(Qt.Orientation.Horizontal)
        self.dry_wet_slider.setRange(0, 100)
        self.dry_wet_slider.setValue(100)
        self.dry_wet_label = QLabel("100% wet")
        dry_wet_row = QHBoxLayout()
        dry_wet_row.addWidget(self.dry_wet_slider)
        dry_wet_row.addWidget(self.dry_wet_label)
        sliders_col.addRow("Dry / Wet:", dry_wet_row)

        self.output_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.output_gain_slider.setRange(-24, 12)
        self.output_gain_slider.setValue(0)
        self.gain_label = QLabel("0 dB")
        gain_row = QHBoxLayout()
        gain_row.addWidget(self.output_gain_slider)
        gain_row.addWidget(self.gain_label)
        sliders_col.addRow("Output gain:", gain_row)

        self.voice_preset_slider = QSlider(Qt.Orientation.Horizontal)
        self.voice_preset_slider.setRange(0, 100)
        self.voice_preset_slider.setValue(0)
        self.voice_preset_label = QLabel("0%")
        voice_preset_row = QHBoxLayout()
        voice_preset_row.addWidget(self.voice_preset_slider)
        voice_preset_row.addWidget(self.voice_preset_label)
        sliders_col.addRow("Voice preset:", voice_preset_row)

        monitor_layout.addLayout(sliders_col)
        layout.addWidget(monitor_group)

        # --- Waveform displays ---
        wave_group = QGroupBox("Waveform Display")
        wave_layout = QHBoxLayout(wave_group)

        self.input_waveform_frame = QFrame()
        self.input_waveform_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.input_waveform_frame.setMinimumHeight(80)
        iwf_layout = QVBoxLayout(self.input_waveform_frame)
        iwf_label = QLabel("Input Waveform")
        iwf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        iwf_layout.addWidget(iwf_label)
        wave_layout.addWidget(self.input_waveform_frame)

        self.output_waveform_frame = QFrame()
        self.output_waveform_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.output_waveform_frame.setMinimumHeight(80)
        owf_layout = QVBoxLayout(self.output_waveform_frame)
        owf_label = QLabel("Output Waveform")
        owf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        owf_layout.addWidget(owf_label)
        wave_layout.addWidget(self.output_waveform_frame)

        layout.addWidget(wave_group)

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        self.btn_browse_model.clicked.connect(self._on_browse_model)
        self.btn_browse_speaker.clicked.connect(self._on_browse_speaker)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.dry_wet_slider.valueChanged.connect(self._on_dry_wet_changed)
        self.output_gain_slider.valueChanged.connect(self._on_gain_changed)
        self.voice_preset_slider.valueChanged.connect(self._on_voice_preset_changed)

    def _populate_audio_devices(self) -> None:
        """Enumerate audio devices using sounddevice."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            self.input_device_combo.clear()
            self.output_device_combo.clear()

            self.input_device_combo.addItem("(System Default)", None)
            self.output_device_combo.addItem("(System Default)", None)

            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    self.input_device_combo.addItem(
                        f"[{i}] {dev['name']}", i,
                    )
                if dev["max_output_channels"] > 0:
                    self.output_device_combo.addItem(
                        f"[{i}] {dev['name']}", i,
                    )
        except Exception as exc:
            logger.warning("Failed to enumerate audio devices: %s", exc)
            self.input_device_combo.clear()
            self.input_device_combo.addItem("(No devices detected)")
            self.output_device_combo.clear()
            self.output_device_combo.addItem("(No devices detected)")

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_model(self) -> None:
        """Browse for ONNX model directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select ONNX Model Directory",
        )
        if path:
            self.model_dir_combo.setCurrentText(path)

    def _on_browse_speaker(self) -> None:
        """Browse for .tmrvc_speaker file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Speaker File", "",
            "Speaker Files (*.tmrvc_speaker);;All Files (*)",
        )
        if path:
            self.speaker_file_combo.setCurrentText(path)

    def _on_start(self) -> None:
        """Start the audio engine."""
        model_dir = self.model_dir_combo.currentText().strip()
        speaker_path = self.speaker_file_combo.currentText().strip()

        if not model_dir or not speaker_path:
            QMessageBox.warning(
                self, "Missing Configuration",
                "Please specify both ONNX model directory and speaker file.",
            )
            return

        model_dir_path = Path(model_dir)
        speaker_file_path = Path(speaker_path)

        if not model_dir_path.is_dir():
            QMessageBox.warning(self, "Error", f"Model directory not found: {model_dir}")
            return
        if not speaker_file_path.is_file():
            QMessageBox.warning(self, "Error", f"Speaker file not found: {speaker_path}")
            return

        from tmrvc_gui.workers.audio_engine import AudioEngine

        self._engine = AudioEngine(model_dir_path, speaker_file_path, parent=self)

        # Connect engine signals
        self._engine.level_updated.connect(self._on_level_updated)
        self._engine.timing_updated.connect(self._on_timing_updated)
        self._engine.buffer_status.connect(self._on_buffer_status)
        self._engine.error.connect(self._on_engine_error)
        self._engine.finished.connect(self._on_engine_finished)

        # Get audio device selections
        input_dev = self.input_device_combo.currentData()
        output_dev = self.output_device_combo.currentData()
        buffer_size = int(self.buffer_size_combo.currentText())

        # Apply current slider values
        self._engine.set_dry_wet(self.dry_wet_slider.value() / 100.0)
        self._engine.set_output_gain(float(self.output_gain_slider.value()))

        # Start the engine thread (models load in run(), then start_stream)
        self._engine.start()

        # Start audio stream after a brief delay for model loading
        try:
            self._engine.start_stream(input_dev, output_dev, buffer_size)
        except Exception as exc:
            QMessageBox.critical(self, "Audio Error", str(exc))
            self._engine.stop()
            self._engine = None
            return

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.set_buffer_status("Running")

    def _on_stop(self) -> None:
        """Stop the audio engine."""
        if self._engine is not None:
            self._engine.stop()
            self._engine.wait(5000)
            self._engine = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.set_buffer_status("Idle")
        self.set_input_level(0)
        self.set_output_level(0)
        self.set_inference_time(0.0)

    def _on_level_updated(self, input_db: float, output_db: float) -> None:
        """Handle level meter updates from the engine."""
        # Convert dB to 0-100 range (-60dB=0, 0dB=100)
        in_pct = max(0, min(100, int((input_db + 60.0) * 100.0 / 60.0)))
        out_pct = max(0, min(100, int((output_db + 60.0) * 100.0 / 60.0)))
        self.set_input_level(in_pct)
        self.set_output_level(out_pct)

    def _on_timing_updated(self, ms: float) -> None:
        """Handle inference timing updates from the engine."""
        self.set_inference_time(ms)
        # Total latency = buffer + inference + 1 hop
        buffer_size = int(self.buffer_size_combo.currentText())
        buffer_ms = buffer_size / SAMPLE_RATE * 1000.0
        hop_ms = HOP_LENGTH / SAMPLE_RATE * 1000.0
        self.set_latency(buffer_ms + ms + hop_ms)

    def _on_buffer_status(self, underrun: bool) -> None:
        """Handle buffer underrun signals."""
        self.set_buffer_status("UNDERRUN" if underrun else "OK")

    def _on_engine_error(self, msg: str) -> None:
        """Handle engine error signals."""
        logger.error("AudioEngine error: %s", msg)
        QMessageBox.critical(self, "Engine Error", msg)
        self._on_stop()

    def _on_engine_finished(self) -> None:
        """Handle engine thread completion."""
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.set_buffer_status("Idle")

    def _on_dry_wet_changed(self, value: int) -> None:
        """Handle dry/wet slider changes."""
        self.dry_wet_label.setText(f"{value}% wet")
        if self._engine is not None:
            self._engine.set_dry_wet(value / 100.0)

    def _on_gain_changed(self, value: int) -> None:
        """Handle output gain slider changes."""
        self.gain_label.setText(f"{value} dB")
        if self._engine is not None:
            self._engine.set_output_gain(float(value))

    def _on_voice_preset_changed(self, value: int) -> None:
        """Handle voice preset alpha slider changes."""
        self.voice_preset_label.setText(f"{value}%")
        if self._engine is not None:
            self._engine.set_voice_source_alpha(value / 100.0)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_inference_time(self, ms: float) -> None:
        """Update the displayed inference time."""
        self.inference_time_label.setText(f"{ms:.1f} ms")

    def set_latency(self, ms: float) -> None:
        """Update the displayed total latency."""
        self.latency_label.setText(f"{ms:.1f} ms")

    def set_buffer_status(self, status: str) -> None:
        """Update the buffer status text."""
        self.buffer_status_label.setText(status)

    def set_input_level(self, value: int) -> None:
        """Set input level meter (0-100)."""
        self.input_meter.setValue(value)

    def set_output_level(self, value: int) -> None:
        """Set output level meter (0-100)."""
        self.output_meter.setValue(value)
