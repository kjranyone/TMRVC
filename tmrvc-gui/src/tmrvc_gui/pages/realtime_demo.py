"""Real-time voice conversion demo page using UCLM v2."""

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

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams

logger = logging.getLogger(__name__)


class RealtimeDemoPage(QWidget):
    """Real-time VC demo using unified UCLM v2 engine."""

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
        self.buffer_size_combo.addItems(["64", "128", "240", "256", "512"])
        self.buffer_size_combo.setCurrentIndex(2)  # default 240 (10ms)
        device_form.addRow("Buffer size:", self.buffer_size_combo)

        top_row.addWidget(device_group)

        # --- Model configuration ---
        model_group = QGroupBox("Model Configuration (UCLM v2)")
        model_form = QFormLayout(model_group)

        # UCLM Checkpoint
        uclm_row = QHBoxLayout()
        self.uclm_ckpt_edit = QLabel("Not selected")
        uclm_row.addWidget(self.uclm_ckpt_edit)
        self.btn_browse_uclm = QPushButton("UCLM...")
        uclm_row.addWidget(self.btn_browse_uclm)
        model_form.addRow("UCLM Checkpoint:", uclm_row)

        # Codec Checkpoint
        codec_row = QHBoxLayout()
        self.codec_ckpt_edit = QLabel("Not selected")
        codec_row.addWidget(self.codec_ckpt_edit)
        self.btn_browse_codec = QPushButton("Codec...")
        codec_row.addWidget(self.btn_browse_codec)
        model_form.addRow("Codec Checkpoint:", codec_row)

        # Speaker file
        speaker_row = QHBoxLayout()
        self.speaker_file_edit = QLabel("Not selected")
        speaker_row.addWidget(self.speaker_file_edit)
        self.btn_browse_speaker = QPushButton("Speaker...")
        speaker_row.addWidget(self.btn_browse_speaker)
        model_form.addRow("Speaker file:", speaker_row)

        top_row.addWidget(model_group)
        layout.addLayout(top_row)

        # --- Control Row ---
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

        # --- Monitoring & Style ---
        mid_row = QHBoxLayout()

        # Monitoring
        monitor_group = QGroupBox("Monitoring")
        monitor_layout = QVBoxLayout(monitor_group)
        
        monitor_layout.addWidget(QLabel("Input Level"))
        self.input_meter = QProgressBar()
        self.input_meter.setRange(0, 100)
        monitor_layout.addWidget(self.input_meter)

        monitor_layout.addWidget(QLabel("Output Level"))
        self.output_meter = QProgressBar()
        self.output_meter.setRange(0, 100)
        monitor_layout.addWidget(self.output_meter)

        self.inference_time_label = QLabel("Inf: -- ms")
        monitor_layout.addWidget(self.inference_time_label)
        mid_row.addWidget(monitor_group)

        # Physical Style Sliders
        style_group = QGroupBox("UCLM Physical Style")
        style_layout = QFormLayout(style_group)
        
        self.breathiness_slider = self._make_style_slider()
        style_layout.addRow("Breathiness:", self.breathiness_slider)
        
        self.tension_slider = self._make_style_slider()
        style_layout.addRow("Tension:", self.tension_slider)
        
        self.arousal_slider = self._make_style_slider()
        style_layout.addRow("Arousal:", self.arousal_slider)
        
        self.energy_slider = self._make_style_slider()
        style_layout.addRow("Energy:", self.energy_slider)
        
        mid_row.addWidget(style_group)
        layout.addLayout(mid_row)

        # Audio mix
        mix_row = QHBoxLayout()
        self.dry_wet_slider = QSlider(Qt.Orientation.Horizontal)
        self.dry_wet_slider.setRange(0, 100)
        self.dry_wet_slider.setValue(100)
        mix_row.addWidget(QLabel("Dry/Wet:"))
        mix_row.addWidget(self.dry_wet_slider)
        
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setRange(-24, 12)
        self.gain_slider.setValue(0)
        mix_row.addWidget(QLabel("Gain (dB):"))
        mix_row.addWidget(self.gain_slider)
        layout.addLayout(mix_row)

        self._uclm_path = None
        self._codec_path = None
        self._speaker_path = None

    def _make_style_slider(self) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        slider.valueChanged.connect(self._on_style_changed)
        return slider

    def _connect_signals(self) -> None:
        self.btn_browse_uclm.clicked.connect(self._on_browse_uclm)
        self.btn_browse_codec.clicked.connect(self._on_browse_codec)
        self.btn_browse_speaker.clicked.connect(self._on_browse_speaker)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.dry_wet_slider.valueChanged.connect(self._on_dry_wet_changed)
        self.gain_slider.valueChanged.connect(self._on_gain_changed)

    def _populate_audio_devices(self) -> None:
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            self.input_device_combo.clear()
            self.output_device_combo.clear()
            self.input_device_combo.addItem("(Default)", None)
            self.output_device_combo.addItem("(Default)", None)
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0:
                    self.input_device_combo.addItem(f"[{i}] {dev['name']}", i)
                if dev["max_output_channels"] > 0:
                    self.output_device_combo.addItem(f"[{i}] {dev['name']}", i)
        except Exception:
            pass

    def _on_browse_uclm(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select UCLM Checkpoint", "", "PyTorch (*.pt)")
        if p:
            self._uclm_path = Path(p)
            self.uclm_ckpt_edit.setText(self._uclm_path.name)

    def _on_browse_codec(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Codec Checkpoint", "", "PyTorch (*.pt)")
        if p:
            self._codec_path = Path(p)
            self.codec_ckpt_edit.setText(self._codec_path.name)

    def _on_browse_speaker(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Speaker", "", "TMRVC (*.tmrvc_speaker *.npy)")
        if p:
            self._speaker_path = Path(p)
            self.speaker_file_edit.setText(self._speaker_path.name)

    def _on_start(self):
        if not all([self._uclm_path, self._codec_path, self._speaker_path]):
            QMessageBox.warning(self, "Error", "Missing model or speaker configuration.")
            return

        from tmrvc_gui.workers.audio_engine import AudioEngine
        self._engine = AudioEngine(self._uclm_path, self._codec_path, self._speaker_path, parent=self)
        self._engine.level_updated.connect(self._on_level_updated)
        self._engine.timing_updated.connect(self._on_timing_updated)
        self._engine.error.connect(lambda m: QMessageBox.critical(self, "Engine Error", m))
        
        input_dev = self.input_device_combo.currentData()
        output_dev = self.output_device_combo.currentData()
        buffer_size = int(self.buffer_size_combo.currentText())

        self._engine.set_dry_wet(self.dry_wet_slider.value() / 100.0)
        self._engine.set_output_gain(float(self.gain_slider.value()))
        self._on_style_changed() # Sync initial style

        self._engine.start()
        self._engine.start_stream(input_dev, output_dev, buffer_size)

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _on_stop(self):
        if self._engine:
            self._engine.stop()
            self._engine.wait(2000)
            self._engine = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_level_updated(self, in_db, out_db):
        self.input_meter.setValue(int((in_db + 60) * 100 / 60))
        self.output_meter.setValue(int((out_db + 60) * 100 / 60))

    def _on_timing_updated(self, ms):
        self.inference_time_label.setText(f"Inf: {ms:.1f} ms")

    def _on_style_changed(self):
        if self._engine:
            style = StyleParams(
                breathiness=self.breathiness_slider.value() / 100.0,
                tension=self.tension_slider.value() / 100.0,
                arousal=self.arousal_slider.value() / 100.0,
                energy=self.energy_slider.value() / 100.0,
            )
            self._engine.set_style(style)

    def _on_dry_wet_changed(self, v):
        if self._engine: self._engine.set_dry_wet(v / 100.0)

    def _on_gain_changed(self, v):
        if self._engine: self._engine.set_output_gain(float(v))
