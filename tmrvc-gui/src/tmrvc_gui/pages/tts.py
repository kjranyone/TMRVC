"""TTS page: text-to-speech generation with character/style control (UCLM v2)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


EMOTION_OPTIONS = [
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "bored", "excited", "tender", "sarcastic", "whisper",
]


class TTSPage(QWidget):
    """Text-to-speech generation page.

    Provides text input, character/speaker selection, physical style sliders
    (UCLM v2), and generates audio via the unified UCLMEngine.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = None
        self._last_audio: "numpy.ndarray | None" = None
        self._last_sr: int = 24000
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        top_row = QHBoxLayout()

        # --- Left: Input & Character ---
        left_col = QVBoxLayout()

        # Text input
        input_group = QGroupBox("Text Input")
        input_layout = QVBoxLayout(input_group)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText(
            "Enter text to synthesize...\n"
            "例: こんにちは！今日もいい天気ですね。"
        )
        self.text_edit.setMaximumHeight(120)
        input_layout.addWidget(self.text_edit)

        left_col.addWidget(input_group)

        # Character / model selection
        model_group = QGroupBox("Character & Model (UCLM v2)")
        model_form = QFormLayout(model_group)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja (Japanese)", "en (English)", "zh (Chinese)", "ko (Korean)"])
        model_form.addRow("Language:", self.language_combo)

        # Speaker file
        spk_row = QHBoxLayout()
        self.speaker_edit = QLineEdit()
        self.speaker_edit.setPlaceholderText("models/speaker.tmrvc_speaker")
        spk_row.addWidget(self.speaker_edit)
        btn_browse_spk = QPushButton("Browse...")
        btn_browse_spk.clicked.connect(self._on_browse_speaker)
        spk_row.addWidget(btn_browse_spk)
        model_form.addRow("Speaker file:", spk_row)

        # UCLM checkpoint
        uclm_row = QHBoxLayout()
        self.uclm_ckpt_edit = QLineEdit()
        self.uclm_ckpt_edit.setPlaceholderText("checkpoints/uclm/uclm_latest.pt")
        uclm_row.addWidget(self.uclm_ckpt_edit)
        btn_browse_uclm = QPushButton("Browse...")
        btn_browse_uclm.clicked.connect(self._on_browse_uclm_ckpt)
        uclm_row.addWidget(btn_browse_uclm)
        model_form.addRow("UCLM checkpoint:", uclm_row)

        # Codec checkpoint
        codec_row = QHBoxLayout()
        self.codec_ckpt_edit = QLineEdit()
        self.codec_ckpt_edit.setPlaceholderText("checkpoints/codec/codec_latest.pt")
        codec_row.addWidget(self.codec_ckpt_edit)
        btn_browse_codec = QPushButton("Browse...")
        btn_browse_codec.clicked.connect(self._on_browse_codec_ckpt)
        codec_row.addWidget(btn_browse_codec)
        model_form.addRow("Codec checkpoint:", codec_row)

        left_col.addWidget(model_group)
        top_row.addLayout(left_col, stretch=2)

        # --- Right: Style controls ---
        right_col = QVBoxLayout()

        style_group = QGroupBox("Physical Style (UCLM)")
        style_form = QFormLayout(style_group)

        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(EMOTION_OPTIONS)
        style_form.addRow("Emotion:", self.emotion_combo)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        style_form.addRow("Speed:", self.speed_spin)

        # 8-dim Physical Sliders
        self.breathiness_slider = self._make_slider("Breathiness", 0, 100, 0)
        style_form.addRow("Breathiness:", self.breathiness_slider)

        self.tension_slider = self._make_slider("Tension", 0, 100, 0)
        style_form.addRow("Tension:", self.tension_slider)

        self.arousal_slider = self._make_slider("Arousal", 0, 100, 0)
        style_form.addRow("Arousal:", self.arousal_slider)

        self.valence_slider = self._make_slider("Valence", -100, 100, 0)
        style_form.addRow("Valence:", self.valence_slider)

        self.roughness_slider = self._make_slider("Roughness", 0, 100, 0)
        style_form.addRow("Roughness:", self.roughness_slider)

        self.voicing_slider = self._make_slider("Voicing", 0, 100, 100) # Default 1.0
        style_form.addRow("Voicing:", self.voicing_slider)

        self.energy_slider = self._make_slider("Energy", 0, 100, 0)
        style_form.addRow("Energy:", self.energy_slider)

        right_col.addWidget(style_group)
        top_row.addLayout(right_col, stretch=1)

        layout.addLayout(top_row)

        # --- Actions ---
        action_row = QHBoxLayout()

        self.btn_generate = QPushButton("Generate")
        self.btn_generate.setMinimumHeight(40)
        self.btn_generate.clicked.connect(self._on_generate)
        action_row.addWidget(self.btn_generate)

        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._on_play)
        action_row.addWidget(self.btn_play)

        self.btn_save = QPushButton("Save WAV...")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        action_row.addWidget(self.btn_save)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        action_row.addWidget(self.btn_cancel)

        layout.addLayout(action_row)

        # --- Log ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(150)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group)

    def _make_slider(self, name: str, min_val: int, max_val: int, default: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval((max_val - min_val) // 4)
        return slider

    def _slider_value(self, slider: QSlider) -> float:
        return slider.value() / 100.0

    def _get_language(self) -> str:
        idx = self.language_combo.currentIndex()
        return ["ja", "en", "zh", "ko"][idx]

    def append_log(self, text: str) -> None:
        self.log_edit.append(text)

    # --- Slots ---

    def _on_browse_speaker(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Speaker File", "",
            "Speaker/NPY Files (*.tmrvc_speaker *.npy);;All Files (*)",
        )
        if path:
            self.speaker_edit.setText(path)

    def _on_browse_uclm_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select UCLM Checkpoint", "",
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.uclm_ckpt_edit.setText(path)

    def _on_browse_codec_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Codec Checkpoint", "",
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.codec_ckpt_edit.setText(path)

    def _on_generate(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.append_log("ERROR: No text entered.")
            return

        uclm_ckpt = self.uclm_ckpt_edit.text().strip()
        codec_ckpt = self.codec_ckpt_edit.text().strip()
        if not uclm_ckpt or not codec_ckpt:
            self.append_log("ERROR: UCLM and Codec checkpoints required.")
            return

        self.append_log(f"Generating: \"{text[:50]}...\"")
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        config = {
            "text": text,
            "language": self._get_language(),
            "uclm_checkpoint": uclm_ckpt,
            "codec_checkpoint": codec_ckpt,
            "speaker_file": self.speaker_edit.text().strip() or None,
            "speed": self.speed_spin.value(),
            "emotion": EMOTION_OPTIONS[self.emotion_combo.currentIndex()],
            "breathiness": self._slider_value(self.breathiness_slider),
            "tension": self._slider_value(self.tension_slider),
            "arousal": self._slider_value(self.arousal_slider),
            "valence": self._slider_value(self.valence_slider),
            "roughness": self._slider_value(self.roughness_slider),
            "voicing": self._slider_value(self.voicing_slider),
            "energy": self._slider_value(self.energy_slider),
        }

        from tmrvc_gui.workers.tts_worker import TTSWorker

        self._worker = TTSWorker(config)
        self._worker.log_message.connect(self.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(lambda msg: self.append_log(f"ERROR: {msg}"))
        self._worker.start()

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.append_log(message)
        if success and self._worker is not None and self._worker.audio is not None:
            self._last_audio = self._worker.audio
            self.btn_play.setEnabled(True)
            self.btn_save.setEnabled(True)
        self._worker = None

    def _on_play(self) -> None:
        if self._last_audio is None:
            self.append_log("No audio to play.")
            return
        try:
            import sounddevice as sd
            sd.play(self._last_audio, self._last_sr)
            self.append_log("Playing audio...")
        except Exception as e:
            self.append_log(f"ERROR: Playback failed: {e}")

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "output.wav",
            "WAV Files (*.wav);;All Files (*)",
        )
        if not path:
            return
        if self._last_audio is None:
            self.append_log("ERROR: No audio to save.")
            return
        try:
            import soundfile as sf
            sf.write(path, self._last_audio, self._last_sr)
            self.append_log(f"Saved to {path}")
        except Exception as e:
            self.append_log(f"ERROR: Save failed: {e}")
