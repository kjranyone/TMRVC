"""TTS page: text-to-speech generation with character/style control."""

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

    Provides text input, character/speaker selection, emotion/style sliders,
    and generates audio via the TTS pipeline.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = None
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
        model_group = QGroupBox("Character & Model")
        model_form = QFormLayout(model_group)

        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja (Japanese)", "en (English)"])
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

        # TTS checkpoint
        ckpt_row = QHBoxLayout()
        self.tts_ckpt_edit = QLineEdit()
        self.tts_ckpt_edit.setPlaceholderText("checkpoints/tts/tts_step200000.pt")
        ckpt_row.addWidget(self.tts_ckpt_edit)
        btn_browse_ckpt = QPushButton("Browse...")
        btn_browse_ckpt.clicked.connect(self._on_browse_tts_ckpt)
        ckpt_row.addWidget(btn_browse_ckpt)
        model_form.addRow("TTS checkpoint:", ckpt_row)

        # VC checkpoint (Converter + Vocoder)
        vc_row = QHBoxLayout()
        self.vc_ckpt_edit = QLineEdit()
        self.vc_ckpt_edit.setPlaceholderText("checkpoints/distill/best.pt")
        vc_row.addWidget(self.vc_ckpt_edit)
        btn_browse_vc = QPushButton("Browse...")
        btn_browse_vc.clicked.connect(self._on_browse_vc_ckpt)
        vc_row.addWidget(btn_browse_vc)
        model_form.addRow("VC checkpoint:", vc_row)

        left_col.addWidget(model_group)
        top_row.addLayout(left_col, stretch=2)

        # --- Right: Style controls ---
        right_col = QVBoxLayout()

        style_group = QGroupBox("Emotion & Style")
        style_form = QFormLayout(style_group)

        self.emotion_combo = QComboBox()
        self.emotion_combo.addItems(EMOTION_OPTIONS)
        style_form.addRow("Emotion:", self.emotion_combo)

        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.1)
        self.speed_spin.setValue(1.0)
        style_form.addRow("Speed:", self.speed_spin)

        # VAD sliders
        self.valence_slider = self._make_slider("Valence")
        style_form.addRow("Valence:", self.valence_slider)

        self.arousal_slider = self._make_slider("Arousal")
        style_form.addRow("Arousal:", self.arousal_slider)

        self.energy_slider = self._make_slider("Energy")
        style_form.addRow("Energy:", self.energy_slider)

        self.pitch_range_slider = self._make_slider("Pitch range")
        style_form.addRow("Pitch range:", self.pitch_range_slider)

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

    def _make_slider(self, name: str) -> QSlider:
        """Create a horizontal slider [-100, 100] mapped to [-1.0, 1.0]."""
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(-100, 100)
        slider.setValue(0)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(25)
        return slider

    def _slider_value(self, slider: QSlider) -> float:
        return slider.value() / 100.0

    def _get_language(self) -> str:
        return "ja" if self.language_combo.currentIndex() == 0 else "en"

    def append_log(self, text: str) -> None:
        self.log_edit.append(text)

    # --- Slots ---

    def _on_browse_speaker(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Speaker File", "",
            "Speaker Files (*.tmrvc_speaker);;All Files (*)",
        )
        if path:
            self.speaker_edit.setText(path)

    def _on_browse_tts_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select TTS Checkpoint", "",
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.tts_ckpt_edit.setText(path)

    def _on_browse_vc_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select VC Checkpoint", "",
            "PyTorch Checkpoint (*.pt);;All Files (*)",
        )
        if path:
            self.vc_ckpt_edit.setText(path)

    def _on_generate(self) -> None:
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.append_log("ERROR: No text entered.")
            return

        tts_ckpt = self.tts_ckpt_edit.text().strip()
        if not tts_ckpt:
            self.append_log("ERROR: No TTS checkpoint selected.")
            return

        self.append_log(f"Generating: \"{text[:50]}...\"")
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        config = {
            "text": text,
            "language": self._get_language(),
            "tts_checkpoint": tts_ckpt,
            "vc_checkpoint": self.vc_ckpt_edit.text().strip() or None,
            "speaker_file": self.speaker_edit.text().strip() or None,
            "speed": self.speed_spin.value(),
            "emotion": EMOTION_OPTIONS[self.emotion_combo.currentIndex()],
            "valence": self._slider_value(self.valence_slider),
            "arousal": self._slider_value(self.arousal_slider),
            "energy": self._slider_value(self.energy_slider),
            "pitch_range": self._slider_value(self.pitch_range_slider),
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
        if success:
            self.btn_play.setEnabled(True)
            self.btn_save.setEnabled(True)
        self._worker = None

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Audio", "output.wav",
            "WAV Files (*.wav);;All Files (*)",
        )
        if path:
            self.append_log(f"Saving to {path}...")
            # TODO: save last generated audio to path
