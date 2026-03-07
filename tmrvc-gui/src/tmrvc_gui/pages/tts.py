"""TTS page: text-to-speech generation with character/style control (UCLM)."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

VOICE_STATE_DIMS = [
    "Pitch",
    "Formant",
    "Breathiness",
    "Tension",
    "Nasality",
    "Creakiness",
    "Loudness",
    "Rate",
]


EMOTION_OPTIONS = [
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "bored", "excited", "tender", "sarcastic", "whisper",
]


class TTSPage(QWidget):
    """Text-to-speech generation page.

    Provides text input, character/speaker selection, physical style sliders
    (UCLM), and generates audio via the unified UCLMEngine.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = None
        self._last_audio: "numpy.ndarray | None" = None
        self._last_sr: int = 24000
        self._compare_b_audio: "numpy.ndarray | None" = None
        self._compare_b_sr: int = 24000
        self._gallery: list[dict] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Session buttons (toolbar area) ---
        session_row = QHBoxLayout()
        self.btn_save_session = QPushButton("Save Session")
        self.btn_save_session.clicked.connect(self._on_save_session)
        session_row.addWidget(self.btn_save_session)

        self.btn_load_session = QPushButton("Load Session")
        self.btn_load_session.clicked.connect(self._on_load_session)
        session_row.addWidget(self.btn_load_session)

        session_row.addStretch()
        layout.addLayout(session_row)

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
        model_group = QGroupBox("Character & Model (UCLM)")
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

        # Voice cloning section
        clone_group = QGroupBox("Voice Cloning")
        clone_layout = QVBoxLayout(clone_group)

        ref_row = QHBoxLayout()
        self.ref_audio_edit = QLineEdit()
        self.ref_audio_edit.setPlaceholderText("Reference audio file for cloning...")
        ref_row.addWidget(self.ref_audio_edit)
        btn_browse_ref = QPushButton("Browse...")
        btn_browse_ref.clicked.connect(self._on_browse_ref_audio)
        ref_row.addWidget(btn_browse_ref)
        clone_layout.addLayout(ref_row)

        self.btn_extract_speaker = QPushButton("Extract Speaker")
        self.btn_extract_speaker.clicked.connect(self._on_extract_speaker)
        clone_layout.addWidget(self.btn_extract_speaker)

        left_col.addWidget(clone_group)

        # --- Casting Gallery section ---
        gallery_group = QGroupBox("Casting Gallery")
        gallery_layout = QVBoxLayout(gallery_group)

        self.gallery_list = QListWidget()
        self.gallery_list.setMaximumHeight(120)
        gallery_layout.addWidget(self.gallery_list)

        gallery_btn_row = QHBoxLayout()

        self.btn_gallery_save = QPushButton("Save to Gallery")
        self.btn_gallery_save.clicked.connect(self._on_gallery_save)
        gallery_btn_row.addWidget(self.btn_gallery_save)

        self.btn_gallery_load = QPushButton("Load from Gallery")
        self.btn_gallery_load.clicked.connect(self._on_gallery_load)
        gallery_btn_row.addWidget(self.btn_gallery_load)

        self.btn_gallery_export = QPushButton("Export Profile")
        self.btn_gallery_export.clicked.connect(self._on_gallery_export)
        gallery_btn_row.addWidget(self.btn_gallery_export)

        self.btn_gallery_import = QPushButton("Import Profile")
        self.btn_gallery_import.clicked.connect(self._on_gallery_import)
        gallery_btn_row.addWidget(self.btn_gallery_import)

        self.btn_gallery_delete = QPushButton("Delete")
        self.btn_gallery_delete.clicked.connect(self._on_gallery_delete)
        gallery_btn_row.addWidget(self.btn_gallery_delete)

        gallery_layout.addLayout(gallery_btn_row)

        left_col.addWidget(gallery_group)

        # Dialogue context
        ctx_group = QGroupBox("Dialogue Context")
        ctx_layout = QVBoxLayout(ctx_group)
        self.dialogue_context_edit = QTextEdit()
        self.dialogue_context_edit.setPlaceholderText(
            "Enter preceding dialogue lines for context-aware generation..."
        )
        self.dialogue_context_edit.setMaximumHeight(80)
        ctx_layout.addWidget(self.dialogue_context_edit)
        left_col.addWidget(ctx_group)

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

        # 8-D Voice State sliders
        voice_state_group = QGroupBox("8-D Voice State")
        voice_state_form = QFormLayout(voice_state_group)

        self.voice_state_sliders: list[QSlider] = []
        for dim_name in VOICE_STATE_DIMS:
            slider = self._make_slider(dim_name, -100, 100, 0)
            voice_state_form.addRow(f"{dim_name}:", slider)
            self.voice_state_sliders.append(slider)

        right_col.addWidget(voice_state_group)

        # CFG scale slider
        cfg_group = QGroupBox("Classifier-Free Guidance")
        cfg_form = QFormLayout(cfg_group)
        self.cfg_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.cfg_scale_slider.setRange(10, 50)
        self.cfg_scale_slider.setValue(10)
        self.cfg_scale_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.cfg_scale_slider.setTickInterval(10)
        self.cfg_scale_label = QLabel("1.0")
        self.cfg_scale_slider.valueChanged.connect(
            lambda v: self.cfg_scale_label.setText(f"{v / 10.0:.1f}")
        )
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(self.cfg_scale_slider)
        cfg_row.addWidget(self.cfg_scale_label)
        cfg_form.addRow("CFG Scale:", cfg_row)
        right_col.addWidget(cfg_group)

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

        self.btn_compare = QPushButton("Compare")
        self.btn_compare.setToolTip("Compare A vs B parameter sets side-by-side")
        self.btn_compare.clicked.connect(self._on_compare)
        action_row.addWidget(self.btn_compare)

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

        voice_state = [
            self._slider_value(s) for s in self.voice_state_sliders
        ]

        config = {
            "text": text,
            "language": self._get_language(),
            "uclm_checkpoint": uclm_ckpt,
            "codec_checkpoint": codec_ckpt,
            "speaker_file": self.speaker_edit.text().strip() or None,
            "reference_audio": self.ref_audio_edit.text().strip() or None,
            "speed": self.speed_spin.value(),
            "emotion": EMOTION_OPTIONS[self.emotion_combo.currentIndex()],
            "breathiness": self._slider_value(self.breathiness_slider),
            "tension": self._slider_value(self.tension_slider),
            "arousal": self._slider_value(self.arousal_slider),
            "valence": self._slider_value(self.valence_slider),
            "roughness": self._slider_value(self.roughness_slider),
            "voicing": self._slider_value(self.voicing_slider),
            "energy": self._slider_value(self.energy_slider),
            "voice_state": voice_state,
            "cfg_scale": self.cfg_scale_slider.value() / 10.0,
            "dialogue_context": self.dialogue_context_edit.toPlainText().strip() or None,
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

    def _on_browse_ref_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "",
            "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)",
        )
        if path:
            self.ref_audio_edit.setText(path)

    def _on_extract_speaker(self) -> None:
        ref_path = self.ref_audio_edit.text().strip()
        if not ref_path:
            self.append_log("ERROR: No reference audio selected for speaker extraction.")
            return
        self.append_log(f"Extracting speaker embedding from: {ref_path}")
        # Delegate to worker / engine when available
        self.append_log("Speaker extraction queued (requires running engine).")

    def _on_compare(self) -> None:
        """Generate with current params as A, then play last audio as B for comparison."""
        if self._last_audio is None:
            self.append_log("No previous audio (B) to compare. Generate at least once first.")
            return
        self.append_log("Generating A with current parameters for A/B comparison...")
        self._compare_b_audio = self._last_audio
        self._compare_b_sr = self._last_sr
        self._on_generate()

    def _play_compare(self) -> None:
        """Play A then B sequentially for side-by-side comparison."""
        try:
            import numpy as np
            import sounddevice as sd

            if self._last_audio is None or self._compare_b_audio is None:
                self.append_log("ERROR: Need both A and B audio for comparison.")
                return
            silence = np.zeros(int(self._last_sr * 0.5), dtype=self._last_audio.dtype)
            combined = np.concatenate([self._last_audio, silence, self._compare_b_audio])
            sd.play(combined, self._last_sr)
            self.append_log("Playing A ... [pause] ... B")
        except Exception as e:
            self.append_log(f"ERROR: Compare playback failed: {e}")

    # ------------------------------------------------------------------
    # Casting Gallery slots
    # ------------------------------------------------------------------

    def _on_gallery_save(self) -> None:
        """Save current speaker prompt to gallery with a user-given name."""
        name, ok = QInputDialog.getText(self, "Save to Gallery", "Profile name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        profile_path = self.speaker_edit.text().strip()
        entry = {"name": name, "profile_path": profile_path}
        self._gallery.append(entry)
        self._refresh_gallery_list()
        self.append_log(f"Saved profile '{name}' to gallery.")

    def _on_gallery_load(self) -> None:
        """Load selected gallery profile into the active speaker state."""
        row = self.gallery_list.currentRow()
        if row < 0 or row >= len(self._gallery):
            self.append_log("No gallery profile selected.")
            return
        entry = self._gallery[row]
        self.speaker_edit.setText(entry.get("profile_path", ""))
        self.append_log(f"Loaded gallery profile: {entry.get('name', '')}")

    def _on_gallery_export(self) -> None:
        """Export selected gallery profile as a .tmrvc_speaker file."""
        row = self.gallery_list.currentRow()
        if row < 0 or row >= len(self._gallery):
            self.append_log("No gallery profile selected for export.")
            return
        entry = self._gallery[row]
        default_name = f"{entry.get('name', 'profile')}.tmrvc_speaker"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Profile", default_name,
            "Speaker Profile (*.tmrvc_speaker);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            self.append_log(f"Exported profile to {path}")
        except Exception as e:
            self.append_log(f"ERROR: Export failed: {e}")

    def _on_gallery_import(self) -> None:
        """Import a .tmrvc_speaker file into the gallery."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Profile", "",
            "Speaker Profile (*.tmrvc_speaker);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            if not isinstance(entry, dict):
                self.append_log("ERROR: Invalid profile format.")
                return
            if "name" not in entry:
                entry["name"] = Path(path).stem
            if "profile_path" not in entry:
                entry["profile_path"] = path
            self._gallery.append(entry)
            self._refresh_gallery_list()
            self.append_log(f"Imported profile: {entry.get('name', '')}")
        except Exception as e:
            self.append_log(f"ERROR: Import failed: {e}")

    def _on_gallery_delete(self) -> None:
        """Remove selected profile from the gallery."""
        row = self.gallery_list.currentRow()
        if row < 0 or row >= len(self._gallery):
            self.append_log("No gallery profile selected for deletion.")
            return
        removed = self._gallery.pop(row)
        self._refresh_gallery_list()
        self.append_log(f"Deleted gallery profile: {removed.get('name', '')}")

    def _refresh_gallery_list(self) -> None:
        """Refresh the gallery QListWidget from internal storage."""
        self.gallery_list.clear()
        for entry in self._gallery:
            name = entry.get("name", "Unnamed")
            profile_path = entry.get("profile_path", "")
            description = f"{name} - {profile_path}" if profile_path else name
            self.gallery_list.addItem(QListWidgetItem(description))

    # ------------------------------------------------------------------
    # Session persistence
    # ------------------------------------------------------------------

    def _collect_session_data(self) -> dict:
        """Collect all current UI state into a serializable dict."""
        voice_state_values = [s.value() for s in self.voice_state_sliders]
        return {
            "text": self.text_edit.toPlainText(),
            "language_index": self.language_combo.currentIndex(),
            "speaker_file": self.speaker_edit.text(),
            "uclm_checkpoint": self.uclm_ckpt_edit.text(),
            "codec_checkpoint": self.codec_ckpt_edit.text(),
            "ref_audio": self.ref_audio_edit.text(),
            "dialogue_context": self.dialogue_context_edit.toPlainText(),
            "emotion_index": self.emotion_combo.currentIndex(),
            "speed": self.speed_spin.value(),
            "breathiness": self.breathiness_slider.value(),
            "tension": self.tension_slider.value(),
            "arousal": self.arousal_slider.value(),
            "valence": self.valence_slider.value(),
            "roughness": self.roughness_slider.value(),
            "voicing": self.voicing_slider.value(),
            "energy": self.energy_slider.value(),
            "voice_state": voice_state_values,
            "cfg_scale": self.cfg_scale_slider.value(),
            "gallery": self._gallery,
        }

    def _restore_session_data(self, data: dict) -> None:
        """Restore UI state from a session dict."""
        self.text_edit.setPlainText(data.get("text", ""))
        idx = data.get("language_index", 0)
        if 0 <= idx < self.language_combo.count():
            self.language_combo.setCurrentIndex(idx)
        self.speaker_edit.setText(data.get("speaker_file", ""))
        self.uclm_ckpt_edit.setText(data.get("uclm_checkpoint", ""))
        self.codec_ckpt_edit.setText(data.get("codec_checkpoint", ""))
        self.ref_audio_edit.setText(data.get("ref_audio", ""))
        self.dialogue_context_edit.setPlainText(data.get("dialogue_context", ""))

        emo_idx = data.get("emotion_index", 0)
        if 0 <= emo_idx < self.emotion_combo.count():
            self.emotion_combo.setCurrentIndex(emo_idx)

        self.speed_spin.setValue(data.get("speed", 1.0))
        self.breathiness_slider.setValue(data.get("breathiness", 0))
        self.tension_slider.setValue(data.get("tension", 0))
        self.arousal_slider.setValue(data.get("arousal", 0))
        self.valence_slider.setValue(data.get("valence", 0))
        self.roughness_slider.setValue(data.get("roughness", 0))
        self.voicing_slider.setValue(data.get("voicing", 100))
        self.energy_slider.setValue(data.get("energy", 0))

        voice_state = data.get("voice_state", [])
        for i, val in enumerate(voice_state):
            if i < len(self.voice_state_sliders):
                self.voice_state_sliders[i].setValue(val)

        self.cfg_scale_slider.setValue(data.get("cfg_scale", 10))

        self._gallery = data.get("gallery", [])
        self._refresh_gallery_list()

    def _on_save_session(self) -> None:
        """Save the current session state as a JSON file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "tts_session.json",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            session = self._collect_session_data()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session, f, indent=2, ensure_ascii=False)
            self.append_log(f"Session saved to {path}")
        except Exception as e:
            self.append_log(f"ERROR: Session save failed: {e}")

    def _on_load_session(self) -> None:
        """Load a session state from a JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._restore_session_data(data)
            self.append_log(f"Session loaded from {path}")
        except Exception as e:
            self.append_log(f"ERROR: Session load failed: {e}")
