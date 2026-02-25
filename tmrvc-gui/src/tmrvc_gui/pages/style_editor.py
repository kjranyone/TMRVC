"""StylePage: .tmrvc_style file editor with audio preview and preset management."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# Emotion categories (matches EMOTION_CATEGORIES in dialogue_types.py)
_EMOTIONS = [
    "neutral", "happy", "sad", "angry", "fearful", "surprised",
    "disgusted", "bored", "excited", "tender", "sarcastic", "whisper",
]


class StyleEditorPage(QWidget):
    """Editor for .tmrvc_style prosody files and emotion style vectors."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_path: Path | None = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)

        # --- File operations ---
        file_row = QHBoxLayout()
        self._path_label = QLabel("No file loaded")
        file_row.addWidget(self._path_label, stretch=1)
        btn_new = QPushButton("New")
        btn_new.clicked.connect(self._on_new)
        file_row.addWidget(btn_new)
        btn_open = QPushButton("Open...")
        btn_open.clicked.connect(self._on_open)
        file_row.addWidget(btn_open)
        self._btn_save = QPushButton("Save")
        self._btn_save.clicked.connect(self._on_save)
        file_row.addWidget(self._btn_save)
        btn_save_as = QPushButton("Save As...")
        btn_save_as.clicked.connect(self._on_save_as)
        file_row.addWidget(btn_save_as)
        layout.addLayout(file_row)

        # --- Prosody parameters (core .tmrvc_style fields) ---
        prosody_group = QGroupBox("Prosody Parameters")
        prosody_layout = QVBoxLayout(prosody_group)

        self._f0_spin = self._add_param_row(
            prosody_layout, "Target log(F0):", 0.0, 10.0, 5.2, 0.01,
        )
        self._articulation_spin = self._add_param_row(
            prosody_layout, "Articulation:", 0.0, 5.0, 1.0, 0.01,
        )
        self._voiced_ratio_spin = self._add_param_row(
            prosody_layout, "Voiced ratio:", 0.0, 1.0, 0.7, 0.01,
        )
        layout.addWidget(prosody_group)

        # --- Emotion style (32-dim style vector preview) ---
        emotion_group = QGroupBox("Emotion Style (Preview)")
        emotion_layout = QVBoxLayout(emotion_group)

        emo_row = QHBoxLayout()
        emo_row.addWidget(QLabel("Primary emotion:"))
        self._emotion_combo = QComboBox()
        self._emotion_combo.addItems(_EMOTIONS)
        emo_row.addWidget(self._emotion_combo)
        emotion_layout.addLayout(emo_row)

        # VAD sliders
        self._valence_slider = self._add_slider_row(emotion_layout, "Valence:", -100, 100, 0)
        self._arousal_slider = self._add_slider_row(emotion_layout, "Arousal:", -100, 100, 0)
        self._dominance_slider = self._add_slider_row(emotion_layout, "Dominance:", -100, 100, 0)

        # Prosody control sliders
        self._speech_rate_slider = self._add_slider_row(
            emotion_layout, "Speech rate:", -100, 100, 0,
        )
        self._energy_slider = self._add_slider_row(emotion_layout, "Energy:", -100, 100, 0)
        self._pitch_range_slider = self._add_slider_row(
            emotion_layout, "Pitch range:", -100, 100, 0,
        )
        layout.addWidget(emotion_group)

        # --- Metadata ---
        meta_group = QGroupBox("Metadata")
        meta_layout = QVBoxLayout(meta_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Display name:"))
        self._name_edit = QLineEdit()
        name_row.addWidget(self._name_edit)
        meta_layout.addLayout(name_row)

        desc_row = QHBoxLayout()
        desc_row.addWidget(QLabel("Description:"))
        self._desc_edit = QLineEdit()
        desc_row.addWidget(self._desc_edit)
        meta_layout.addLayout(desc_row)

        layout.addWidget(meta_group)

        # --- Presets ---
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        self._preset_list = QListWidget()
        self._preset_list.addItems([
            "Neutral (default)",
            "ASMR Soft",
            "ASMR Intimate",
            "Excited Shouting",
            "Calm Whisper",
            "Sad Reading",
        ])
        self._preset_list.currentRowChanged.connect(self._on_preset_selected)
        preset_layout.addWidget(self._preset_list)

        preset_btn_col = QVBoxLayout()
        btn_apply = QPushButton("Apply Preset")
        btn_apply.clicked.connect(self._on_apply_preset)
        preset_btn_col.addWidget(btn_apply)
        preset_btn_col.addStretch()
        preset_layout.addLayout(preset_btn_col)
        layout.addWidget(preset_group)

        # --- Log ---
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(100)
        layout.addWidget(self._log)

    def _add_param_row(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        min_val: float,
        max_val: float,
        default: float,
        step: float,
    ) -> QDoubleSpinBox:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(default)
        spin.setDecimals(2)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return spin

    def _add_slider_row(
        self,
        parent_layout: QVBoxLayout,
        label: str,
        min_val: int,
        max_val: int,
        default: int,
    ) -> QSlider:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        val_label = QLabel(f"{default / 100:.2f}")
        slider.valueChanged.connect(lambda v, lbl=val_label: lbl.setText(f"{v / 100:.2f}"))
        row.addWidget(slider, stretch=1)
        row.addWidget(val_label)
        parent_layout.addLayout(row)
        return slider

    def _log_msg(self, msg: str) -> None:
        self._log.append(msg)

    # --- File operations ---

    def _on_new(self) -> None:
        self._current_path = None
        self._f0_spin.setValue(5.2)
        self._articulation_spin.setValue(1.0)
        self._voiced_ratio_spin.setValue(0.7)
        self._name_edit.clear()
        self._desc_edit.clear()
        self._emotion_combo.setCurrentIndex(0)
        for slider in (
            self._valence_slider, self._arousal_slider, self._dominance_slider,
            self._speech_rate_slider, self._energy_slider, self._pitch_range_slider,
        ):
            slider.setValue(0)
        self._path_label.setText("New style (unsaved)")
        self._log_msg("New style created")

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Style File", "", "TMRVC Style (*.tmrvc_style);;All Files (*)",
        )
        if not path:
            return
        try:
            self._load_file(Path(path))
            self._log_msg(f"Loaded: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def _on_save(self) -> None:
        if self._current_path is None:
            self._on_save_as()
            return
        try:
            self._save_file(self._current_path)
            self._log_msg(f"Saved: {self._current_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Style File", "", "TMRVC Style (*.tmrvc_style);;All Files (*)",
        )
        if not path:
            return
        try:
            self._save_file(Path(path))
            self._current_path = Path(path)
            self._path_label.setText(str(self._current_path))
            self._log_msg(f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def _load_file(self, path: Path) -> None:
        import json
        import struct

        data = path.read_bytes()
        if len(data) < 56:
            raise ValueError("File too small")
        if data[:4] != b"TMST":
            raise ValueError("Invalid magic")

        meta_size = struct.unpack("<I", data[8:12])[0]
        self._f0_spin.setValue(struct.unpack("<f", data[12:16])[0])
        self._articulation_spin.setValue(struct.unpack("<f", data[16:20])[0])
        self._voiced_ratio_spin.setValue(struct.unpack("<f", data[20:24])[0])

        if meta_size > 0:
            meta = json.loads(data[24:24 + meta_size])
            self._name_edit.setText(meta.get("display_name", ""))
            self._desc_edit.setText(meta.get("description", ""))

        self._current_path = path
        self._path_label.setText(str(path))

    def _save_file(self, path: Path) -> None:
        import hashlib
        import json
        import struct

        meta = {
            "display_name": self._name_edit.text(),
            "created_at": "",
            "description": self._desc_edit.text(),
            "source_audio_files": [],
            "source_sample_count": 0,
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False).encode("utf-8")

        buf = bytearray()
        buf += b"TMST"
        buf += struct.pack("<I", 1)  # version
        buf += struct.pack("<I", len(meta_bytes))
        buf += struct.pack("<f", self._f0_spin.value())
        buf += struct.pack("<f", self._articulation_spin.value())
        buf += struct.pack("<f", self._voiced_ratio_spin.value())
        buf += meta_bytes

        checksum = hashlib.sha256(bytes(buf)).digest()
        buf += checksum

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(buf))

    # --- Presets ---

    _PRESETS = {
        0: {"f0": 5.2, "art": 1.0, "voiced": 0.7, "val": 0, "aro": 0, "dom": 0, "spd": 0, "eng": 0, "pit": 0, "emo": 0},
        1: {"f0": 5.0, "art": 0.6, "voiced": 0.5, "val": 20, "aro": -55, "dom": -30, "spd": -10, "eng": -55, "pit": -20, "emo": 11},  # ASMR Soft → whisper
        2: {"f0": 4.8, "art": 0.5, "voiced": 0.4, "val": 30, "aro": -65, "dom": -45, "spd": -18, "eng": -65, "pit": -30, "emo": 11},  # ASMR Intimate
        3: {"f0": 5.8, "art": 1.8, "voiced": 0.85, "val": 50, "aro": 80, "dom": 60, "spd": 20, "eng": 80, "pit": 50, "emo": 8},  # Excited
        4: {"f0": 4.5, "art": 0.4, "voiced": 0.3, "val": 0, "aro": -70, "dom": -40, "spd": -20, "eng": -70, "pit": -40, "emo": 11},  # Calm Whisper
        5: {"f0": 4.9, "art": 0.7, "voiced": 0.65, "val": -60, "aro": -30, "dom": -20, "spd": -15, "eng": -30, "pit": -20, "emo": 2},  # Sad
    }

    def _on_preset_selected(self, row: int) -> None:
        pass

    def _on_apply_preset(self) -> None:
        row = self._preset_list.currentRow()
        if row < 0 or row not in self._PRESETS:
            return
        p = self._PRESETS[row]
        self._f0_spin.setValue(p["f0"])
        self._articulation_spin.setValue(p["art"])
        self._voiced_ratio_spin.setValue(p["voiced"])
        self._valence_slider.setValue(p["val"])
        self._arousal_slider.setValue(p["aro"])
        self._dominance_slider.setValue(p["dom"])
        self._speech_rate_slider.setValue(p["spd"])
        self._energy_slider.setValue(p["eng"])
        self._pitch_range_slider.setValue(p["pit"])
        self._emotion_combo.setCurrentIndex(p["emo"])
        self._log_msg(f"Applied preset: {self._preset_list.currentItem().text()}")
