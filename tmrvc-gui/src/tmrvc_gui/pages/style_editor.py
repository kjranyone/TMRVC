"""Style Editor page: manage 8-dim physical voice state presets."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from tmrvc_core.dialogue_types import StyleParams


class StyleEditorPage(QWidget):
    """Editor for UCLM v2 8-dimensional physical style presets."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._presets: dict[str, StyleParams] = {
            "neutral": StyleParams(emotion="neutral"),
            "whisper_soft": StyleParams(emotion="whisper", breathiness=0.7, voicing=0.3, energy=0.4),
            "angry_intense": StyleParams(emotion="angry", tension=0.8, arousal=0.7, energy=0.8),
        }
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)

        # --- Left: Preset List ---
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("Style Presets"))
        
        self.preset_list = QListWidget()
        self.preset_list.itemSelectionChanged.connect(self._on_selection_changed)
        left_panel.addWidget(self.preset_list)

        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self._on_new_preset)
        btn_row.addWidget(self.btn_new)
        
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete_preset)
        btn_row.addWidget(self.btn_delete)
        left_panel.addLayout(btn_row)
        
        layout.addLayout(left_panel, stretch=1)

        # --- Right: Parameter Editor ---
        right_panel = QVBoxLayout()
        edit_group = QGroupBox("Edit Physical Parameters")
        form = QFormLayout(edit_group)

        self.name_edit = QLineEdit()
        form.addRow("Preset Name:", self.name_edit)

        self.emotion_edit = QLineEdit()
        form.addRow("Emotion Label:", self.emotion_edit)

        # Sliders for 8-dim parameters
        self.sliders: dict[str, QSlider] = {}
        params = [
            ("breathiness", 0, 100),
            ("tension", 0, 100),
            ("arousal", 0, 100),
            ("valence", -100, 100),
            ("roughness", 0, 100),
            ("voicing", 0, 100),
            ("energy", 0, 100),
            ("speech_rate", 50, 200),
        ]

        for name, min_v, max_val in params:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_v, max_val)
            slider.valueChanged.connect(self._on_slider_changed)
            self.sliders[name] = slider
            form.addRow(f"{name.replace('_', ' ').title()}:", slider)

        right_panel.addWidget(edit_group)

        # Persistence
        io_row = QHBoxLayout()
        btn_save_file = QPushButton("Export to JSON...")
        btn_save_file.clicked.connect(self._on_export_json)
        io_row.addWidget(btn_save_file)
        
        btn_load_file = QPushButton("Import from JSON...")
        btn_load_file.clicked.connect(self._on_import_json)
        io_row.addWidget(btn_load_file)
        right_panel.addLayout(io_row)
        
        layout.addLayout(right_panel, stretch=2)

    def _refresh_list(self) -> None:
        self.preset_list.clear()
        for name in sorted(self._presets.keys()):
            self.preset_list.addItem(name)

    def _on_selection_changed(self) -> None:
        items = self.preset_list.selectedItems()
        if not items: return
        name = items[0].text()
        style = self._presets[name]
        
        self.name_edit.setText(name)
        self.emotion_edit.setText(style.emotion)
        
        # Sync sliders
        self.sliders["breathiness"].setValue(int(style.breathiness * 100))
        self.sliders["tension"].setValue(int(style.tension * 100))
        self.sliders["arousal"].setValue(int(style.arousal * 100))
        self.sliders["valence"].setValue(int(style.valence * 100))
        self.sliders["roughness"].setValue(int(style.roughness * 100))
        self.sliders["voicing"].setValue(int(style.voicing * 100))
        self.sliders["energy"].setValue(int(style.energy * 100))
        self.sliders["speech_rate"].setValue(int(style.speech_rate * 100))

    def _on_slider_changed(self) -> None:
        # Update current selected preset in memory
        items = self.preset_list.selectedItems()
        if not items: return
        name = items[0].text()
        
        new_style = StyleParams(
            emotion=self.emotion_edit.text(),
            breathiness=self.sliders["breathiness"].value() / 100.0,
            tension=self.sliders["tension"].value() / 100.0,
            arousal=self.sliders["arousal"].value() / 100.0,
            valence=self.sliders["valence"].value() / 100.0,
            roughness=self.sliders["roughness"].value() / 100.0,
            voicing=self.sliders["voicing"].value() / 100.0,
            energy=self.sliders["energy"].value() / 100.0,
            speech_rate=self.sliders["speech_rate"].value() / 100.0,
        )
        self._presets[name] = new_style

    def _on_new_preset(self) -> None:
        name = f"preset_{len(self._presets)}"
        self._presets[name] = StyleParams()
        self._refresh_list()

    def _on_delete_preset(self) -> None:
        items = self.preset_list.selectedItems()
        if not items: return
        del self._presets[items[0].text()]
        self._refresh_list()

    def _on_export_json(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Presets", "presets.json", "JSON (*.json)")
        if not path: return
        data = {name: vars(style) for name, style in self._presets.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _on_import_json(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Presets", "", "JSON (*.json)")
        if not path: return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for name, d in data.items():
            self._presets[name] = StyleParams.from_dict(d)
        self._refresh_list()
