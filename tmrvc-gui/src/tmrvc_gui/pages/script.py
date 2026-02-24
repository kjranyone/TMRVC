"""Script page: YAML script editor with batch TTS generation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ScriptPage(QWidget):
    """Script editor and batch TTS generation page.

    Loads YAML scripts, displays dialogue entries in a table,
    and generates audio for all entries in batch.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = None
        self._script = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- File bar ---
        file_row = QHBoxLayout()

        file_row.addWidget(QLabel("Script:"))
        self.script_path_edit = QLineEdit()
        self.script_path_edit.setPlaceholderText("path/to/script.yaml")
        file_row.addWidget(self.script_path_edit, stretch=1)

        btn_open = QPushButton("Open...")
        btn_open.clicked.connect(self._on_open)
        file_row.addWidget(btn_open)

        btn_save = QPushButton("Save")
        btn_save.clicked.connect(self._on_save)
        file_row.addWidget(btn_save)

        btn_save_as = QPushButton("Save As...")
        btn_save_as.clicked.connect(self._on_save_as)
        file_row.addWidget(btn_save_as)

        layout.addLayout(file_row)

        # --- Splitter: editor (left) + table (right) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: YAML editor
        editor_group = QGroupBox("YAML Editor")
        editor_layout = QVBoxLayout(editor_group)
        self.yaml_edit = QTextEdit()
        self.yaml_edit.setPlaceholderText(
            "title: \"Scene 1\"\n"
            "characters:\n"
            "  narrator:\n"
            "    name: \"Narrator\"\n"
            "dialogue:\n"
            "  - speaker: narrator\n"
            "    text: \"Hello world\"\n"
        )
        self.yaml_edit.setAcceptRichText(False)
        editor_layout.addWidget(self.yaml_edit)

        btn_parse = QPushButton("Parse YAML")
        btn_parse.clicked.connect(self._on_parse)
        editor_layout.addWidget(btn_parse)

        splitter.addWidget(editor_group)

        # Right: dialogue table
        table_group = QGroupBox("Dialogue")
        table_layout = QVBoxLayout(table_group)

        self.info_label = QLabel("No script loaded.")
        table_layout.addWidget(self.info_label)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["#", "Speaker", "Text", "Hint/Emotion"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_layout.addWidget(self.table)

        splitter.addWidget(table_group)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter, stretch=1)

        # --- Model paths ---
        model_row = QHBoxLayout()

        model_row.addWidget(QLabel("TTS ckpt:"))
        self.tts_ckpt_edit = QLineEdit()
        self.tts_ckpt_edit.setPlaceholderText("checkpoints/tts/tts_step200000.pt")
        model_row.addWidget(self.tts_ckpt_edit, stretch=1)

        model_row.addWidget(QLabel("VC ckpt:"))
        self.vc_ckpt_edit = QLineEdit()
        self.vc_ckpt_edit.setPlaceholderText("checkpoints/distill/best.pt")
        model_row.addWidget(self.vc_ckpt_edit, stretch=1)

        layout.addLayout(model_row)

        # --- Output dir + actions ---
        action_row = QHBoxLayout()

        action_row.addWidget(QLabel("Output:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("output/")
        action_row.addWidget(self.output_dir_edit, stretch=1)

        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._on_browse_output)
        action_row.addWidget(btn_browse_out)

        self.btn_generate = QPushButton("Generate All")
        self.btn_generate.setMinimumHeight(36)
        self.btn_generate.clicked.connect(self._on_generate)
        action_row.addWidget(self.btn_generate)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        action_row.addWidget(self.btn_cancel)

        layout.addLayout(action_row)

        # --- Log ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setMaximumHeight(120)
        log_layout.addWidget(self.log_edit)
        layout.addWidget(log_group)

    def append_log(self, text: str) -> None:
        self.log_edit.append(text)

    def _populate_table(self) -> None:
        """Fill the table from the current script."""
        if self._script is None:
            self.table.setRowCount(0)
            self.info_label.setText("No script loaded.")
            return

        self.info_label.setText(
            f"Title: {self._script.title or '(untitled)'} | "
            f"Characters: {len(self._script.characters)} | "
            f"Entries: {len(self._script.entries)}"
        )

        self.table.setRowCount(len(self._script.entries))
        for i, entry in enumerate(self._script.entries):
            self.table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.table.setItem(i, 1, QTableWidgetItem(entry.speaker))
            self.table.setItem(i, 2, QTableWidgetItem(entry.text))
            hint = entry.hint or ""
            if entry.style_override and entry.style_override.emotion != "neutral":
                hint = f"[{entry.style_override.emotion}] {hint}"
            self.table.setItem(i, 3, QTableWidgetItem(hint))

    # --- Slots ---

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.script_path_edit.setText(path)
            try:
                text = Path(path).read_text(encoding="utf-8")
                self.yaml_edit.setPlainText(text)
                self._on_parse()
            except Exception as e:
                self.append_log(f"ERROR: {e}")

    def _on_save(self) -> None:
        path = self.script_path_edit.text().strip()
        if not path:
            self._on_save_as()
            return
        try:
            Path(path).write_text(self.yaml_edit.toPlainText(), encoding="utf-8")
            self.append_log(f"Saved: {path}")
        except Exception as e:
            self.append_log(f"ERROR saving: {e}")

    def _on_save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "script.yaml",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if path:
            self.script_path_edit.setText(path)
            self._on_save()

    def _on_parse(self) -> None:
        """Parse the YAML text and update the table."""
        text = self.yaml_edit.toPlainText().strip()
        if not text:
            self.append_log("No YAML content to parse.")
            return

        try:
            from tmrvc_data.script_parser import load_script_from_string
            self._script = load_script_from_string(text)
            self._populate_table()
            self.append_log(
                f"Parsed: {len(self._script.entries)} entries, "
                f"{len(self._script.characters)} characters"
            )
        except Exception as e:
            self.append_log(f"Parse error: {e}")

    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.output_dir_edit.setText(path)

    def _on_generate(self) -> None:
        if self._script is None or not self._script.entries:
            self.append_log("ERROR: No script loaded or no entries.")
            return

        output_dir = self.output_dir_edit.text().strip() or "output"

        self.append_log(f"Generating {len(self._script.entries)} entries...")
        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)

        from tmrvc_gui.workers.script_worker import ScriptWorker

        config = {
            "script_yaml": self.yaml_edit.toPlainText(),
            "output_dir": output_dir,
            "tts_checkpoint": self.tts_ckpt_edit.text().strip() or None,
            "vc_checkpoint": self.vc_ckpt_edit.text().strip() or None,
        }

        self._worker = ScriptWorker(config)
        self._worker.log_message.connect(self.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(lambda msg: self.append_log(f"ERROR: {msg}"))
        self.btn_cancel.clicked.connect(self._worker.cancel)
        self._worker.start()

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.append_log(message)
        self._worker = None
