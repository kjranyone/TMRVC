"""Curation Auditor page: review and fix dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QComboBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class CurationPage(QWidget):
    """Curation Auditor: browse, review, and fix dataset manifests."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manifest_path: str | None = None
        self._entries: list[dict] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Load / Save controls ---
        io_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Manifest")
        self.btn_load.clicked.connect(self._on_load_manifest)
        io_row.addWidget(self.btn_load)

        self.btn_save = QPushButton("Save Manifest")
        self.btn_save.clicked.connect(self._on_save_manifest)
        io_row.addWidget(self.btn_save)

        self.manifest_label = QLabel("No manifest loaded")
        io_row.addWidget(self.manifest_label)
        io_row.addStretch()
        layout.addLayout(io_row)

        # --- Filter controls ---
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Min confidence:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.0)
        filter_layout.addWidget(self.confidence_spin)

        filter_layout.addWidget(QLabel("Status:"))
        self.status_filter_combo = QComboBox()
        self.status_filter_combo.addItems(["All", "pending", "promoted", "rejected"])
        filter_layout.addWidget(self.status_filter_combo)

        self.btn_apply_filter = QPushButton("Apply Filter")
        self.btn_apply_filter.clicked.connect(self._on_apply_filter)
        filter_layout.addWidget(self.btn_apply_filter)

        filter_layout.addStretch()
        layout.addWidget(filter_group)

        # --- Manifest table ---
        table_group = QGroupBox("Manifest Browser")
        table_layout = QVBoxLayout(table_group)

        self.manifest_table = QTableWidget(0, 4)
        self.manifest_table.setHorizontalHeaderLabels(
            ["Utterance ID", "Transcript", "Confidence", "Status"]
        )
        header = self.manifest_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.manifest_table.currentCellChanged.connect(self._on_row_selected)
        table_layout.addWidget(self.manifest_table)

        # Review buttons
        review_row = QHBoxLayout()
        self.btn_promote = QPushButton("Promote")
        self.btn_promote.clicked.connect(self._on_promote)
        review_row.addWidget(self.btn_promote)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.clicked.connect(self._on_reject)
        review_row.addWidget(self.btn_reject)

        review_row.addStretch()
        table_layout.addLayout(review_row)

        layout.addWidget(table_group)

        # --- Inline text editor ---
        edit_group = QGroupBox("Transcript Editor")
        edit_layout = QVBoxLayout(edit_group)

        self.transcript_edit = QTextEdit()
        self.transcript_edit.setMaximumHeight(80)
        self.transcript_edit.setPlaceholderText(
            "Select a row above to edit its transcript..."
        )
        edit_layout.addWidget(self.transcript_edit)

        self.btn_apply_edit = QPushButton("Apply Edit")
        self.btn_apply_edit.clicked.connect(self._on_apply_edit)
        edit_layout.addWidget(self.btn_apply_edit)

        layout.addWidget(edit_group)

        # --- Segment Boundary Correction ---
        boundary_group = QGroupBox("Segment Boundary Correction")
        boundary_layout = QHBoxLayout(boundary_group)

        boundary_layout.addWidget(QLabel("Start (sec):"))
        self.start_sec_spin = QDoubleSpinBox()
        self.start_sec_spin.setRange(0.0, 99999.0)
        self.start_sec_spin.setSingleStep(0.01)
        self.start_sec_spin.setDecimals(3)
        self.start_sec_spin.setValue(0.0)
        boundary_layout.addWidget(self.start_sec_spin)

        boundary_layout.addWidget(QLabel("End (sec):"))
        self.end_sec_spin = QDoubleSpinBox()
        self.end_sec_spin.setRange(0.0, 99999.0)
        self.end_sec_spin.setSingleStep(0.01)
        self.end_sec_spin.setDecimals(3)
        self.end_sec_spin.setValue(0.0)
        boundary_layout.addWidget(self.end_sec_spin)

        self.btn_apply_bounds = QPushButton("Apply Bounds")
        self.btn_apply_bounds.clicked.connect(self._on_apply_bounds)
        boundary_layout.addWidget(self.btn_apply_bounds)

        boundary_layout.addStretch()
        layout.addWidget(boundary_group)

        # --- Speaker Merge/Split ---
        speaker_group = QGroupBox("Speaker Merge / Split")
        speaker_layout = QHBoxLayout(speaker_group)

        self.btn_merge_speakers = QPushButton("Merge Speakers")
        self.btn_merge_speakers.clicked.connect(self._on_merge_speakers)
        speaker_layout.addWidget(self.btn_merge_speakers)

        self.btn_split_speaker = QPushButton("Split Speaker")
        self.btn_split_speaker.clicked.connect(self._on_split_speaker)
        speaker_layout.addWidget(self.btn_split_speaker)

        speaker_layout.addStretch()
        layout.addWidget(speaker_group)

        # --- Language-Span Correction ---
        lang_group = QGroupBox("Language-Span Correction")
        lang_layout = QHBoxLayout(lang_group)

        lang_layout.addWidget(QLabel("Language:"))
        self.language_combo = QComboBox()
        self.language_combo.addItems(["ja", "en", "zh", "ko", "mixed"])
        lang_layout.addWidget(self.language_combo)

        self.btn_apply_language = QPushButton("Apply Language")
        self.btn_apply_language.clicked.connect(self._on_apply_language)
        lang_layout.addWidget(self.btn_apply_language)

        lang_layout.addStretch()
        layout.addWidget(lang_group)

        # --- Record Owner & Blocking Reason ---
        owner_group = QGroupBox("Record Owner & Blocking Reason")
        owner_layout = QVBoxLayout(owner_group)

        self.owner_label = QLabel("Owner: --")
        owner_layout.addWidget(self.owner_label)

        self.blocking_reason_label = QLabel("Blocking reason: --")
        owner_layout.addWidget(self.blocking_reason_label)

        layout.addWidget(owner_group)

        # --- Approval History ---
        history_group = QGroupBox("Approval History")
        history_layout = QVBoxLayout(history_group)

        self.approval_history_edit = QTextEdit()
        self.approval_history_edit.setReadOnly(True)
        self.approval_history_edit.setMaximumHeight(120)
        self.approval_history_edit.setPlaceholderText(
            "Per-record approval history will appear here..."
        )
        history_layout.addWidget(self.approval_history_edit)

        layout.addWidget(history_group)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_load_manifest(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Manifest", "",
            "JSONL Files (*.jsonl);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        self._manifest_path = path
        self._entries = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._entries.append(json.loads(line))
        except Exception as e:
            self.manifest_label.setText(f"Error: {e}")
            return
        self.manifest_label.setText(
            f"Loaded: {Path(path).name} ({len(self._entries)} entries)"
        )
        self._populate_table(self._entries)

    def _on_save_manifest(self) -> None:
        if not self._entries:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Manifest",
            self._manifest_path or "manifest.jsonl",
            "JSONL Files (*.jsonl);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                for entry in self._entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.manifest_label.setText(
                f"Saved: {Path(path).name} ({len(self._entries)} entries)"
            )
        except Exception as e:
            self.manifest_label.setText(f"Save error: {e}")

    def _populate_table(self, entries: list[dict]) -> None:
        self.manifest_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            uid = entry.get("id", entry.get("utterance_id", ""))
            transcript = entry.get("transcript", entry.get("text", ""))
            confidence = entry.get("confidence", "")
            status = entry.get("status", "pending")
            self.manifest_table.setItem(row, 0, QTableWidgetItem(str(uid)))
            self.manifest_table.setItem(row, 1, QTableWidgetItem(str(transcript)))
            self.manifest_table.setItem(
                row, 2, QTableWidgetItem(f"{confidence}" if confidence != "" else "")
            )
            self.manifest_table.setItem(row, 3, QTableWidgetItem(str(status)))

    def _on_apply_filter(self) -> None:
        min_conf = self.confidence_spin.value()
        status_filter = self.status_filter_combo.currentText()

        filtered = []
        for entry in self._entries:
            conf = entry.get("confidence", 1.0)
            if isinstance(conf, (int, float)) and conf < min_conf:
                continue
            entry_status = entry.get("status", "pending")
            if status_filter != "All" and entry_status != status_filter:
                continue
            filtered.append(entry)
        self._populate_table(filtered)

    def _on_row_selected(self, row: int, _col: int, _prev_row: int, _prev_col: int) -> None:
        if row < 0:
            return
        item = self.manifest_table.item(row, 1)
        if item:
            self.transcript_edit.setText(item.text())

        # Look up the backing entry to populate boundary, owner, history
        uid_item = self.manifest_table.item(row, 0)
        if uid_item is None:
            return
        uid = uid_item.text()
        entry = self._find_entry(uid)
        if entry is None:
            return

        # Populate segment boundary spinboxes
        self.start_sec_spin.setValue(entry.get("start_sec", 0.0))
        self.end_sec_spin.setValue(entry.get("end_sec", 0.0))

        # Populate owner & blocking reason
        owner = entry.get("owner", "--")
        self.owner_label.setText(f"Owner: {owner}")
        blocking = entry.get("blocking_reason", "--")
        self.blocking_reason_label.setText(f"Blocking reason: {blocking}")

        # Populate approval history
        history = entry.get("approval_history", [])
        if history:
            lines = []
            for h in history:
                ts = h.get("timestamp", "")
                action = h.get("action", "")
                by = h.get("by", "")
                lines.append(f"[{ts}] {action} by {by}")
            self.approval_history_edit.setText("\n".join(lines))
        else:
            self.approval_history_edit.setText("No approval history for this record.")

    def _on_promote(self) -> None:
        self._set_current_status("promoted")

    def _on_reject(self) -> None:
        self._set_current_status("rejected")

    def _set_current_status(self, status: str) -> None:
        row = self.manifest_table.currentRow()
        if row < 0:
            return
        uid_item = self.manifest_table.item(row, 0)
        if uid_item is None:
            return
        uid = uid_item.text()
        # Update table display
        self.manifest_table.setItem(row, 3, QTableWidgetItem(status))
        # Update backing data
        for entry in self._entries:
            entry_uid = entry.get("id", entry.get("utterance_id", ""))
            if str(entry_uid) == uid:
                entry["status"] = status
                break

    def _on_apply_edit(self) -> None:
        row = self.manifest_table.currentRow()
        if row < 0:
            return
        new_text = self.transcript_edit.toPlainText().strip()
        uid_item = self.manifest_table.item(row, 0)
        if uid_item is None:
            return
        uid = uid_item.text()
        # Update table display
        self.manifest_table.setItem(row, 1, QTableWidgetItem(new_text))
        # Update backing data
        for entry in self._entries:
            entry_uid = entry.get("id", entry.get("utterance_id", ""))
            if str(entry_uid) == uid:
                if "transcript" in entry:
                    entry["transcript"] = new_text
                else:
                    entry["text"] = new_text
                break

    # ------------------------------------------------------------------
    # Segment boundary correction
    # ------------------------------------------------------------------

    def _on_apply_bounds(self) -> None:
        """Apply the start/end second values to the selected manifest entry."""
        row = self.manifest_table.currentRow()
        if row < 0:
            return
        uid_item = self.manifest_table.item(row, 0)
        if uid_item is None:
            return
        uid = uid_item.text()
        start_sec = self.start_sec_spin.value()
        end_sec = self.end_sec_spin.value()
        entry = self._find_entry(uid)
        if entry is not None:
            entry["start_sec"] = start_sec
            entry["end_sec"] = end_sec

    # ------------------------------------------------------------------
    # Speaker merge / split
    # ------------------------------------------------------------------

    def _on_merge_speakers(self) -> None:
        """Merge selected rows to the same speaker_cluster."""
        selected_rows = self.manifest_table.selectionModel().selectedRows()
        if len(selected_rows) < 2:
            return
        # Use the speaker_cluster of the first selected row as the target
        first_uid_item = self.manifest_table.item(selected_rows[0].row(), 0)
        if first_uid_item is None:
            return
        first_entry = self._find_entry(first_uid_item.text())
        if first_entry is None:
            return
        target_cluster = first_entry.get("speaker_cluster", first_uid_item.text())

        for idx in selected_rows:
            uid_item = self.manifest_table.item(idx.row(), 0)
            if uid_item is None:
                continue
            entry = self._find_entry(uid_item.text())
            if entry is not None:
                entry["speaker_cluster"] = target_cluster

    def _on_split_speaker(self) -> None:
        """Assign the selected row to a new unique speaker_cluster."""
        row = self.manifest_table.currentRow()
        if row < 0:
            return
        uid_item = self.manifest_table.item(row, 0)
        if uid_item is None:
            return
        uid = uid_item.text()
        entry = self._find_entry(uid)
        if entry is not None:
            # Generate a new cluster id by appending _split suffix
            existing = entry.get("speaker_cluster", uid)
            new_cluster = f"{existing}_split_{uid}"
            entry["speaker_cluster"] = new_cluster

    # ------------------------------------------------------------------
    # Language-span correction
    # ------------------------------------------------------------------

    def _on_apply_language(self) -> None:
        """Change the language of the selected manifest row."""
        row = self.manifest_table.currentRow()
        if row < 0:
            return
        uid_item = self.manifest_table.item(row, 0)
        if uid_item is None:
            return
        uid = uid_item.text()
        lang = self.language_combo.currentText()
        entry = self._find_entry(uid)
        if entry is not None:
            entry["language"] = lang

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_entry(self, uid: str) -> dict | None:
        """Find a backing entry by its utterance ID."""
        for entry in self._entries:
            entry_uid = entry.get("id", entry.get("utterance_id", ""))
            if str(entry_uid) == uid:
                return entry
        return None
