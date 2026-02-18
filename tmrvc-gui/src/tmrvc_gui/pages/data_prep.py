"""Data preparation page for corpus management and preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tmrvc_core.constants import HOP_LENGTH, N_MELS, SAMPLE_RATE


class DataPrepPage(QWidget):
    """Corpus management and preprocessing pipeline configuration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Corpus management table ---
        corpus_group = QGroupBox("Corpus Management")
        corpus_layout = QVBoxLayout(corpus_group)

        self.corpus_table = QTableWidget(4, 5)
        self.corpus_table.setHorizontalHeaderLabels(
            ["Corpus", "Speakers", "Hours", "Path", "Status"]
        )
        header = self.corpus_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        corpora = [
            ("VCTK", "110", "~44h", "", "Not downloaded"),
            ("JVS", "100", "~30h", "", "Not downloaded"),
            ("LibriTTS-R", "2456", "~585h", "", "Not downloaded"),
            ("Emilia", "~50k", "~100kh", "", "Not downloaded"),
        ]
        for row, (name, speakers, hours, path, status) in enumerate(corpora):
            self.corpus_table.setItem(row, 0, QTableWidgetItem(name))
            self.corpus_table.setItem(row, 1, QTableWidgetItem(speakers))
            self.corpus_table.setItem(row, 2, QTableWidgetItem(hours))
            self.corpus_table.setItem(row, 3, QTableWidgetItem(path))
            self.corpus_table.setItem(row, 4, QTableWidgetItem(status))

        corpus_layout.addWidget(self.corpus_table)
        layout.addWidget(corpus_group)

        # --- Preprocessing pipeline config ---
        pipeline_group = QGroupBox("Preprocessing Pipeline")
        pipeline_layout = QVBoxLayout(pipeline_group)

        checks_layout = QHBoxLayout()

        self.cb_resample = QCheckBox("Resample to 24kHz")
        self.cb_resample.setChecked(True)
        checks_layout.addWidget(self.cb_resample)

        self.cb_normalize = QCheckBox("Normalize")
        self.cb_normalize.setChecked(True)
        checks_layout.addWidget(self.cb_normalize)

        self.cb_vad_trim = QCheckBox("VAD Trim")
        self.cb_vad_trim.setChecked(True)
        checks_layout.addWidget(self.cb_vad_trim)

        self.cb_segment = QCheckBox("Segment")
        self.cb_segment.setChecked(True)
        checks_layout.addWidget(self.cb_segment)

        self.cb_features = QCheckBox("Extract Features")
        self.cb_features.setChecked(True)
        checks_layout.addWidget(self.cb_features)

        pipeline_layout.addLayout(checks_layout)

        # Cache directory
        cache_row = QHBoxLayout()
        cache_row.addWidget(QLabel("Cache dir:"))
        self.cache_dir_edit = QLineEdit()
        self.cache_dir_edit.setPlaceholderText("data/cache")
        self.cache_dir_edit.setText("data/cache")
        cache_row.addWidget(self.cache_dir_edit)
        self.btn_browse_cache = QPushButton("Browse...")
        self.btn_browse_cache.clicked.connect(self._on_browse_cache)
        cache_row.addWidget(self.btn_browse_cache)
        pipeline_layout.addLayout(cache_row)

        # Worker count
        worker_layout = QHBoxLayout()
        worker_layout.addWidget(QLabel("Worker count:"))
        self.worker_spinbox = QSpinBox()
        self.worker_spinbox.setRange(1, 64)
        self.worker_spinbox.setValue(4)
        worker_layout.addWidget(self.worker_spinbox)
        worker_layout.addStretch()

        self.btn_run = QPushButton("Run Preprocessing")
        self.btn_run.clicked.connect(self._on_run)
        worker_layout.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        worker_layout.addWidget(self.btn_cancel)

        pipeline_layout.addLayout(worker_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        pipeline_layout.addWidget(self.progress_bar)

        layout.addWidget(pipeline_group)

        # --- Log viewer ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)

        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMinimumHeight(120)
        self.log_viewer.setPlaceholderText("Preprocessing logs will appear here...")
        log_layout.addWidget(self.log_viewer)

        layout.addWidget(log_group)

    # ------------------------------------------------------------------
    # Worker state
    # ------------------------------------------------------------------

    _worker = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_cache(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(self, "Select Cache Directory")
        if dir_path:
            self.cache_dir_edit.setText(dir_path)

    def _on_run(self) -> None:
        from tmrvc_gui.workers.data_worker import DataWorker

        # Gather corpus info from table
        corpus_paths: list[str] = []
        dataset_names: list[str] = []
        dataset_name_map = {"VCTK": "vctk", "JVS": "jvs", "LibriTTS-R": "libritts_r", "Emilia": "emilia"}

        for row in range(self.corpus_table.rowCount()):
            path_item = self.corpus_table.item(row, 3)
            name_item = self.corpus_table.item(row, 0)
            if path_item and path_item.text().strip():
                corpus_paths.append(path_item.text().strip())
                corpus_name = name_item.text() if name_item else "vctk"
                dataset_names.append(dataset_name_map.get(corpus_name, corpus_name.lower()))

        if not corpus_paths:
            self.append_log("No corpus paths set. Fill the Path column in the table first.")
            return

        # Gather steps
        steps: list[str] = []
        if self.cb_resample.isChecked():
            steps.append("resample")
        if self.cb_normalize.isChecked():
            steps.append("normalize")
        if self.cb_vad_trim.isChecked():
            steps.append("vad_trim")
        if self.cb_segment.isChecked():
            steps.append("segment")
        if self.cb_features.isChecked():
            steps.append("features")

        config = {
            "corpus_paths": corpus_paths,
            "dataset_names": dataset_names,
            "cache_dir": self.cache_dir_edit.text() or "data/cache",
            "steps": steps,
            "n_workers": self.worker_spinbox.value(),
            "device": "cpu",
        }

        self._worker = DataWorker(config)
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(self.append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(lambda msg: self.append_log(f"ERROR: {msg}"))

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setValue(0)
        self._worker.start()

    def _on_cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.progress_bar.setValue(int(100 * current / total))

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.append_log(f"Finished: {message}")
        if success:
            self.progress_bar.setValue(100)
        self._worker = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def append_log(self, text: str) -> None:
        """Append a line to the log viewer."""
        self.log_viewer.append(text)

    def set_progress(self, value: int) -> None:
        """Set progress bar value (0-100)."""
        self.progress_bar.setValue(value)
