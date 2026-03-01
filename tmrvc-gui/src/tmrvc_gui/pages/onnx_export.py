"""ONNX export page for UCLM v2 components."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Unified UCLM v2 ONNX components
ONNX_MODELS: list[str] = [
    "uclm",
    "codec",
    "speaker_encoder",
]


class OnnxExportPage(QWidget):
    """ONNX model export and quantization for UCLM v2."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Export configuration ---
        export_group = QGroupBox("UCLM v2 Export Configuration")
        export_form = QFormLayout(export_group)

        # Checkpoint selector
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setEditable(True)
        self.checkpoint_combo.setPlaceholderText("Select model checkpoint (.pt)")
        export_form.addRow("Checkpoint:", self.checkpoint_combo)

        # Output directory
        out_dir_row = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Output directory for ONNX files")
        out_dir_row.addWidget(self.output_dir_edit)
        self.btn_browse_out = QPushButton("Browse...")
        self.btn_browse_out.clicked.connect(self._on_browse_output)
        out_dir_row.addWidget(self.btn_browse_out)
        export_form.addRow("Output dir:", out_dir_row)

        # Model checkboxes
        models_layout = QHBoxLayout()
        self.model_checkboxes: dict[str, QCheckBox] = {}
        for model_name in ONNX_MODELS:
            cb = QCheckBox(model_name)
            cb.setChecked(True)
            self.model_checkboxes[model_name] = cb
            models_layout.addWidget(cb)
        export_form.addRow("Target Models:", models_layout)

        # Export / quantize buttons
        btn_row = QHBoxLayout()
        self.btn_export_fp32 = QPushButton("Export FP32")
        self.btn_export_fp32.clicked.connect(lambda: self._on_export(quantize=False))
        btn_row.addWidget(self.btn_export_fp32)
        self.btn_quantize_int8 = QPushButton("Export + Quantize INT8")
        self.btn_quantize_int8.clicked.connect(lambda: self._on_export(quantize=True))
        btn_row.addWidget(self.btn_quantize_int8)
        btn_row.addStretch()
        export_form.addRow("", btn_row)

        self.export_progress = QProgressBar()
        self.export_progress.setValue(0)
        export_form.addRow("", self.export_progress)

        self.export_status_label = QLabel("")
        export_form.addRow("", self.export_status_label)

        layout.addWidget(export_group)

        # --- Simple Status Table ---
        status_group = QGroupBox("Model Status")
        status_layout = QVBoxLayout(status_group)
        self.status_table = QTableWidget(len(ONNX_MODELS), 2)
        self.status_table.setHorizontalHeaderLabels(["Model", "Status"])
        self.status_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for row, name in enumerate(ONNX_MODELS):
            self.status_table.setItem(row, 0, QTableWidgetItem(name))
            self.status_table.setItem(row, 1, QTableWidgetItem("Not Exported"))
        status_layout.addWidget(self.status_table)
        layout.addWidget(status_group)
        layout.addStretch()

    def _on_browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path: self.output_dir_edit.setText(path)

    def _on_export(self, quantize: bool) -> None:
        from tmrvc_gui.workers.export_worker import ExportWorker
        ckpt_path = self.checkpoint_combo.currentText().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not ckpt_path or not output_dir:
            self.export_status_label.setText("Error: Checkpoint and Output dir required.")
            return

        models = [name for name, cb in self.model_checkboxes.items() if cb.isChecked()]
        if not models: return

        self._worker = ExportWorker(Path(ckpt_path), Path(output_dir), models, quantize)
        self._worker.progress.connect(lambda c, t: self.export_progress.setValue(int(100*c/t)))
        self._worker.log_message.connect(self.export_status_label.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

        self.btn_export_fp32.setEnabled(False)
        self.btn_quantize_int8.setEnabled(False)

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_export_fp32.setEnabled(True)
        self.btn_quantize_int8.setEnabled(True)
        self.export_status_label.setText(message)
