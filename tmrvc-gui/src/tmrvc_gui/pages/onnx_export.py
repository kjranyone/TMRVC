"""ONNX export page for model export, quantization, and parity verification."""

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

# The 5 ONNX models that compose the inference pipeline
ONNX_MODELS: list[str] = [
    "content_encoder",
    "converter",
    "ir_estimator",
    "vocoder",
    "speaker_encoder",
]


class OnnxExportPage(QWidget):
    """ONNX model export, INT8 quantization, parity verification, and benchmarking."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Export configuration ---
        export_group = QGroupBox("Export Configuration")
        export_form = QFormLayout(export_group)

        # Checkpoint selector
        self.checkpoint_combo = QComboBox()
        self.checkpoint_combo.setEditable(True)
        self.checkpoint_combo.setPlaceholderText("Select or type checkpoint path")
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
        export_form.addRow("Models:", models_layout)

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

        # --- Parity verification table ---
        parity_group = QGroupBox("Parity Verification (Python vs C++ / FP32 vs INT8)")
        parity_layout = QVBoxLayout(parity_group)

        self.parity_table = QTableWidget(len(ONNX_MODELS), 4)
        self.parity_table.setHorizontalHeaderLabels(
            ["Model", "Max Abs Error", "Mean Abs Error", "Status"]
        )
        parity_header = self.parity_table.horizontalHeader()
        if parity_header is not None:
            parity_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, model_name in enumerate(ONNX_MODELS):
            self.parity_table.setItem(row, 0, QTableWidgetItem(model_name))
            self.parity_table.setItem(row, 1, QTableWidgetItem("--"))
            self.parity_table.setItem(row, 2, QTableWidgetItem("--"))
            self.parity_table.setItem(row, 3, QTableWidgetItem("Pending"))

        self.btn_run_parity = QPushButton("Run Parity Check")
        parity_layout.addWidget(self.parity_table)
        parity_layout.addWidget(self.btn_run_parity)

        layout.addWidget(parity_group)

        # --- Benchmark results table ---
        bench_group = QGroupBox("Inference Benchmark")
        bench_layout = QVBoxLayout(bench_group)

        self.bench_table = QTableWidget(len(ONNX_MODELS), 3)
        self.bench_table.setHorizontalHeaderLabels(
            ["Model", "FP32 Time (ms)", "INT8 Time (ms)"]
        )
        bench_header = self.bench_table.horizontalHeader()
        if bench_header is not None:
            bench_header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)

        for row, model_name in enumerate(ONNX_MODELS):
            self.bench_table.setItem(row, 0, QTableWidgetItem(model_name))
            self.bench_table.setItem(row, 1, QTableWidgetItem("--"))
            self.bench_table.setItem(row, 2, QTableWidgetItem("--"))

        self.btn_run_bench = QPushButton("Run Benchmark")
        bench_layout.addWidget(self.bench_table)
        bench_layout.addWidget(self.btn_run_bench)

        layout.addWidget(bench_group)

    # ------------------------------------------------------------------
    # Worker state
    # ------------------------------------------------------------------

    _worker = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse_output(self) -> None:
        """Open a directory dialog for the export output path."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        if dir_path:
            self.output_dir_edit.setText(dir_path)

    def _on_export(self, quantize: bool) -> None:
        from tmrvc_gui.workers.export_worker import ExportWorker

        ckpt_path = self.checkpoint_combo.currentText().strip()
        output_dir = self.output_dir_edit.text().strip()

        if not ckpt_path:
            self.export_status_label.setText("Select a checkpoint first.")
            return
        if not output_dir:
            self.export_status_label.setText("Set an output directory first.")
            return

        models = self.get_selected_models()
        if not models:
            self.export_status_label.setText("Select at least one model.")
            return

        self._worker = ExportWorker(
            checkpoint_path=Path(ckpt_path),
            output_dir=Path(output_dir),
            models=models,
            quantize=quantize,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.log_message.connect(
            lambda msg: self.export_status_label.setText(msg)
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(
            lambda msg: self.export_status_label.setText(f"ERROR: {msg}")
        )

        self.btn_export_fp32.setEnabled(False)
        self.btn_quantize_int8.setEnabled(False)
        self.export_progress.setValue(0)
        self.export_status_label.setText("Exporting...")
        self._worker.start()

    def _on_progress(self, current: int, total: int) -> None:
        if total > 0:
            self.export_progress.setValue(int(100 * current / total))

    def _on_finished(self, success: bool, message: str) -> None:
        self.btn_export_fp32.setEnabled(True)
        self.btn_quantize_int8.setEnabled(True)
        self.export_status_label.setText(message)
        if success:
            self.export_progress.setValue(100)
        self._worker = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_selected_models(self) -> list[str]:
        """Return the list of model names that are checked for export."""
        return [
            name
            for name, cb in self.model_checkboxes.items()
            if cb.isChecked()
        ]

    def set_parity_result(
        self,
        model_name: str,
        max_abs: float,
        mean_abs: float,
        passed: bool,
    ) -> None:
        """Update a row in the parity verification table."""
        for row in range(self.parity_table.rowCount()):
            item = self.parity_table.item(row, 0)
            if item is not None and item.text() == model_name:
                self.parity_table.setItem(
                    row, 1, QTableWidgetItem(f"{max_abs:.6f}")
                )
                self.parity_table.setItem(
                    row, 2, QTableWidgetItem(f"{mean_abs:.6f}")
                )
                status = "PASS" if passed else "FAIL"
                self.parity_table.setItem(row, 3, QTableWidgetItem(status))
                break

    def set_benchmark_result(
        self,
        model_name: str,
        fp32_ms: float,
        int8_ms: float,
    ) -> None:
        """Update a row in the benchmark results table."""
        for row in range(self.bench_table.rowCount()):
            item = self.bench_table.item(row, 0)
            if item is not None and item.text() == model_name:
                self.bench_table.setItem(
                    row, 1, QTableWidgetItem(f"{fp32_ms:.2f}")
                )
                self.bench_table.setItem(
                    row, 2, QTableWidgetItem(f"{int8_ms:.2f}")
                )
                break
