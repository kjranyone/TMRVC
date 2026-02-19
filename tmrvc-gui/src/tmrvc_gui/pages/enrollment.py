"""Enrollment page for speaker embedding generation and quick voice test."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tmrvc_gui.widgets.file_drop import FileDropArea

# Speaker embedding dimension from constants
D_SPEAKER: int = 192

logger = logging.getLogger(__name__)


class _SpeakerGenerateWorker(QThread):
    """Background worker for speaker embedding extraction."""

    finished = Signal(str)  # path to generated file
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        audio_paths: list[str],
        speaker_name: str,
        output_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._audio_paths = audio_paths
        self._speaker_name = speaker_name
        self._output_path = output_path

    def run(self) -> None:
        try:
            import torch

            from tmrvc_core.constants import LORA_DELTA_SIZE
            from tmrvc_data.speaker import SpeakerEncoder
            from tmrvc_export.speaker_file import write_speaker_file

            encoder = SpeakerEncoder()
            embeddings = []

            for i, path in enumerate(self._audio_paths):
                self.progress.emit(f"Extracting embedding from file {i + 1}/{len(self._audio_paths)}...")
                emb = encoder.extract_from_file(path)
                embeddings.append(emb)

            # Average embeddings
            self.progress.emit("Averaging embeddings...")
            avg_embed = torch.stack(embeddings).mean(dim=0)
            avg_embed = torch.nn.functional.normalize(avg_embed, p=2, dim=-1)

            # Write speaker file (lora_delta = zeros for now)
            from datetime import datetime, timezone

            spk_embed_np = avg_embed.numpy().astype(np.float32)
            lora_delta_np = np.zeros(LORA_DELTA_SIZE, dtype=np.float32)

            metadata = {
                "profile_name": self._speaker_name,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "description": "",
                "source_audio_files": [Path(p).name for p in self._audio_paths],
                "source_sample_count": 0,
                "training_mode": "embedding",
                "checkpoint_name": "",
            }
            write_speaker_file(
                self._output_path, spk_embed_np, lora_delta_np, metadata=metadata,
            )

            self.finished.emit(str(self._output_path))
        except Exception as exc:
            self.error.emit(str(exc))


class EnrollmentPage(QWidget):
    """Speaker enrollment: reference audio upload and quick voice conversion test."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = None
        self._finetune_worker = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Reference audio section ---
        ref_group = QGroupBox("Reference Audio")
        ref_layout = QVBoxLayout(ref_group)

        # Drop area (drag audio files here -> added to file list)
        self.drop_area = FileDropArea(
            prompt_text="Drop .wav / .flac / .mp3 files here",
        )
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        ref_layout.addWidget(self.drop_area)

        # File list + browse button
        list_row = QHBoxLayout()
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.file_list.setMinimumHeight(80)
        list_row.addWidget(self.file_list)

        list_btn_col = QVBoxLayout()
        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self._on_browse)
        list_btn_col.addWidget(self.btn_browse)

        self.btn_remove = QPushButton("Remove Selected")
        list_btn_col.addWidget(self.btn_remove)
        self.btn_remove.clicked.connect(self._on_remove_selected)

        list_btn_col.addStretch()
        list_row.addLayout(list_btn_col)

        ref_layout.addLayout(list_row)

        # Speaker name + generate
        gen_row = QHBoxLayout()
        gen_row.addWidget(QLabel("Speaker name:"))
        self.speaker_name_edit = QLineEdit()
        self.speaker_name_edit.setPlaceholderText("e.g. Speaker_A")
        gen_row.addWidget(self.speaker_name_edit)

        self.btn_generate = QPushButton("Generate .tmrvc_speaker")
        self.btn_generate.clicked.connect(self._on_generate)
        gen_row.addWidget(self.btn_generate)
        ref_layout.addLayout(gen_row)

        self.status_label = QLabel("")
        ref_layout.addWidget(self.status_label)

        layout.addWidget(ref_group)

        # --- Fine-tune section ---
        ft_group = QGroupBox("Fine-tune (Optional)")
        ft_layout = QVBoxLayout(ft_group)

        # Checkpoint path
        ckpt_row = QHBoxLayout()
        ckpt_row.addWidget(QLabel("Student Checkpoint:"))
        self.ft_checkpoint_edit = QLineEdit()
        self.ft_checkpoint_edit.setPlaceholderText("distill.pt")
        ckpt_row.addWidget(self.ft_checkpoint_edit)
        self.btn_ft_browse_ckpt = QPushButton("Browse...")
        self.btn_ft_browse_ckpt.clicked.connect(self._on_ft_browse_checkpoint)
        ckpt_row.addWidget(self.btn_ft_browse_ckpt)
        ft_layout.addLayout(ckpt_row)

        # Parameters row
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Steps:"))
        self.ft_steps_spin = QSpinBox()
        self.ft_steps_spin.setRange(10, 10000)
        self.ft_steps_spin.setValue(200)
        param_row.addWidget(self.ft_steps_spin)

        param_row.addWidget(QLabel("LR:"))
        self.ft_lr_spin = QDoubleSpinBox()
        self.ft_lr_spin.setRange(1e-5, 1.0)
        self.ft_lr_spin.setDecimals(5)
        self.ft_lr_spin.setValue(1e-3)
        self.ft_lr_spin.setSingleStep(1e-4)
        param_row.addWidget(self.ft_lr_spin)

        self.ft_gtm_check = QCheckBox("Use GTM")
        param_row.addWidget(self.ft_gtm_check)
        ft_layout.addLayout(param_row)

        # Fine-tune button + Cancel button
        ft_btn_row = QHBoxLayout()
        self.btn_finetune = QPushButton("Fine-tune && Save .tmrvc_speaker")
        self.btn_finetune.clicked.connect(self._on_finetune)
        ft_btn_row.addWidget(self.btn_finetune)

        self.btn_cancel_finetune = QPushButton("Cancel")
        self.btn_cancel_finetune.setEnabled(False)
        self.btn_cancel_finetune.clicked.connect(self._on_cancel_finetune)
        ft_btn_row.addWidget(self.btn_cancel_finetune)

        ft_layout.addLayout(ft_btn_row)

        # Progress bar
        self.ft_progress_bar = QProgressBar()
        self.ft_progress_bar.setVisible(False)
        ft_layout.addWidget(self.ft_progress_bar)

        self.ft_status_label = QLabel("")
        ft_layout.addWidget(self.ft_status_label)

        layout.addWidget(ft_group)

        # --- Quick test section ---
        test_group = QGroupBox("Quick Conversion Test")
        test_layout = QVBoxLayout(test_group)

        test_info = QLabel(
            f"Uses the generated speaker embedding (d_speaker={D_SPEAKER}) for a quick offline conversion."
        )
        test_layout.addWidget(test_info)

        btn_row = QHBoxLayout()
        self.btn_record = QPushButton("Record")
        btn_row.addWidget(self.btn_record)

        self.btn_browse_test = QPushButton("Browse Test File...")
        btn_row.addWidget(self.btn_browse_test)

        self.btn_play_result = QPushButton("Play Result")
        self.btn_play_result.setEnabled(False)
        btn_row.addWidget(self.btn_play_result)

        test_layout.addLayout(btn_row)

        # Waveform display placeholder
        self.waveform_frame = QFrame()
        self.waveform_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.waveform_frame.setMinimumHeight(80)
        wf_layout = QVBoxLayout(self.waveform_frame)
        wf_label = QLabel("Waveform Display")
        wf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wf_layout.addWidget(wf_label)
        test_layout.addWidget(self.waveform_frame)

        layout.addWidget(test_group)
        layout.addStretch()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_files_dropped(self, paths: list) -> None:
        """Handle files dropped onto the drop area."""
        existing = {self.file_list.item(i).text() for i in range(self.file_list.count())}
        for path in paths:
            if path not in existing:
                self.file_list.addItem(path)

    def _on_browse(self) -> None:
        """Open a file dialog to select reference audio files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)",
        )
        for f in files:
            self.file_list.addItem(f)

    def _on_remove_selected(self) -> None:
        """Remove selected items from the file list."""
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)

    def _on_generate(self) -> None:
        """Generate .tmrvc_speaker file from listed audio files."""
        audio_paths = self.get_reference_paths()
        if not audio_paths:
            QMessageBox.warning(
                self, "No Audio Files",
                "Please add at least one reference audio file.",
            )
            return

        speaker_name = self.speaker_name_edit.text().strip()
        if not speaker_name:
            QMessageBox.warning(
                self, "No Speaker Name",
                "Please enter a speaker name.",
            )
            return

        # Ask where to save
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Speaker File",
            f"{speaker_name}.tmrvc_speaker",
            "Speaker Files (*.tmrvc_speaker);;All Files (*)",
        )
        if not save_path:
            return

        self.btn_generate.setEnabled(False)
        self.status_label.setText("Generating speaker file...")

        self._worker = _SpeakerGenerateWorker(
            audio_paths, speaker_name, Path(save_path), parent=self,
        )
        self._worker.finished.connect(self._on_generate_finished)
        self._worker.error.connect(self._on_generate_error)
        self._worker.progress.connect(self._on_generate_progress)
        self._worker.start()

    def _on_generate_finished(self, path: str) -> None:
        """Handle successful speaker file generation."""
        self.btn_generate.setEnabled(True)
        self.status_label.setText(f"Speaker file saved: {path}")
        logger.info("Speaker file generated: %s", path)

    def _on_generate_error(self, msg: str) -> None:
        """Handle speaker file generation error."""
        self.btn_generate.setEnabled(True)
        self.status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Generation Error", msg)

    def _on_generate_progress(self, msg: str) -> None:
        """Handle progress updates."""
        self.status_label.setText(msg)

    # ------------------------------------------------------------------
    # Fine-tune slots
    # ------------------------------------------------------------------

    def _on_ft_browse_checkpoint(self) -> None:
        """Browse for student checkpoint file."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Student Checkpoint",
            "",
            "Checkpoint Files (*.pt *.pth);;All Files (*)",
        )
        if path:
            self.ft_checkpoint_edit.setText(path)

    def _on_finetune(self) -> None:
        """Start few-shot LoRA fine-tuning."""
        audio_paths = self.get_reference_paths()
        if not audio_paths:
            QMessageBox.warning(
                self, "No Audio Files",
                "Please add at least one reference audio file.",
            )
            return

        ckpt_path = self.ft_checkpoint_edit.text().strip()
        if not ckpt_path:
            QMessageBox.warning(
                self, "No Checkpoint",
                "Please select a student checkpoint file.",
            )
            return

        speaker_name = self.speaker_name_edit.text().strip()
        if not speaker_name:
            QMessageBox.warning(
                self, "No Speaker Name",
                "Please enter a speaker name.",
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Speaker File",
            f"{speaker_name}.tmrvc_speaker",
            "Speaker Files (*.tmrvc_speaker);;All Files (*)",
        )
        if not save_path:
            return

        self.btn_finetune.setEnabled(False)
        self.btn_cancel_finetune.setEnabled(True)
        self.ft_progress_bar.setVisible(True)
        self.ft_progress_bar.setValue(0)
        self.ft_status_label.setText("Starting fine-tune...")

        from tmrvc_gui.workers.finetune_worker import FinetuneWorker

        config = {
            "audio_paths": audio_paths,
            "checkpoint_path": ckpt_path,
            "output_path": save_path,
            "max_steps": self.ft_steps_spin.value(),
            "lr": self.ft_lr_spin.value(),
            "use_gtm": self.ft_gtm_check.isChecked(),
        }
        self._finetune_worker = FinetuneWorker(config, parent=self)
        self._finetune_worker.progress.connect(self._on_finetune_progress)
        self._finetune_worker.log_message.connect(self._on_finetune_log)
        self._finetune_worker.metric.connect(self._on_finetune_metric)
        self._finetune_worker.finished.connect(self._on_finetune_finished)
        self._finetune_worker.error.connect(self._on_finetune_error)
        self._finetune_worker.start()

    def _on_cancel_finetune(self) -> None:
        """Cancel the running fine-tune worker."""
        if self._finetune_worker is not None:
            self._finetune_worker.cancel()

    def _on_finetune_progress(self, current: int, total: int) -> None:
        self.ft_progress_bar.setMaximum(total)
        self.ft_progress_bar.setValue(current)

    def _on_finetune_log(self, msg: str) -> None:
        self.ft_status_label.setText(msg)

    def _on_finetune_metric(self, name: str, value: float, step: int) -> None:
        self.ft_status_label.setText(f"Step {step}  {name}={value:.4f}")

    def _on_finetune_finished(self, success: bool, msg: str) -> None:
        self.btn_finetune.setEnabled(True)
        self.btn_cancel_finetune.setEnabled(False)
        self.ft_progress_bar.setVisible(False)
        if success:
            self.ft_status_label.setText(msg)
            logger.info("Fine-tune completed: %s", msg)
        else:
            self.ft_status_label.setText(f"Failed: {msg}")

    def _on_finetune_error(self, msg: str) -> None:
        self.btn_finetune.setEnabled(True)
        self.btn_cancel_finetune.setEnabled(False)
        self.ft_progress_bar.setVisible(False)
        self.ft_status_label.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Fine-tune Error", msg)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_reference_paths(self) -> list[str]:
        """Return all reference audio file paths currently listed."""
        return [
            self.file_list.item(i).text()
            for i in range(self.file_list.count())
        ]
