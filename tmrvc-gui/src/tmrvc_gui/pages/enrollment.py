"""Speaker Enrollment Page — UCLM v2 Personalization (LoRA)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from tmrvc_gui.widgets.file_drop import FileDropArea

logger = logging.getLogger(__name__)


class _EnrollmentWorker(QThread):
    """Background worker for UCLM v2 Speaker Enrollment & LoRA Fine-tuning."""

    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        audio_paths: list[str],
        speaker_name: str,
        output_path: Path,
        level: str,
        uclm_checkpoint: str | None = None,
        codec_checkpoint: str | None = None,
        finetune_steps: int = 200,
        device: str = "cpu",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._audio_paths = audio_paths
        self._speaker_name = speaker_name
        self._output_path = output_path
        self._level = level
        self._uclm_checkpoint = uclm_checkpoint
        self._codec_checkpoint = codec_checkpoint
        self._finetune_steps = finetune_steps
        self._device = device
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import torch
            import numpy as np
            import soundfile as sf
            import librosa

            if self._cancelled: return

            # 1. Speaker Embedding
            self.progress.emit("Extracting speaker embedding...")
            from tmrvc_train.models.speaker_encoder import SpeakerEncoderWithLoRA
            spk_enc = SpeakerEncoderWithLoRA().to(self._device).eval()
            
            embeds = []
            for p in self._audio_paths:
                audio, sr = sf.read(p)
                if audio.ndim > 1: audio = audio[:, 0]
                if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                with torch.no_grad():
                    e = spk_enc(torch.from_numpy(audio).float().unsqueeze(0).to(self._device))
                    embeds.append(e.cpu())
            
            spk_embed = torch.stack(embeds).mean(dim=0).numpy()

            # 2. Token Extraction (for Standard/Full)
            a_tokens = None
            b_tokens = None
            if self._level in ("standard", "full") and self._codec_checkpoint:
                self.progress.emit("Extracting tokens from reference...")
                from tmrvc_train.models import EmotionAwareCodec
                codec = EmotionAwareCodec().to(self._device).eval()
                codec.load_state_dict(torch.load(self._codec_checkpoint, map_location=self._device)["model"])
                
                # Encode first audio for few-shot ref
                audio, sr = sf.read(self._audio_paths[0])
                if sr != 24000: audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                audio_t = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    a_tokens, b_tokens = codec.encode(audio_t)

            # 3. LoRA Fine-tuning (Full only)
            lora_delta = None
            if self._level == "full" and self._uclm_checkpoint and self._finetune_steps > 0:
                self.progress.emit(f"Fine-tuning UCLM with LoRA ({self._finetune_steps} steps)...")
                from tmrvc_train.models import DisentangledUCLM
                from tmrvc_train.lora import finetune_uclm_lora
                
                uclm = DisentangledUCLM().to(self._device)
                uclm.load_state_dict(torch.load(self._uclm_checkpoint, map_location=self._device)["model"])
                
                # Simplified: use first audio segment for personalization
                lora_delta_t = finetune_uclm_lora(
                    uclm, a_tokens, b_tokens, 
                    torch.from_numpy(spk_embed).to(self._device),
                    torch.zeros(1, a_tokens.shape[-1], 8).to(self._device), # neutral state
                    n_steps=self._finetune_steps, device=self._device
                )
                lora_delta = lora_delta_t.cpu().numpy()

            # 4. Save
            from tmrvc_export.speaker_file import write_speaker_file
            write_speaker_file(
                self._output_path,
                spk_embed.squeeze(),
                lora_delta=lora_delta,
                metadata={"name": self._speaker_name, "level": self._level}
            )
            self.finished.emit(str(self._output_path))

        except Exception as e:
            logger.exception("Enrollment failed")
            self.error.emit(str(e))


class EnrollmentPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        
        # Audio Selection
        group_ref = QGroupBox("Reference Audio")
        layout_ref = QVBoxLayout(group_ref)
        self.drop_area = FileDropArea()
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        layout_ref.addWidget(self.drop_area)
        self.file_list = QListWidget()
        layout_ref.addWidget(self.file_list)
        self.speaker_name = QLineEdit()
        self.speaker_name.setPlaceholderText("Speaker Name")
        layout_ref.addWidget(self.speaker_name)
        layout.addWidget(group_ref)

        # Config
        group_cfg = QGroupBox("UCLM v2 Personalization")
        form = QFormLayout(group_cfg)
        self.level_combo = QComboBox()
        self.level_combo.addItems(["Light (Embed only)", "Full (LoRA Adaptation)"])
        form.addRow("Mode:", self.level_combo)
        
        self.uclm_ckpt = QLineEdit()
        btn_uclm = QPushButton("...")
        btn_uclm.clicked.connect(self._browse_uclm)
        row_uclm = QHBoxLayout()
        row_uclm.addWidget(self.uclm_ckpt)
        row_uclm.addWidget(btn_uclm)
        form.addRow("UCLM Checkpoint:", row_uclm)

        self.codec_ckpt = QLineEdit()
        btn_codec = QPushButton("...")
        btn_codec.clicked.connect(self._browse_codec)
        row_codec = QHBoxLayout()
        row_codec.addWidget(self.codec_ckpt)
        row_codec.addWidget(btn_codec)
        form.addRow("Codec Checkpoint:", row_codec)

        self.steps = QSpinBox()
        self.steps.setRange(10, 1000)
        self.steps.setValue(100)
        form.addRow("LoRA Steps:", self.steps)
        layout.addWidget(group_cfg)

        self.btn_run = QPushButton("Create Speaker Profile (.tmrvc_speaker)")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.clicked.connect(self._on_run)
        layout.addWidget(self.btn_run)
        
        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        layout.addStretch()

    def _on_files_dropped(self, paths):
        for p in paths: self.file_list.addItem(p)

    def _browse_uclm(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select UCLM Checkpoint", "", "*.pt")
        if p: self.uclm_ckpt.setText(p)

    def _browse_codec(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Codec Checkpoint", "", "*.pt")
        if p: self.codec_ckpt.setText(p)

    def _on_run(self):
        paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        if not paths or not self.speaker_name.text(): return
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Profile", f"{self.speaker_name.text()}.tmrvc_speaker")
        if not save_path: return

        self.btn_run.setEnabled(False)
        self._worker = _EnrollmentWorker(
            audio_paths=paths,
            speaker_name=self.speaker_name.text(),
            output_path=Path(save_path),
            level="full" if self.level_combo.currentIndex() == 1 else "light",
            uclm_checkpoint=self.uclm_ckpt.text(),
            codec_checkpoint=self.codec_ckpt.text(),
            finetune_steps=self.steps.value(),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    def _on_finished(self, path):
        self.btn_run.setEnabled(True)
        self.status.setText(f"Done: {path}")
        QMessageBox.information(self, "Success", "Speaker profile created.")

import torch
