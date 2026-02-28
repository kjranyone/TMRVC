"""Speaker Enrollment Page — Hierarchical Adaptation (v3 format).

Uses tmrvc-export CLI logic for actual processing.
"""

from __future__ import annotations

import logging
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
    """Background worker that uses tmrvc_export CLI logic."""

    finished = Signal(str)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        audio_paths: list[str],
        speaker_name: str,
        output_path: Path,
        level: str,
        codec_checkpoint: str | None = None,
        token_model: str | None = None,
        finetune_steps: int = 200,
        max_ref_frames: int = 150,
        device: str = "cpu",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._audio_paths = audio_paths
        self._speaker_name = speaker_name
        self._output_path = output_path
        self._level = level
        self._codec_checkpoint = codec_checkpoint
        self._token_model = token_model
        self._finetune_steps = finetune_steps
        self._max_ref_frames = max_ref_frames
        self._device = device
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            import torch

            if self._cancelled:
                return

            # 1. Extract spk_embed
            self.progress.emit("Extracting speaker embedding...")
            from tmrvc_data.speaker import SpeakerEncoder

            encoder = SpeakerEncoder(device=self._device)
            embeddings = []
            for path in self._audio_paths:
                if self._cancelled:
                    return
                emb = encoder.extract_from_file(path)
                embeddings.append(emb)

            avg_embed = torch.stack(embeddings).mean(dim=0)
            avg_embed = torch.nn.functional.normalize(avg_embed, p=2, dim=-1)
            spk_embed = avg_embed.numpy().astype(np.float32)

            # 2. Extract style_embed (standard/full)
            style_embed = None
            if self._level in ("standard", "full"):
                self.progress.emit("Extracting style embedding...")
                from tmrvc_data.style import compute_style_from_files

                style_embed = compute_style_from_files(
                    self._audio_paths, device=self._device
                )

            # 3. Extract reference_tokens (standard/full)
            reference_tokens = None
            if self._level in ("standard", "full") and self._codec_checkpoint:
                self.progress.emit("Extracting reference tokens...")
                import soundfile as sf
                import numpy as np
                import librosa

                from tmrvc_train.models.streaming_codec import (
                    StreamingCodec,
                    CodecConfig,
                )

                ckpt = torch.load(
                    self._codec_checkpoint,
                    map_location=self._device,
                    weights_only=False,
                )
                codec = StreamingCodec(CodecConfig())
                if "state_dict" in ckpt:
                    codec.load_state_dict(ckpt["state_dict"])
                else:
                    codec.load_state_dict(ckpt)
                codec = codec.to(self._device).eval()

                all_tokens = []
                with torch.no_grad():
                    for path in self._audio_paths:
                        if self._cancelled:
                            return
                        audio, sr = sf.read(path)
                        if audio.ndim > 1:
                            audio = audio[:, 0]
                        if sr != 24000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
                        audio_tensor = (
                            torch.from_numpy(audio.astype(np.float32))
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .to(self._device)
                        )
                        indices, _, _ = codec.encode(audio_tensor)
                        indices_np = (
                            indices.squeeze(0)
                            .transpose(0, 1)
                            .cpu()
                            .numpy()
                            .astype(np.int32)
                        )
                        all_tokens.append(indices_np)
                        if sum(len(t) for t in all_tokens) >= self._max_ref_frames:
                            break

                if all_tokens:
                    reference_tokens = np.concatenate(all_tokens, axis=0)[
                        : self._max_ref_frames
                    ]

            # 4. LoRA fine-tuning (full)
            lora_delta = None
            if self._level == "full" and self._token_model and self._finetune_steps > 0:
                if reference_tokens is None:
                    self.progress.emit("ERROR: Full level requires reference tokens")
                else:
                    self.progress.emit(
                        f"Fine-tuning with LoRA ({self._finetune_steps} steps)..."
                    )
                    from tmrvc_train.models.token_model import (
                        TokenModel,
                        TokenModelConfig,
                    )
                    from tmrvc_train.lora import finetune_token_model_lora

                    ckpt = torch.load(
                        self._token_model, map_location=self._device, weights_only=False
                    )
                    token_model = TokenModel(TokenModelConfig())
                    if "model_state_dict" in ckpt:
                        token_model.load_state_dict(ckpt["model_state_dict"])
                    elif "state_dict" in ckpt:
                        token_model.load_state_dict(ckpt["state_dict"])
                    else:
                        token_model.load_state_dict(ckpt)

                    ref_tokens_tensor = torch.from_numpy(reference_tokens).long()
                    spk_embed_tensor = torch.from_numpy(spk_embed).float()

                    delta_flat = finetune_token_model_lora(
                        model=token_model,
                        reference_tokens=ref_tokens_tensor,
                        spk_embed=spk_embed_tensor,
                        n_steps=self._finetune_steps,
                        lr=1e-4,
                        device=self._device,
                    )
                    lora_delta = delta_flat.cpu().numpy().astype(np.float32)

            # 5. Write speaker file (v3 format)
            self.progress.emit("Writing speaker file...")
            from tmrvc_export.speaker_file import write_speaker_file
            from datetime import datetime, timezone

            metadata = {
                "profile_name": self._speaker_name,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "description": f"Created with TMRVC GUI ({self._level} level)",
                "source_audio_files": [Path(p).name for p in self._audio_paths],
                "adaptation_level": self._level,
                "checkpoint_name": Path(self._token_model).name
                if self._token_model
                else "",
            }

            write_speaker_file(
                output_path=self._output_path,
                spk_embed=spk_embed,
                style_embed=style_embed,
                reference_tokens=reference_tokens,
                lora_delta=lora_delta,
                metadata=metadata,
            )

            self.finished.emit(str(self._output_path))

        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class EnrollmentPage(QWidget):
    """Speaker enrollment with hierarchical adaptation levels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._worker = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # --- Reference audio section ---
        ref_group = QGroupBox("Reference Audio")
        ref_layout = QVBoxLayout(ref_group)

        self.drop_area = FileDropArea(prompt_text="Drop .wav / .flac / .mp3 files here")
        self.drop_area.files_dropped.connect(self._on_files_dropped)
        ref_layout.addWidget(self.drop_area)

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

        self.btn_remove = QPushButton("Remove")
        self.btn_remove.clicked.connect(self._on_remove_selected)
        list_btn_col.addWidget(self.btn_remove)
        list_btn_col.addStretch()
        list_row.addLayout(list_btn_col)
        ref_layout.addLayout(list_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Speaker Name:"))
        self.speaker_name_edit = QLineEdit()
        self.speaker_name_edit.setPlaceholderText("e.g. Character_A")
        name_row.addWidget(self.speaker_name_edit)
        ref_layout.addLayout(name_row)

        self.duration_label = QLabel("Add audio files to see duration")
        self.duration_label.setStyleSheet("color: gray;")
        ref_layout.addWidget(self.duration_label)

        layout.addWidget(ref_group)

        # --- Adaptation Level ---
        level_group = QGroupBox("Adaptation Level")
        level_layout = QFormLayout(level_group)

        self.level_combo = QComboBox()
        self.level_combo.addItems(["Light", "Standard", "Full"])
        self.level_combo.setCurrentIndex(1)
        self.level_combo.currentIndexChanged.connect(self._on_level_changed)
        level_layout.addRow("Level:", self.level_combo)

        self.level_desc = QLabel()
        self._update_level_description()
        level_layout.addRow("", self.level_desc)

        layout.addWidget(level_group)

        # --- Level-specific options ---
        self.options_stack = QStackedWidget()

        # Light
        light_widget = QWidget()
        light_layout = QVBoxLayout(light_widget)
        light_layout.addWidget(
            QLabel("Quick speaker cloning with embedding only (3-10 sec audio).")
        )
        self.options_stack.addWidget(light_widget)

        # Standard
        standard_widget = QWidget()
        standard_layout = QFormLayout(standard_widget)
        self.codec_edit = QLineEdit()
        self.codec_edit.setPlaceholderText("checkpoints/codec/best.pt")
        btn_codec = QPushButton("Browse...")
        btn_codec.clicked.connect(
            lambda: self._browse_file(self.codec_edit, "Codec Checkpoint", "*.pt *.pth")
        )
        codec_row = QHBoxLayout()
        codec_row.addWidget(self.codec_edit)
        codec_row.addWidget(btn_codec)
        standard_layout.addRow("Codec Checkpoint:", codec_row)

        self.max_ref_frames_spin = QSpinBox()
        self.max_ref_frames_spin.setRange(10, 500)
        self.max_ref_frames_spin.setValue(150)
        standard_layout.addRow("Max Ref Frames:", self.max_ref_frames_spin)
        self.options_stack.addWidget(standard_widget)

        # Full
        full_widget = QWidget()
        full_layout = QFormLayout(full_widget)

        self.token_model_edit = QLineEdit()
        self.token_model_edit.setPlaceholderText("checkpoints/token_student.pt")
        btn_token = QPushButton("Browse...")
        btn_token.clicked.connect(
            lambda: self._browse_file(
                self.token_model_edit, "Token Model", "*.pt *.pth"
            )
        )
        token_row = QHBoxLayout()
        token_row.addWidget(self.token_model_edit)
        token_row.addWidget(btn_token)
        full_layout.addRow("Token Model:", token_row)

        self.finetune_steps_spin = QSpinBox()
        self.finetune_steps_spin.setRange(0, 10000)
        self.finetune_steps_spin.setValue(200)
        full_layout.addRow("Finetune Steps:", self.finetune_steps_spin)
        self.options_stack.addWidget(full_widget)

        layout.addWidget(self.options_stack)

        # --- Generate button ---
        gen_row = QHBoxLayout()
        self.btn_generate = QPushButton("Generate .tmrvc_speaker")
        self.btn_generate.clicked.connect(self._on_generate)
        gen_row.addWidget(self.btn_generate)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel)
        gen_row.addWidget(self.btn_cancel)
        layout.addLayout(gen_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self._on_level_changed(self.level_combo.currentIndex())

    def _update_level_description(self) -> None:
        level = self.level_combo.currentText()
        descriptions = {
            "Light": "3-10 sec audio, <1 sec processing. spk_embed only.",
            "Standard": "10-30 sec audio, ~5 sec processing. + style_embed + reference_tokens.",
            "Full": "1-5 min audio, 1-5 min processing. + LoRA fine-tuning for best quality.",
        }
        self.level_desc.setText(descriptions.get(level, ""))

    def _on_level_changed(self, index: int) -> None:
        self._update_level_description()
        self.options_stack.setCurrentIndex(index)

    def _browse_file(self, line_edit: QLineEdit, title: str, filter: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, "", filter)
        if path:
            line_edit.setText(path)

    def _on_files_dropped(self, paths: list) -> None:
        existing = {
            self.file_list.item(i).text() for i in range(self.file_list.count())
        }
        for path in paths:
            if path not in existing:
                self.file_list.addItem(path)
        self._update_duration_label()

    def _on_browse(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Reference Audio",
            "",
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)",
        )
        for f in files:
            self.file_list.addItem(f)
        self._update_duration_label()

    def _on_remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
        self._update_duration_label()

    def _update_duration_label(self) -> None:
        paths = self.get_reference_paths()
        if not paths:
            self.duration_label.setText("Add audio files to see duration")
            return

        total_sec = 0.0
        for p in paths:
            try:
                import soundfile as sf

                info = sf.info(p)
                total_sec += info.duration
            except Exception:
                pass

        mins, secs = divmod(int(total_sec), 60)
        suggestion = (
            "Light" if total_sec < 10 else ("Standard" if total_sec < 60 else "Full")
        )
        self.duration_label.setText(
            f"Total: {mins}:{secs:02d} ({len(paths)} files) — Suggested: {suggestion}"
        )

    def get_reference_paths(self) -> list[str]:
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]

    def _on_generate(self) -> None:
        audio_paths = self.get_reference_paths()
        if not audio_paths:
            QMessageBox.warning(self, "No Audio", "Please add reference audio files.")
            return

        speaker_name = self.speaker_name_edit.text().strip()
        if not speaker_name:
            QMessageBox.warning(self, "No Name", "Please enter a speaker name.")
            return

        level = self.level_combo.currentText().lower()

        if level == "full" and not self.token_model_edit.text():
            QMessageBox.warning(
                self, "No Token Model", "Full level requires a token model checkpoint."
            )
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Speaker File",
            f"{speaker_name}.tmrvc_speaker",
            "Speaker Files (*.tmrvc_speaker)",
        )
        if not save_path:
            return

        self.btn_generate.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Starting...")

        self._worker = _EnrollmentWorker(
            audio_paths=audio_paths,
            speaker_name=speaker_name,
            output_path=Path(save_path),
            level=level,
            codec_checkpoint=self.codec_edit.text() or None,
            token_model=self.token_model_edit.text() or None,
            finetune_steps=self.finetune_steps_spin.value() if level == "full" else 0,
            max_ref_frames=self.max_ref_frames_spin.value(),
            device=self._get_device(),
            parent=self,
        )
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.start()

    def _get_device(self) -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _on_cancel(self) -> None:
        if self._worker:
            self._worker.cancel()
            self.status_label.setText("Cancelling...")

    def _on_finished(self, path: str) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Saved: {path}")
        QMessageBox.information(self, "Success", f"Speaker file created:\n{path}")

    def _on_error(self, msg: str) -> None:
        self.btn_generate.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Error", msg)

    def _on_progress(self, msg: str) -> None:
        self.status_label.setText(msg)
