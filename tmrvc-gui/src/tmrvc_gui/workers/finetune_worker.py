"""Background worker for few-shot LoRA fine-tuning.

Wraps :class:`~tmrvc_train.fewshot.FewShotFinetuner` in a
:class:`BaseWorker` so that fine-tuning can be driven from the GUI with
live progress, metrics, and cancellation support.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base_worker import BaseWorker

logger = logging.getLogger(__name__)


class FinetuneWorker(BaseWorker):
    """Background worker for few-shot LoRA fine-tuning.

    Parameters
    ----------
    config : dict
        Fine-tuning configuration with the following keys:

        * **audio_paths** (*list[str]*) -- Reference audio file paths.
        * **checkpoint_path** (*str | Path*) -- Distillation checkpoint.
        * **output_path** (*str | Path*) -- Output .tmrvc_speaker path.
        * **max_steps** (*int*) -- Fine-tuning steps (default: 200).
        * **lr** (*float*) -- Learning rate (default: 1e-3).
        * **use_gtm** (*bool*) -- Use GTM adapter (default: False).
    """

    def __init__(self, config: dict[str, Any], parent=None) -> None:
        super().__init__(parent)
        self.audio_paths: list[str] = config["audio_paths"]
        self.checkpoint_path: Path = Path(config["checkpoint_path"])
        self.output_path: Path = Path(config["output_path"])
        self.max_steps: int = config.get("max_steps", 200)
        self.lr: float = config.get("lr", 1e-3)
        self.use_gtm: bool = config.get("use_gtm", False)

    def run(self) -> None:
        self._safe_run(self._run_finetune)

    def _run_finetune(self) -> None:
        import torch

        from tmrvc_data.speaker import SpeakerEncoder
        from tmrvc_export.speaker_file import write_speaker_file
        from tmrvc_train.fewshot import FewShotConfig, FewShotFinetuner
        from tmrvc_train.models.content_encoder import ContentEncoderStudent
        from tmrvc_train.models.converter import ConverterStudent, ConverterStudentGTM

        n_audio = len(self.audio_paths)
        total_work = n_audio + self.max_steps

        # 1. Speaker embedding extraction
        self.log_message.emit("Extracting speaker embeddings...")
        encoder = SpeakerEncoder()
        embeddings = []
        for i, path in enumerate(self.audio_paths):
            if self.is_cancelled:
                self.finished.emit(False, "Cancelled")
                return
            self.progress.emit(i, total_work)
            emb = encoder.extract_from_file(path)
            embeddings.append(emb)

        spk_embed = torch.stack(embeddings).mean(0)
        spk_embed = torch.nn.functional.normalize(spk_embed, p=2, dim=-1)

        # 2. Load student models
        self.log_message.emit("Loading student models...")
        ckpt = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False,
        )

        content_encoder = ContentEncoderStudent()
        content_encoder.load_state_dict(ckpt["content_encoder"])

        if self.use_gtm:
            converter = ConverterStudentGTM()
            converter.load_state_dict(ckpt["converter_gtm"])
        else:
            converter = ConverterStudent()
            converter.load_state_dict(ckpt["converter"])

        # 3. Fine-tune
        config = FewShotConfig(
            max_steps=self.max_steps, lr=self.lr, use_gtm=self.use_gtm,
        )
        finetuner = FewShotFinetuner(
            converter, content_encoder, spk_embed, config,
        )

        self.log_message.emit("Preparing data...")
        data = finetuner.prepare_data(self.audio_paths)

        self.log_message.emit("Fine-tuning LoRA...")
        for step, loss in finetuner.finetune_iter(data):
            if self.is_cancelled:
                self.finished.emit(False, "Cancelled")
                return
            self.progress.emit(n_audio + step, total_work)
            if step % config.log_every == 0:
                self.metric.emit("loss", loss, step)

        # 4. Save .tmrvc_speaker
        lora_delta = finetuner.get_lora_delta()
        write_speaker_file(
            self.output_path,
            spk_embed.numpy().astype("float32"),
            lora_delta.detach().numpy().astype("float32"),
        )
        self.log_message.emit(f"Saved: {self.output_path}")
        self.finished.emit(True, f"Fine-tuning complete: {self.output_path}")
