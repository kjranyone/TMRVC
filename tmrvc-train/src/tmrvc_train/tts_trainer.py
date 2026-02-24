"""TTSTrainer: training loop for TTS front-end models.

Trains TextEncoder, DurationPredictor, F0Predictor, and ContentSynthesizer
while keeping Converter and Vocoder frozen (using VC-trained weights).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tmrvc_core.types import TTSBatch
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor, length_regulate
from tmrvc_train.models.text_encoder import TextEncoder

logger = logging.getLogger(__name__)


@dataclass
class TTSTrainerConfig:
    """Configuration for TTS front-end training."""

    lr: float = 1e-4
    warmup_steps: int = 5000
    max_steps: int = 200_000
    save_every: int = 10_000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints/tts"
    grad_clip: float = 1.0

    # Loss weights
    lambda_duration: float = 1.0
    lambda_f0: float = 0.5
    lambda_content: float = 1.0
    lambda_voiced: float = 0.2

    use_wandb: bool = False


class TTSTrainer:
    """Training loop for TTS front-end.

    Trains the text-to-speech pipeline:
    TextEncoder → DurationPredictor → F0Predictor → ContentSynthesizer

    The content loss aligns ContentSynthesizer output with the frozen
    ContentEncoder's output, ensuring Converter compatibility.

    Args:
        text_encoder: TextEncoder model.
        duration_predictor: DurationPredictor model.
        f0_predictor: F0Predictor model.
        content_synthesizer: ContentSynthesizer model.
        optimizer: Optimizer for all trainable parameters.
        dataloader: DataLoader yielding TTSBatch.
        config: Training configuration.
        lr_scheduler: Optional LR scheduler.
    """

    def __init__(
        self,
        text_encoder: TextEncoder,
        duration_predictor: DurationPredictor,
        f0_predictor: F0Predictor,
        content_synthesizer: ContentSynthesizer,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        config: TTSTrainerConfig,
        lr_scheduler: LambdaLR | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.text_encoder = text_encoder
        self.duration_predictor = duration_predictor
        self.f0_predictor = f0_predictor
        self.content_synthesizer = content_synthesizer
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.dataloader = dataloader
        self.config = config
        self.device = torch.device(device)
        self.global_step = 0

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(project="tmrvc-tts", config=vars(config))
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not available, skipping")

    def train_step(self, batch: TTSBatch) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: TTSBatch with phoneme_ids, durations, content, f0, mel_target.

        Returns:
            Dict of loss values for logging.
        """
        batch = batch.to(self.device)
        device = self.device
        B = batch.phoneme_ids.shape[0]

        # --- Text Encoding ---
        text_features = self.text_encoder(
            batch.phoneme_ids,
            batch.language_ids,
            batch.phoneme_lengths,
        )  # [B, d, L]

        # --- Duration Prediction ---
        pred_durations = self.duration_predictor(
            text_features,
            phoneme_lengths=batch.phoneme_lengths,
        )  # [B, L]

        # Duration loss (MSE in log domain for numerical stability)
        gt_durations = batch.durations.float()
        # Phoneme mask
        L_max = batch.phoneme_ids.shape[1]
        phone_mask = (
            torch.arange(L_max, device=device).unsqueeze(0)
            < batch.phoneme_lengths.unsqueeze(1)
        ).float()  # [B, L]

        dur_loss = F.mse_loss(
            torch.log1p(pred_durations) * phone_mask,
            torch.log1p(gt_durations) * phone_mask,
            reduction="sum",
        ) / phone_mask.sum().clamp(min=1)

        # --- Length Regulation (use ground-truth durations for teacher forcing) ---
        expanded_features = length_regulate(
            text_features, gt_durations,
        )  # [B, d, T]

        # Truncate or pad to match target frame length
        T_target = batch.mel_target.shape[-1]
        T_expanded = expanded_features.shape[-1]
        if T_expanded > T_target:
            expanded_features = expanded_features[:, :, :T_target]
        elif T_expanded < T_target:
            pad_size = T_target - T_expanded
            expanded_features = F.pad(expanded_features, (0, pad_size))

        # Frame mask
        frame_mask = (
            torch.arange(T_target, device=device).unsqueeze(0)
            < batch.frame_lengths.unsqueeze(1)
        ).float().unsqueeze(1)  # [B, 1, T]

        # --- F0 Prediction (with style conditioning if available) ---
        style_vec = batch.style if hasattr(batch, "style") and batch.style is not None else None
        pred_f0, pred_voiced = self.f0_predictor(expanded_features, style_vec)  # [B,1,T], [B,1,T]

        # F0 loss only on voiced frames (where gt f0 > 0)
        gt_voiced_mask = (batch.f0 > 0).float()  # [B, 1, T]
        voiced_frame_mask = gt_voiced_mask * frame_mask
        f0_loss = F.mse_loss(
            pred_f0 * voiced_frame_mask,
            batch.f0 * voiced_frame_mask,
            reduction="sum",
        ) / voiced_frame_mask.sum().clamp(min=1)

        # Voiced/unvoiced classification loss
        # GT voiced: f0 > 0
        gt_voiced = (batch.f0 > 0).float()
        voiced_loss = F.binary_cross_entropy(
            pred_voiced * frame_mask,
            gt_voiced * frame_mask,
            reduction="sum",
        ) / frame_mask.sum().clamp(min=1)

        # --- Content Synthesis ---
        pred_content = self.content_synthesizer(expanded_features)  # [B, 256, T]

        # Align with ground-truth content (from ContentEncoder)
        # Truncate content target to D_CONTENT=256 if it's from ContentVec/WavLM
        gt_content = batch.content
        if gt_content.shape[1] != pred_content.shape[1]:
            # Project or truncate if dimension mismatch
            gt_content = gt_content[:, :pred_content.shape[1], :]

        content_loss = F.mse_loss(
            pred_content * frame_mask,
            gt_content * frame_mask,
            reduction="sum",
        ) / (frame_mask.sum().clamp(min=1) * pred_content.shape[1])

        # --- Total Loss ---
        total = (
            self.config.lambda_duration * dur_loss
            + self.config.lambda_f0 * f0_loss
            + self.config.lambda_content * content_loss
            + self.config.lambda_voiced * voiced_loss
        )

        self.optimizer.zero_grad()
        total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                self._trainable_params(), self.config.grad_clip,
            )
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1

        return {
            "total": total.item(),
            "duration": dur_loss.item(),
            "f0": f0_loss.item(),
            "content": content_loss.item(),
            "voiced": voiced_loss.item(),
        }

    def _trainable_params(self) -> Iterator[nn.Parameter]:
        """Yield all trainable parameters."""
        for model in [
            self.text_encoder,
            self.duration_predictor,
            self.f0_predictor,
            self.content_synthesizer,
        ]:
            yield from model.parameters()

    def train_iter(self) -> Iterator[tuple[int, dict[str, float]]]:
        """Generator-based training loop.

        Yields:
            Tuples of (step_number, losses_dict).
        """
        for model in [
            self.text_encoder,
            self.duration_predictor,
            self.f0_predictor,
            self.content_synthesizer,
        ]:
            model.train()

        while self.global_step < self.config.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                losses = self.train_step(batch)

                if self.global_step % self.config.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Step %d | total=%.4f dur=%.4f f0=%.4f content=%.4f voiced=%.4f lr=%.2e",
                        self.global_step,
                        losses["total"],
                        losses["duration"],
                        losses["f0"],
                        losses["content"],
                        losses["voiced"],
                        lr,
                    )
                    if self._wandb is not None:
                        self._wandb.log(
                            {f"tts/{k}": v for k, v in losses.items()},
                            step=self.global_step,
                        )

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                yield self.global_step, losses

    def save_checkpoint(self, path: Path | None = None) -> Path:
        """Save training checkpoint.

        Args:
            path: Override save path. Default: checkpoint_dir/tts_step{N}.pt

        Returns:
            Path to saved checkpoint.
        """
        if path is None:
            path = self.checkpoint_dir / f"tts_step{self.global_step}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "text_encoder": self.text_encoder.state_dict(),
            "duration_predictor": self.duration_predictor.state_dict(),
            "f0_predictor": self.f0_predictor.state_dict(),
            "content_synthesizer": self.content_synthesizer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Saved TTS checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.global_step = checkpoint["global_step"]
        self.text_encoder.load_state_dict(checkpoint["text_encoder"])
        self.duration_predictor.load_state_dict(checkpoint["duration_predictor"])
        self.f0_predictor.load_state_dict(checkpoint["f0_predictor"])
        self.content_synthesizer.load_state_dict(checkpoint["content_synthesizer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info("Loaded TTS checkpoint from step %d: %s", self.global_step, path)
