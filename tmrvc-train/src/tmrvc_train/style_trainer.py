"""StyleTrainer: training loop for StyleEncoder.

Phase 3a — Audio-referenced style encoding:
  - Emotion classification (12 categories, cross-entropy)
  - VAD regression (Valence/Arousal/Dominance, MSE)
  - Prosody estimation (speech rate, energy, pitch range, MSE)

Expects a DataLoader yielding dicts with keys:
  mel: [B, 80, T]  reference mel spectrogram
  emotion_id: [B]  integer emotion label (0..11)
  vad: [B, 3]  Valence/Arousal/Dominance float targets (optional)
  prosody: [B, 3]  rate/energy/pitch_range float targets (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tmrvc_train.models.style_encoder import StyleEncoder

logger = logging.getLogger(__name__)


@dataclass
class StyleTrainerConfig:
    lr: float = 5e-4
    warmup_steps: int = 2000
    max_steps: int = 50_000
    save_every: int = 5_000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints/style"
    grad_clip: float = 1.0

    lambda_emotion: float = 1.0
    lambda_vad: float = 0.5
    lambda_prosody: float = 0.3

    use_wandb: bool = False


class StyleTrainer:
    """Training loop for StyleEncoder (Phase 3a).

    Trains the audio-based style encoder with auxiliary classification
    and regression heads for emotion, VAD, and prosody.

    Args:
        model: StyleEncoder model.
        optimizer: Optimizer.
        dataloader: DataLoader yielding batches (see module docstring).
        config: Training config.
        lr_scheduler: Optional LR scheduler.
    """

    def __init__(
        self,
        model: StyleEncoder,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        config: StyleTrainerConfig,
        lr_scheduler: LambdaLR | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.lr_scheduler = lr_scheduler
        self.device = torch.device(device)
        self.global_step = 0

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._wandb = None
        if config.use_wandb:
            try:
                import wandb
                wandb.init(project="tmrvc-style", config=vars(config))
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not available, skipping")

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dict with 'mel', 'emotion_id', optional 'vad' and 'prosody'.

        Returns:
            Dict of loss values.
        """
        mel = batch["mel"].to(self.device)
        emotion_id = batch["emotion_id"].to(self.device)

        # Forward through style encoder
        style = self.model(mel)  # [B, d_style]
        preds = self.model.predict_emotion(style)

        # Emotion classification loss
        emotion_loss = F.cross_entropy(preds["emotion_logits"], emotion_id)

        total = self.config.lambda_emotion * emotion_loss
        losses = {"emotion": emotion_loss.item()}

        # VAD regression (if targets available)
        if "vad" in batch:
            vad_loss = F.mse_loss(preds["vad"], batch["vad"].to(self.device))
            total = total + self.config.lambda_vad * vad_loss
            losses["vad"] = vad_loss.item()

        # Prosody regression (if targets available)
        if "prosody" in batch:
            prosody_loss = F.mse_loss(preds["prosody"], batch["prosody"].to(self.device))
            total = total + self.config.lambda_prosody * prosody_loss
            losses["prosody"] = prosody_loss.item()

        self.optimizer.zero_grad()
        total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.global_step += 1
        losses["total"] = total.item()

        # Track accuracy
        with torch.no_grad():
            pred_ids = preds["emotion_logits"].argmax(dim=-1)
            acc = (pred_ids == emotion_id).float().mean().item()
            losses["accuracy"] = acc

        return losses

    def train_iter(self) -> Iterator[tuple[int, dict[str, float]]]:
        """Generator-based training loop.

        Yields:
            Tuples of (step_number, losses_dict).
        """
        self.model.train()

        while self.global_step < self.config.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.config.max_steps:
                    break

                losses = self.train_step(batch)

                if self.global_step % self.config.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    parts = [f"Step {self.global_step}"]
                    parts.append(f"total={losses['total']:.4f}")
                    parts.append(f"emotion={losses['emotion']:.4f}")
                    if "vad" in losses:
                        parts.append(f"vad={losses['vad']:.4f}")
                    if "prosody" in losses:
                        parts.append(f"prosody={losses['prosody']:.4f}")
                    parts.append(f"acc={losses['accuracy']:.2%}")
                    parts.append(f"lr={lr:.2e}")
                    logger.info(" | ".join(parts))

                    if self._wandb is not None:
                        self._wandb.log(
                            {f"style/{k}": v for k, v in losses.items()},
                            step=self.global_step,
                        )

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                yield self.global_step, losses

    def save_checkpoint(self, path: Path | None = None) -> Path:
        if path is None:
            path = self.checkpoint_dir / f"style_step{self.global_step}.pt"

        checkpoint = {
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, path)
        logger.info("Saved style checkpoint: %s", path)
        return path

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        logger.info("Loaded style checkpoint from step %d: %s", self.global_step, path)
