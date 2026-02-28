"""UCLM Trainer for training the Unified Codec Language Model.

Handles:
- Model initialization
- Training loop with loss computation
- Checkpoint saving/loading
- Logging and evaluation
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class UCLMTrainerConfig:
    """Configuration for UCLM training."""

    # Data
    cache_dir: str = "data/cache"
    datasets: list[str] = field(default_factory=lambda: ["libritts_r"])
    max_frames: int = 400
    min_frames: int = 20
    batch_size: int = 16
    num_workers: int = 4

    # Model
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 12
    dropout: float = 0.1

    # Training
    lr: float = 1.0e-4
    weight_decay: float = 0.01
    warmup_steps: int = 10000
    max_steps: int = 200000
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    # Checkpointing
    checkpoint_dir: str = "checkpoints/uclm"
    save_every: int = 5000
    eval_every: int = 1000
    log_every: int = 100

    # Device
    device: str = "cuda"


class UCLMTrainer:
    """Trainer for UCLM model.

    Args:
        config: Trainer configuration.
    """

    def __init__(self, config: UCLMTrainerConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

        # Initialize model
        self._init_model()

        # Initialize data loaders
        self._init_dataloaders()

        # Initialize optimizer
        self._init_optimizer()

        # Training state
        self.global_step = 0
        self.best_loss = float("inf")

    def _init_model(self) -> None:
        """Initialize UCLM model."""
        from tmrvc_train.models.uclm import UCLM, UCLMConfig

        model_config = UCLMConfig(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
        )

        self.model = UCLM(model_config).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("UCLM model initialized: %.2fM parameters", n_params / 1e6)

    def _init_dataloaders(self) -> None:
        """Initialize data loaders."""
        from tmrvc_data.uclm_dataset import UCLMDataset, collate_uclm_batch

        self.train_dataset = UCLMDataset(
            cache_dir=self.config.cache_dir,
            datasets=self.config.datasets,
            max_frames=self.config.max_frames,
            min_frames=self.config.min_frames,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=collate_uclm_batch,
            pin_memory=True,
            drop_last=True,
        )

        logger.info(
            "DataLoader initialized: %d samples, batch_size=%d",
            len(self.train_dataset),
            self.config.batch_size,
        )

    def _init_optimizer(self) -> None:
        """Initialize optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Warmup + cosine decay scheduler
        def lr_lambda(step: int) -> float:
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(
                1, self.config.max_steps - self.config.warmup_steps
            )
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> None:
        """Run training loop."""
        logger.info("Starting training for %d steps", self.config.max_steps)

        self.model.train()
        start_time = time.time()
        accumulated_loss = 0.0

        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break

                # Move batch to device
                batch = self._move_batch_to_device(batch)

                # Forward pass
                loss, loss_dict = self._forward_step(batch)

                # Backward pass
                loss = loss / self.config.gradient_accumulation
                loss.backward()

                accumulated_loss += loss.item()

                # Gradient accumulation
                if (self.global_step + 1) % self.config.gradient_accumulation == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Logging
                    if self.global_step % self.config.log_every == 0:
                        elapsed = time.time() - start_time
                        avg_loss = accumulated_loss / self.config.log_every
                        lr = self.scheduler.get_last_lr()[0]

                        logger.info(
                            "step %d | loss %.4f | lr %.2e | %.1f steps/s",
                            self.global_step,
                            avg_loss,
                            lr,
                            self.config.log_every / elapsed,
                        )

                        accumulated_loss = 0.0
                        start_time = time.time()

                    # Checkpointing
                    if self.global_step % self.config.save_every == 0:
                        self._save_checkpoint()

                self.global_step += 1

        # Final checkpoint
        self._save_checkpoint()
        logger.info("Training completed!")

    def _forward_step(self, batch: Any) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute forward pass and loss.

        Args:
            batch: UCLMBatch with input data.

        Returns:
            Tuple of (total_loss, loss_dict).
        """
        from tmrvc_train.models.uclm import uclm_loss

        # Get text features from phonemes
        # Note: For simplicity, we create a dummy text feature
        # In production, this would use a proper TextEncoder with d_text=256
        # Here we use d_model directly since text_proj is Identity when d_text == d_model
        text_features = torch.zeros(
            batch.phoneme_ids.shape[0],
            batch.phoneme_ids.shape[1],
            self.config.d_model,  # Use d_model since we're using Identity projection
            device=self.device,
        )

        # Forward pass (skip text_proj by using d_model-sized features directly)
        output = self.model(
            text_features=text_features,
            voice_state=batch.voice_state,
            speaker_embed=batch.spk_embed,
            target_tokens=batch.codec_tokens,
            mode="tts",
        )

        # Compute loss
        loss_dict = uclm_loss(
            logits_ar=output["logits_ar"],
            logits_parallel=output["logits_parallel"],
            target_tokens=batch.codec_tokens,
        )

        return loss_dict["loss"], {
            "loss": loss_dict["loss"].item(),
            "loss_ar": loss_dict["loss_ar"].item(),
            "loss_parallel": loss_dict["loss_parallel"].item(),
        }

    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch tensors to device."""
        return batch.__class__(
            codec_tokens=batch.codec_tokens.to(self.device),
            voice_state=batch.voice_state.to(self.device),
            phoneme_ids=batch.phoneme_ids.to(self.device),
            durations=batch.durations.to(self.device),
            spk_embed=batch.spk_embed.to(self.device),
            text=batch.text,
            utterance_ids=batch.utterance_ids,
            frame_lengths=batch.frame_lengths.to(self.device),
            phoneme_lengths=batch.phoneme_lengths.to(self.device),
        )

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.__dict__,
            "best_loss": self.best_loss,
        }

        path = checkpoint_dir / f"uclm_step{self.global_step}.pt"
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint: %s", path)

        # Keep only last 3 checkpoints
        checkpoints = sorted(checkpoint_dir.glob("uclm_step*.pt"))
        if len(checkpoints) > 3:
            for old_ckpt in checkpoints[:-3]:
                old_ckpt.unlink()
                logger.debug("Removed old checkpoint: %s", old_ckpt)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint.get("best_loss", float("inf"))

        logger.info("Loaded checkpoint from step %d", self.global_step)
