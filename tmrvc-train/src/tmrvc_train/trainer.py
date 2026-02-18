"""TeacherTrainer: multi-phase training loop for the Teacher U-Net."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tmrvc_core.types import TrainingBatch
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.losses import FlowMatchingLoss, MultiResolutionSTFTLoss, SpeakerConsistencyLoss
from tmrvc_train.models.teacher_unet import TeacherUNet

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for teacher training."""

    phase: str = "0"  # "0", "1a", "1b", "2", "reflow"
    lr: float = 2e-4
    warmup_steps: int = 0
    max_steps: int = 100_000
    save_every: int = 10_000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints"
    lambda_stft: float = 0.5
    lambda_spk: float = 0.3
    lambda_ir: float = 0.1
    grad_clip: float = 1.0
    use_wandb: bool = False
    use_ot_cfm: bool = False
    p_uncond: float = 0.0  # Probability of dropping conditioning (CFG-free training)


class TeacherTrainer:
    """Training loop for Teacher U-Net across multiple phases.

    Phase 0:  50K-100K steps, lr=2e-4, L_flow only.
    Phase 1a: 300K-500K steps, lr=1e-4, L_flow.
    Phase 1b: +100K-200K steps, lr=5e-5, L_flow + L_stft + L_spk.
    Phase 2:  +100K-200K steps, lr=5e-5, + L_ir, IR augmentation.
    """

    def __init__(
        self,
        teacher: TeacherUNet,
        scheduler: FlowMatchingScheduler,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        config: TrainerConfig,
    ) -> None:
        self.teacher = teacher
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.global_step = 0

        self.flow_loss = FlowMatchingLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.spk_loss = SpeakerConsistencyLoss()

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # WandB integration (optional)
        self._wandb = None
        if config.use_wandb:
            try:
                import wandb

                wandb.init(
                    project="tmrvc-teacher",
                    config={
                        "phase": config.phase,
                        "lr": config.lr,
                        "max_steps": config.max_steps,
                        "lambda_stft": config.lambda_stft,
                        "lambda_spk": config.lambda_spk,
                        "lambda_ir": config.lambda_ir,
                        "grad_clip": config.grad_clip,
                    },
                    resume="allow",
                )
                self._wandb = wandb
                logger.info("WandB logging enabled (project=tmrvc-teacher)")
            except ImportError:
                logger.warning("wandb not installed, skipping WandB logging")

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        """Execute a single training step.

        Returns:
            Dict of loss values for logging.
        """
        self.teacher.train()
        device = next(self.teacher.parameters()).device
        B = batch.mel_target.shape[0]

        mel_target = batch.mel_target.to(device)
        content = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        # Conditioning dropout (CFG-free training)
        if self.config.p_uncond > 0.0:
            drop_mask = torch.rand(B, device=device) < self.config.p_uncond
            if drop_mask.any():
                content = content.clone()
                spk_embed = spk_embed.clone()
                f0 = f0.clone()
                content[drop_mask] = 0.0
                spk_embed[drop_mask] = 0.0
                f0[drop_mask] = 0.0

        # Random timestep
        t = torch.rand(B, 1, 1, device=device)

        # Forward process
        if self.config.use_ot_cfm:
            x_t, v_target = self.scheduler.ot_forward_process(mel_target, t)
        else:
            x_t, v_target = self.scheduler.forward_process(mel_target, t)

        # Teacher prediction
        v_pred = self.teacher(
            x_t, t.squeeze(-1).squeeze(-1), content, f0, spk_embed,
        )

        # Losses
        losses = {}
        loss_total = self.flow_loss(v_pred, v_target)
        losses["flow"] = loss_total.item()

        phase = self.config.phase
        if phase in ("1b", "2"):
            l_stft = self.stft_loss(v_pred, v_target)
            l_stft_weighted = self.config.lambda_stft * l_stft
            loss_total = loss_total + l_stft_weighted
            losses["stft"] = l_stft.item()

        if phase in ("1b", "2"):
            # Speaker consistency on predicted velocity (proxy)
            pred_mean = v_pred.mean(dim=-1)  # [B, 80]
            target_mean = v_target.mean(dim=-1)  # [B, 80]
            l_spk = self.spk_loss(pred_mean, target_mean)
            l_spk_weighted = self.config.lambda_spk * l_spk
            loss_total = loss_total + l_spk_weighted
            losses["spk"] = l_spk.item()

        losses["total"] = loss_total.item()

        # Backward
        self.optimizer.zero_grad()
        loss_total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.teacher.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return losses

    def train_iter(self):
        """Iterate over training steps, yielding ``(step, losses)`` each step.

        This generator is designed for GUI integration, allowing the
        caller to inspect metrics and check cancellation between steps.

        Yields:
            Tuple of ``(global_step, losses_dict)``.
        """
        while self.global_step < self.config.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.config.max_steps:
                    return

                losses = self.train_step(batch)
                self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
                    logger.info("Step %d: %s", self.global_step, loss_str)

                    if self._wandb is not None:
                        self._wandb.log(losses, step=self.global_step)

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                yield self.global_step, losses

    def train_epoch(self) -> None:
        """Train for one full epoch over the dataloader."""
        for _step, _losses in self.train_iter():
            if self.global_step >= self.config.max_steps:
                break

    def save_checkpoint(self, path: str | None = None) -> Path:
        """Save model checkpoint."""
        if path is None:
            ckpt_path = self.checkpoint_dir / f"teacher_step{self.global_step}.pt"
        else:
            ckpt_path = Path(path)

        torch.save(
            {
                "step": self.global_step,
                "model_state_dict": self.teacher.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            ckpt_path,
        )
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.teacher.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["step"]
        logger.info("Loaded checkpoint from %s (step %d)", path, self.global_step)


class ReflowTrainer:
    """Re-train Teacher on straightened ODE trajectories (Reflow / VoiceFlow).

    Uses pre-generated (noise, clean) pairs from :func:`generate_reflow_pairs`
    instead of random noise, producing straighter flow trajectories that
    enable higher-quality 1-step distillation.
    """

    def __init__(
        self,
        teacher: TeacherUNet,
        scheduler: FlowMatchingScheduler,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        config: TrainerConfig,
    ) -> None:
        self.teacher = teacher
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.global_step = 0

        self.flow_loss = FlowMatchingLoss()

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(
        self,
        batch: TrainingBatch,
        x_1_noise: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Execute a single Reflow training step.

        Args:
            batch: Standard training batch (mel_target = x_0_teacher).
            x_1_noise: ``[B, 80, T]`` pre-generated noise endpoint.
                If ``None``, generated on-the-fly via Teacher ODE.

        Returns:
            Dict of loss values for logging.
        """
        self.teacher.train()
        device = next(self.teacher.parameters()).device
        B = batch.mel_target.shape[0]

        x_0_teacher = batch.mel_target.to(device)
        content = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        if x_1_noise is not None:
            x_1 = x_1_noise.to(device)
        else:
            # Generate noise endpoint on-the-fly (slower but doesn't require pre-generation)
            with torch.no_grad():
                self.teacher.eval()
                x_1, _ = self.scheduler.generate_reflow_pairs(
                    self.teacher, x_0_teacher, steps=20,
                    content=content, f0=f0, spk_embed=spk_embed,
                )
                self.teacher.train()

        # Random timestep
        t = torch.rand(B, 1, 1, device=device)

        # Reflow forward process
        x_t, v_target = self.scheduler.reflow_forward_process(x_0_teacher, x_1, t)

        # Teacher prediction
        v_pred = self.teacher(
            x_t, t.squeeze(-1).squeeze(-1), content, f0, spk_embed,
        )

        # Loss
        loss = self.flow_loss(v_pred, v_target)
        losses = {"flow": loss.item(), "total": loss.item()}

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.teacher.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return losses

    def train_iter(self):
        """Iterate over training steps, yielding ``(step, losses)`` each step."""
        while self.global_step < self.config.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.config.max_steps:
                    return

                losses = self.train_step(batch)
                self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
                    logger.info("Reflow step %d: %s", self.global_step, loss_str)

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                yield self.global_step, losses

    def save_checkpoint(self, path: str | None = None) -> Path:
        """Save model checkpoint."""
        if path is None:
            ckpt_path = self.checkpoint_dir / f"reflow_step{self.global_step}.pt"
        else:
            ckpt_path = Path(path)

        torch.save(
            {
                "step": self.global_step,
                "model_state_dict": self.teacher.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            ckpt_path,
        )
        logger.info("Saved reflow checkpoint to %s", ckpt_path)
        return ckpt_path
