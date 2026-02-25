"""TeacherTrainer: multi-phase training loop for the Teacher U-Net."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from tmrvc_core.constants import IR_UPDATE_INTERVAL, N_IR_PARAMS
from tmrvc_core.types import TrainingBatch
from tmrvc_train.base_trainer import BaseTrainer, BaseTrainerConfig
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.losses import FlowMatchingLoss, MultiResolutionSTFTLoss, SpeakerConsistencyLoss
from tmrvc_train.models.teacher_unet import TeacherUNet

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig(BaseTrainerConfig):
    """Configuration for teacher training."""

    phase: str = "0"  # "0", "1a", "1b", "2", "reflow"
    lr: float = 2e-4
    warmup_steps: int = 0
    lambda_stft: float = 0.5
    lambda_spk: float = 0.3
    lambda_ir: float = 0.1
    lambda_voice: float = 0.2
    use_ot_cfm: bool = False
    p_uncond: float = 0.0  # Probability of dropping conditioning (CFG-free training)
    voice_source_checkpoint: str | None = None


class TeacherTrainer(BaseTrainer):
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
        ir_estimator: nn.Module | None = None,
        voice_source_loss: nn.Module | None = None,
        lr_scheduler: LambdaLR | None = None,
    ) -> None:
        super().__init__(optimizer, dataloader, config, lr_scheduler)
        self.teacher = teacher
        self.scheduler = scheduler

        self.flow_loss = FlowMatchingLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.spk_loss = SpeakerConsistencyLoss()

        # Phase 2: IR estimator and voice source loss
        self.ir_estimator = ir_estimator
        self.voice_source_loss = voice_source_loss

    @property
    def _wandb_project(self) -> str:
        return "teacher"

    def _init_wandb(self) -> None:
        try:
            import wandb

            wandb.init(
                project="tmrvc-teacher",
                config={
                    "phase": self.config.phase,
                    "lr": self.config.lr,
                    "max_steps": self.config.max_steps,
                    "lambda_stft": self.config.lambda_stft,
                    "lambda_spk": self.config.lambda_spk,
                    "lambda_ir": self.config.lambda_ir,
                    "grad_clip": self.config.grad_clip,
                },
                resume="allow",
            )
            self._wandb = wandb
            logger.info("WandB logging enabled (project=tmrvc-teacher)")
        except ImportError:
            logger.warning("wandb not installed, skipping WandB logging")

    def _format_log(self, losses: dict[str, float]) -> str:
        loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
        cur_lr = self.optimizer.param_groups[0]["lr"]
        return f"{loss_str} (lr={cur_lr:.2e})"

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
        T = mel_target.shape[-1]

        # Length mask: [B, 1, T] — 1 for valid frames, 0 for padding
        lengths = batch.lengths.to(device)
        time_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        time_mask = time_mask.unsqueeze(1).float()  # [B, 1, T]

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

        # Forward process (zero noise in padded regions — F5-TTS/VoiceFlow style)
        if self.config.use_ot_cfm:
            x_t, v_target = self.scheduler.ot_forward_process(mel_target, t, mask=time_mask)
        else:
            x_t, v_target = self.scheduler.forward_process(mel_target, t, mask=time_mask)

        # Phase 2: compute IR params BEFORE teacher forward so we can condition on them
        phase = self.config.phase
        acoustic_params_pred = None
        if phase == "2" and self.ir_estimator is not None:
            T = mel_target.shape[-1]
            if T >= IR_UPDATE_INTERVAL:
                mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
            else:
                mel_chunk = nn.functional.pad(
                    mel_target, (0, IR_UPDATE_INTERVAL - T),
                )
            acoustic_params_pred = self.ir_estimator(mel_chunk)[0]  # [B, 32]

        # Teacher prediction (single forward — with IR params in Phase 2)
        v_pred = self.teacher(
            x_t, t.squeeze(-1).squeeze(-1), content, f0, spk_embed,
            acoustic_params=acoustic_params_pred,
        )

        # Losses (masked by valid lengths to ignore padding)
        losses = {}
        loss_total = self.flow_loss(v_pred, v_target, mask=time_mask)
        losses["flow"] = loss_total.item()

        if phase in ("1b", "2"):
            # Mask pred/target before STFT: zero out padded regions
            v_pred_masked = v_pred * time_mask
            v_target_masked = v_target * time_mask
            l_stft = self.stft_loss(v_pred_masked, v_target_masked)
            l_stft_weighted = self.config.lambda_stft * l_stft
            loss_total = loss_total + l_stft_weighted
            losses["stft"] = l_stft.item()

        if phase in ("1b", "2"):
            # Speaker consistency: masked time-average
            denom = time_mask.sum(dim=-1).clamp(min=1)  # [B, 1]
            pred_mean = (v_pred * time_mask).sum(dim=-1) / denom  # [B, 80]
            target_mean = (v_target * time_mask).sum(dim=-1) / denom
            l_spk = self.spk_loss(pred_mean, target_mean)
            l_spk_weighted = self.config.lambda_spk * l_spk
            loss_total = loss_total + l_spk_weighted
            losses["spk"] = l_spk.item()

        if phase == "2" and acoustic_params_pred is not None:
            # IR params regression: zero-room target for env params (0:N_IR_PARAMS)
            ir_target = torch.zeros_like(acoustic_params_pred[:, :N_IR_PARAMS])
            l_ir = nn.functional.mse_loss(
                acoustic_params_pred[:, :N_IR_PARAMS], ir_target,
            )
            loss_total = loss_total + self.config.lambda_ir * l_ir
            losses["ir"] = l_ir.item()

            # Voice source external distillation (optional)
            if self.voice_source_loss is not None:
                voice_source = acoustic_params_pred[:, N_IR_PARAMS:]
                l_voice = self.voice_source_loss(mel_target, voice_source)
                loss_total = loss_total + self.config.lambda_voice * l_voice
                losses["voice"] = l_voice.item()

        losses["total"] = loss_total.item()

        # Backward
        self._backward_step(loss_total, self.teacher.parameters())

        return losses

    def save_checkpoint(self, path: str | None = None) -> Path:
        """Save model checkpoint."""
        if path is None:
            ckpt_path = self.checkpoint_dir / f"teacher_step{self.global_step}.pt"
        else:
            ckpt_path = Path(path)

        ckpt_data = {
            "step": self.global_step,
            "model_state_dict": self.teacher.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.lr_scheduler is not None:
            ckpt_data["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        torch.save(ckpt_data, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if "model_state_dict" not in ckpt:
            raise RuntimeError(
                f"Unsupported checkpoint format: {path} "
                "(missing 'model_state_dict'). Legacy checkpoints are not supported."
            )
        if "optimizer_state_dict" not in ckpt:
            raise RuntimeError(
                f"Unsupported checkpoint format: {path} "
                "(missing 'optimizer_state_dict'). Legacy checkpoints are not supported."
            )
        if "step" not in ckpt or not isinstance(ckpt["step"], int):
            raise RuntimeError(
                f"Unsupported checkpoint metadata: {path} "
                "(missing/invalid 'step')."
            )

        state_dict = ckpt["model_state_dict"]
        model_sd = self.teacher.state_dict()

        missing_keys = sorted(set(model_sd) - set(state_dict))
        unexpected_keys = sorted(set(state_dict) - set(model_sd))
        shape_mismatches = [
            (k, tuple(state_dict[k].shape), tuple(model_sd[k].shape))
            for k in model_sd.keys() & state_dict.keys()
            if state_dict[k].shape != model_sd[k].shape
        ]
        if missing_keys or unexpected_keys or shape_mismatches:
            details: list[str] = []
            if missing_keys:
                details.append(f"missing={missing_keys[:5]}")
            if unexpected_keys:
                details.append(f"unexpected={unexpected_keys[:5]}")
            if shape_mismatches:
                k, got, expected = shape_mismatches[0]
                details.append(f"shape_mismatch={k}: {got} != {expected}")
            raise RuntimeError(
                "Checkpoint is incompatible with current TeacherUNet and was rejected. "
                + "; ".join(details)
            )

        self.teacher.load_state_dict(state_dict, strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["step"]

        # Restore LR scheduler state.
        if self.lr_scheduler is not None:
            if "lr_scheduler_state_dict" not in ckpt:
                raise RuntimeError(
                    f"Checkpoint {path} is missing 'lr_scheduler_state_dict'. "
                    "Legacy checkpoints are not supported."
                )
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
            logger.info("Restored LR scheduler state")
        logger.info("Loaded checkpoint from %s (step %d)", path, self.global_step)


class ReflowTrainer(BaseTrainer):
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
        super().__init__(optimizer, dataloader, config)
        self.teacher = teacher
        self.scheduler = scheduler

        self.flow_loss = FlowMatchingLoss()

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
        T = x_0_teacher.shape[-1]

        # Length mask for padding zeroing
        lengths = batch.lengths.to(device)
        time_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        time_mask = time_mask.unsqueeze(1).float()  # [B, 1, T]

        if x_1_noise is not None:
            x_1 = x_1_noise.to(device) * time_mask
        else:
            # Generate noise endpoint on-the-fly (slower but doesn't require pre-generation)
            with torch.no_grad():
                self.teacher.eval()
                x_1, _ = self.scheduler.generate_reflow_pairs(
                    self.teacher, x_0_teacher, steps=20,
                    content=content, f0=f0, spk_embed=spk_embed,
                )
                x_1 = x_1 * time_mask
                self.teacher.train()

        # Random timestep
        t = torch.rand(B, 1, 1, device=device)

        # Reflow forward process (x_0 and x_1 already masked)
        x_0_masked = x_0_teacher * time_mask
        x_t, v_target = self.scheduler.reflow_forward_process(x_0_masked, x_1, t)

        # Teacher prediction
        v_pred = self.teacher(
            x_t, t.squeeze(-1).squeeze(-1), content, f0, spk_embed,
        )

        # Loss (masked)
        loss = self.flow_loss(v_pred, v_target, mask=time_mask)
        losses = {"flow": loss.item(), "total": loss.item()}

        # Backward
        self._backward_step(loss, self.teacher.parameters())

        return losses

    def _log_step(self, losses: dict[str, float]) -> None:
        logger.info("Reflow step %d: %s", self.global_step, self._format_log(losses))

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

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.teacher.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt["step"]
        logger.info("Loaded reflow checkpoint from %s (step %d)", path, self.global_step)
