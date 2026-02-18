"""DistillationTrainer: Teacher → Student distillation (ODE trajectory + DMD)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tmrvc_core.constants import IR_UPDATE_INTERVAL, N_IR_PARAMS
from tmrvc_core.types import TrainingBatch
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.losses import DMD2Loss, FlowMatchingLoss, MultiResolutionSTFTLoss, SVLoss, SpeakerConsistencyLoss
from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.converter import ConverterStudent
from tmrvc_train.models.discriminator import MelDiscriminator
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.teacher_unet import TeacherUNet
from tmrvc_train.models.vocoder import VocoderStudent

logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""

    phase: str = "A"  # "A", "B", "B2", "C"
    lr: float = 1e-4
    max_steps: int = 200_000
    teacher_steps: int = 20
    save_every: int = 10_000
    log_every: int = 100
    checkpoint_dir: str = "checkpoints/distill"
    lambda_stft: float = 0.5
    lambda_spk: float = 0.3
    lambda_ir: float = 0.1
    grad_clip: float = 1.0
    # Phase B2 (DMD2) settings
    disc_lr: float = 2e-4
    disc_update_ratio: int = 2
    lambda_gan: float = 1.0
    # Phase C (Metric Optimization) settings
    lambda_sv: float = 0.5


class DistillationTrainer:
    """Distill Teacher U-Net into Student models.

    Phase A: ODE Trajectory Pre-training.
      Teacher generates multi-step ODE trajectory.
      Student learns to predict in 1 step.

    Phase B: Distribution Matching Distillation (DMD).
      Refines Student with distribution-level loss.
    """

    def __init__(
        self,
        teacher: TeacherUNet,
        content_encoder: ContentEncoderStudent,
        converter: ConverterStudent,
        vocoder: VocoderStudent,
        ir_estimator: IREstimator,
        scheduler: FlowMatchingScheduler,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        config: DistillationConfig,
        discriminator: MelDiscriminator | None = None,
        disc_optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.teacher = teacher
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.content_encoder = content_encoder
        self.converter = converter
        self.vocoder = vocoder
        self.ir_estimator = ir_estimator
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.config = config
        self.global_step = 0

        self.flow_loss = FlowMatchingLoss()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.spk_loss = SpeakerConsistencyLoss()
        self.dmd2_loss = DMD2Loss()
        self.sv_loss = SVLoss()

        # Discriminator (Phase B2)
        self.discriminator = discriminator
        self.disc_optimizer = disc_optimizer

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step_phase_a(self, batch: TrainingBatch) -> dict[str, float]:
        """Phase A: ODE trajectory distillation.

        Teacher runs multi-step ODE to produce trajectory.
        Student tries to match teacher's velocity in 1 step.
        """
        device = next(self.converter.parameters()).device
        B = batch.mel_target.shape[0]

        mel_target = batch.mel_target.to(device)
        content_teacher = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        # Generate teacher's velocity at a random timestep
        t = torch.rand(B, 1, 1, device=device)
        x_t, v_target = self.scheduler.forward_process(mel_target, t)

        with torch.no_grad():
            v_teacher = self.teacher(
                x_t, t.squeeze(-1).squeeze(-1),
                content_teacher, f0, spk_embed,
            )

        # Student: content encoder → IR estimator → converter
        content_student = self.content_encoder(
            mel_target, f0,
        )[0]  # [B, 256, T]

        # IR Estimator: predict IR params from mel chunks
        # Use chunks of IR_UPDATE_INTERVAL frames, pad if needed
        T = mel_target.shape[-1]
        if T >= IR_UPDATE_INTERVAL:
            mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
        else:
            mel_chunk = torch.nn.functional.pad(
                mel_target, (0, IR_UPDATE_INTERVAL - T),
            )
        ir_params_pred = self.ir_estimator(mel_chunk)[0]  # [B, 24]

        pred_features = self.converter(
            content_student, spk_embed, ir_params_pred,
        )[0]  # [B, 513, T]

        # Project converter output to mel space and match teacher velocity
        pred_mel = pred_features[:, :80, :]  # [B, 80, T]

        # Ensure time dimensions match
        if pred_mel.shape[-1] != v_teacher.shape[-1]:
            pred_mel = torch.nn.functional.interpolate(
                pred_mel, size=v_teacher.shape[-1], mode="linear", align_corners=False,
            )

        # Loss: student mel prediction should match teacher velocity
        loss_flow = self.flow_loss(pred_mel, v_teacher)

        # IR Estimator regression loss: predict zero-room IR params
        # (Teacher target: zeros, representing anechoic reference)
        ir_target = torch.zeros_like(ir_params_pred)
        loss_ir = torch.nn.functional.mse_loss(ir_params_pred, ir_target)

        loss_total = loss_flow + self.config.lambda_ir * loss_ir

        losses = {
            "flow": loss_flow.item(),
            "ir": loss_ir.item(),
            "total": loss_total.item(),
        }

        self.optimizer.zero_grad()
        loss_total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.content_encoder.parameters())
                + list(self.converter.parameters())
                + list(self.vocoder.parameters())
                + list(self.ir_estimator.parameters()),
                self.config.grad_clip,
            )
        self.optimizer.step()

        return losses

    def train_step_phase_b(self, batch: TrainingBatch) -> dict[str, float]:
        """Phase B: Distribution Matching Distillation.

        Student generates samples, teacher scores them.
        Loss = distribution matching + perceptual losses.
        """
        device = next(self.converter.parameters()).device
        B = batch.mel_target.shape[0]

        mel_target = batch.mel_target.to(device)
        content_teacher = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        # Student forward: content encoder → IR estimator → converter
        content_student = self.content_encoder(mel_target, f0)[0]
        T = mel_target.shape[-1]
        if T >= IR_UPDATE_INTERVAL:
            mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
        else:
            mel_chunk = torch.nn.functional.pad(
                mel_target, (0, IR_UPDATE_INTERVAL - T),
            )
        ir_params = self.ir_estimator(mel_chunk)[0]
        pred_features = self.converter(content_student, spk_embed, ir_params)[0]

        # Vocoder: features → mag + phase
        mag, phase, _ = self.vocoder(pred_features)

        # Teacher's "real" score: run teacher on target
        with torch.no_grad():
            t_ones = torch.ones(B, device=device)
            noise = torch.randn_like(mel_target)
            v_real = self.teacher(
                noise, t_ones, content_teacher, f0, spk_embed,
            )

        # DMD loss: score matching
        t_student = torch.ones(B, device=device)
        # Use pred_features as proxy for student's generated mel
        pred_mel_proxy = pred_features[:, :80, :]  # take first 80 channels as proxy
        if pred_mel_proxy.shape[-1] != mel_target.shape[-1]:
            pred_mel_proxy = torch.nn.functional.interpolate(
                pred_mel_proxy, size=mel_target.shape[-1], mode="linear", align_corners=False,
            )
        v_fake = self.teacher(
            pred_mel_proxy.detach(), t_student, content_teacher, f0, spk_embed,
        )

        loss_dmd = ((v_fake - v_real) ** 2).mean()

        # Perceptual losses
        pred_mean = pred_features.mean(dim=-1)[:, :80]
        target_mean = mel_target.mean(dim=-1)
        if pred_mean.shape[-1] < target_mean.shape[-1]:
            pred_mean = torch.nn.functional.pad(
                pred_mean, (0, target_mean.shape[-1] - pred_mean.shape[-1]),
            )
        loss_spk = self.spk_loss(pred_mean, target_mean)

        loss_total = loss_dmd + self.config.lambda_spk * loss_spk
        losses = {
            "dmd": loss_dmd.item(),
            "spk": loss_spk.item(),
            "total": loss_total.item(),
        }

        self.optimizer.zero_grad()
        loss_total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.content_encoder.parameters())
                + list(self.converter.parameters())
                + list(self.vocoder.parameters())
                + list(self.ir_estimator.parameters()),
                self.config.grad_clip,
            )
        self.optimizer.step()

        return losses

    def _student_forward(
        self, batch: TrainingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run student forward pass, returning (pred_features, ir_params, mel_target)."""
        device = next(self.converter.parameters()).device
        mel_target = batch.mel_target.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        content_student = self.content_encoder(mel_target, f0)[0]
        T = mel_target.shape[-1]
        if T >= IR_UPDATE_INTERVAL:
            mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
        else:
            mel_chunk = nn.functional.pad(mel_target, (0, IR_UPDATE_INTERVAL - T))
        ir_params = self.ir_estimator(mel_chunk)[0]
        pred_features = self.converter(content_student, spk_embed, ir_params)[0]
        return pred_features, ir_params, mel_target

    def train_step_phase_b2(self, batch: TrainingBatch) -> dict[str, float]:
        """Phase B2: DMD2 — GAN-based distribution matching.

        Two time-scale update: discriminator updated ``disc_update_ratio`` times
        per student update.
        """
        assert self.discriminator is not None, "Phase B2 requires a discriminator"
        assert self.disc_optimizer is not None, "Phase B2 requires a disc_optimizer"

        device = next(self.converter.parameters()).device
        mel_target = batch.mel_target.to(device)

        # Student forward
        pred_features, ir_params, _ = self._student_forward(batch)
        pred_mel = pred_features[:, :80, :]  # [B, 80, T]
        if pred_mel.shape[-1] != mel_target.shape[-1]:
            pred_mel = nn.functional.interpolate(
                pred_mel, size=mel_target.shape[-1], mode="linear", align_corners=False,
            )

        # --- Discriminator update ---
        logits_real = self.discriminator(mel_target.detach())
        logits_fake = self.discriminator(pred_mel.detach())
        _, loss_disc = self.dmd2_loss(logits_real, logits_fake)

        self.disc_optimizer.zero_grad()
        loss_disc.backward()
        self.disc_optimizer.step()

        # --- Generator (student) update ---
        logits_fake_for_gen = self.discriminator(pred_mel)
        loss_gen, _ = self.dmd2_loss(logits_real.detach(), logits_fake_for_gen)

        loss_total = self.config.lambda_gan * loss_gen

        losses = {
            "gen": loss_gen.item(),
            "disc": loss_disc.item(),
            "total": loss_total.item(),
        }

        self.optimizer.zero_grad()
        loss_total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.content_encoder.parameters())
                + list(self.converter.parameters())
                + list(self.vocoder.parameters())
                + list(self.ir_estimator.parameters()),
                self.config.grad_clip,
            )
        self.optimizer.step()

        return losses

    def train_step_phase_c(self, batch: TrainingBatch) -> dict[str, float]:
        """Phase C: Metric Optimization — SV loss + spectral loss.

        Directly optimizes student with frozen speaker encoder and
        multi-resolution STFT loss for perceptual quality.
        """
        device = next(self.converter.parameters()).device
        mel_target = batch.mel_target.to(device)

        pred_features, _, _ = self._student_forward(batch)
        pred_mel = pred_features[:, :80, :]
        if pred_mel.shape[-1] != mel_target.shape[-1]:
            pred_mel = nn.functional.interpolate(
                pred_mel, size=mel_target.shape[-1], mode="linear", align_corners=False,
            )

        # Multi-resolution STFT loss
        loss_stft = self.stft_loss(pred_mel, mel_target)

        # Speaker verification loss (proxy: use mean of mel as embedding)
        pred_mean = pred_mel.mean(dim=-1)  # [B, 80]
        target_mean = mel_target.mean(dim=-1)  # [B, 80]
        loss_sv = self.sv_loss(pred_mean, target_mean)

        loss_total = self.config.lambda_stft * loss_stft + self.config.lambda_sv * loss_sv

        losses = {
            "stft": loss_stft.item(),
            "sv": loss_sv.item(),
            "total": loss_total.item(),
        }

        self.optimizer.zero_grad()
        loss_total.backward()
        if self.config.grad_clip > 0:
            nn.utils.clip_grad_norm_(
                list(self.content_encoder.parameters())
                + list(self.converter.parameters())
                + list(self.vocoder.parameters())
                + list(self.ir_estimator.parameters()),
                self.config.grad_clip,
            )
        self.optimizer.step()

        return losses

    def train_step(self, batch: TrainingBatch) -> dict[str, float]:
        if self.config.phase == "A":
            return self.train_step_phase_a(batch)
        if self.config.phase == "B2":
            return self.train_step_phase_b2(batch)
        if self.config.phase == "C":
            return self.train_step_phase_c(batch)
        return self.train_step_phase_b(batch)

    def train_iter(self):
        """Iterate over training steps, yielding ``(step, losses)`` each step.

        Yields:
            Tuple of ``(global_step, losses_dict)``.
        """
        self.content_encoder.train()
        self.converter.train()
        self.vocoder.train()
        self.ir_estimator.train()

        while self.global_step < self.config.max_steps:
            for batch in self.dataloader:
                if self.global_step >= self.config.max_steps:
                    return

                losses = self.train_step(batch)
                self.global_step += 1

                if self.global_step % self.config.log_every == 0:
                    loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
                    logger.info("Step %d: %s", self.global_step, loss_str)

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint()

                yield self.global_step, losses

    def train_epoch(self) -> None:
        """Train for one full epoch over the dataloader."""
        for _step, _losses in self.train_iter():
            if self.global_step >= self.config.max_steps:
                break

    def save_checkpoint(self, path: str | None = None) -> Path:
        if path is None:
            ckpt_path = self.checkpoint_dir / f"distill_step{self.global_step}.pt"
        else:
            ckpt_path = Path(path)

        ckpt_data = {
            "step": self.global_step,
            "content_encoder": self.content_encoder.state_dict(),
            "converter": self.converter.state_dict(),
            "vocoder": self.vocoder.state_dict(),
            "ir_estimator": self.ir_estimator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.discriminator is not None:
            ckpt_data["discriminator"] = self.discriminator.state_dict()
        if self.disc_optimizer is not None:
            ckpt_data["disc_optimizer"] = self.disc_optimizer.state_dict()

        torch.save(ckpt_data, ckpt_path)
        logger.info("Saved checkpoint to %s", ckpt_path)
        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.content_encoder.load_state_dict(ckpt["content_encoder"])
        self.converter.load_state_dict(ckpt["converter"])
        self.vocoder.load_state_dict(ckpt["vocoder"])
        if "ir_estimator" in ckpt:
            self.ir_estimator.load_state_dict(ckpt["ir_estimator"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.global_step = ckpt["step"]
        logger.info("Loaded distillation checkpoint from %s (step %d)", path, self.global_step)
