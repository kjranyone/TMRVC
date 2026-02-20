"""DistillationTrainer: Teacher → Student distillation (ODE trajectory + DMD)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tmrvc_core.audio import create_mel_filterbank
from tmrvc_core.constants import IR_UPDATE_INTERVAL, N_IR_PARAMS, N_ACOUSTIC_PARAMS
from tmrvc_core.types import TrainingBatch
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.losses import (
    DMD2Loss,
    FlowMatchingLoss,
    MultiResolutionSTFTLoss,
    SVLoss,
    SpeakerConsistencyLoss,
)
from tmrvc_train.models.content_encoder import ContentEncoderStudent
from tmrvc_train.models.converter import ConverterStudent
from tmrvc_train.models.discriminator import MelDiscriminator
from tmrvc_train.models.ir_estimator import IREstimator
from tmrvc_train.models.teacher_unet import TeacherUNet
from tmrvc_train.models.vocoder import VocoderStudent
from tmrvc_train.models.voice_source_estimator import (
    VoiceSourceEstimator,
    VoiceSourceDistillationLoss,
    create_voice_source_teacher,
)
from tmrvc_train.voice_source_stats import VoiceSourceStatsTracker

logger = logging.getLogger(__name__)

# VQ commitment loss weight
VQ_COMMITMENT_LAMBDA = 0.25


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
    lambda_voice: float = 0.2  # Voice source external distillation weight
    grad_clip: float = 1.0
    # Phase B2 (DMD2) settings
    disc_lr: float = 2e-4
    disc_update_ratio: int = 2
    lambda_gan: float = 1.0
    # Phase C (Metric Optimization) settings
    lambda_sv: float = 0.5
    # Voice source external distillation
    voice_source_checkpoint: str | None = None


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
        speaker_encoder: nn.Module | None = None,
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

        # Speaker encoder (Phase C) — frozen if provided
        self.speaker_encoder = speaker_encoder
        if speaker_encoder is not None:
            speaker_encoder.eval()
            for p in speaker_encoder.parameters():
                p.requires_grad = False

        # Mel filterbank for converting STFT features to mel
        self.mel_basis = create_mel_filterbank()  # [80, 513]

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Voice source statistics tracker
        self.voice_source_tracker = VoiceSourceStatsTracker()

        # Voice source external distillation (Phase 2)
        self.voice_source_teacher: VoiceSourceEstimator | None = None
        self.voice_source_loss: VoiceSourceDistillationLoss | None = None
        self._voice_source_on_device = False
        if config.voice_source_checkpoint:
            self.voice_source_teacher = create_voice_source_teacher(
                config.voice_source_checkpoint,
                device="cpu",  # Moved to correct device on first train_step
            )
            if self.voice_source_teacher is not None:
                self.voice_source_loss = VoiceSourceDistillationLoss(
                    self.voice_source_teacher,
                    lambda_voice=config.lambda_voice,
                )
                logger.info(
                    f"Voice source external distillation enabled (λ={config.lambda_voice})"
                )

    def _vq_step(self, device: torch.device) -> torch.Tensor:
        """Run VQ EMA update and return commitment loss (0 if VQ disabled)."""
        if self.content_encoder.vq is not None:
            self.content_encoder.update_vq_ema()
            return getattr(
                self.content_encoder, "_vq_commitment_loss",
                torch.tensor(0.0, device=device),
            )
        return torch.tensor(0.0, device=device)

    def _pred_to_mel(self, pred_features: torch.Tensor) -> torch.Tensor:
        """Convert converter output to log-mel via vocoder + mel filterbank.

        Args:
            pred_features: ``[B, 513, T]`` converter output.

        Returns:
            ``[B, 80, T]`` log-mel spectrogram.
        """
        mag, _phase, _ = self.vocoder(pred_features)  # [B, 513, T]
        power = mag.pow(2)  # [B, 513, T]
        mel_basis = self.mel_basis.to(device=power.device, dtype=power.dtype)
        mel = torch.matmul(mel_basis, power)  # [B, 80, T]
        return mel.clamp(min=1e-10).log()

    def train_step_phase_a(self, batch: TrainingBatch) -> dict[str, float]:
        """Phase A: ODE Trajectory Pre-training.

        Teacher generates mel via multi-step ODE. Student learns to match
        teacher output in 1 step.
        """
        device = next(self.converter.parameters()).device

        mel_target = batch.mel_target.to(device)
        content_teacher = batch.content.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        # Teacher generates target mel via multi-step ODE
        with torch.no_grad():
            teacher_mel = self.scheduler.sample(
                self.teacher,
                shape=mel_target.shape,
                steps=self.config.teacher_steps,
                device=str(device),
                content=content_teacher,
                f0=f0,
                spk_embed=spk_embed,
            )

        # Student: content encoder → IR estimator → converter
        content_student = self.content_encoder(
            mel_target,
            f0,
        )[0]  # [B, 256, T]

        # IR Estimator: predict acoustic params from mel chunks
        # Use chunks of IR_UPDATE_INTERVAL frames, pad if needed
        T = mel_target.shape[-1]
        if T >= IR_UPDATE_INTERVAL:
            mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
        else:
            mel_chunk = torch.nn.functional.pad(
                mel_target,
                (0, IR_UPDATE_INTERVAL - T),
            )
        acoustic_params_pred = self.ir_estimator(mel_chunk)[0]  # [B, 32]

        # VQ EMA update + commitment loss
        vq_loss = self._vq_step(device)

        # Track voice source statistics
        if batch.speaker_ids:
            self.voice_source_tracker.update(acoustic_params_pred, batch.speaker_ids)

        pred_features = self.converter(
            content_student,
            spk_embed,
            acoustic_params_pred,
        )[0]  # [B, 513, T]

        # Convert to proper mel via vocoder + mel filterbank
        pred_mel = self._pred_to_mel(pred_features)  # [B, 80, T]

        # Ensure time dimensions match
        if pred_mel.shape[-1] != mel_target.shape[-1]:
            pred_mel = torch.nn.functional.interpolate(
                pred_mel,
                size=mel_target.shape[-1],
                mode="linear",
                align_corners=False,
            )

        # Loss: L1 against teacher-generated mel (not GT mel)
        if teacher_mel.shape[-1] != pred_mel.shape[-1]:
            teacher_mel = torch.nn.functional.interpolate(
                teacher_mel,
                size=pred_mel.shape[-1],
                mode="linear",
                align_corners=False,
            )
        loss_mel = nn.functional.l1_loss(pred_mel, teacher_mel)

        # IR params regression loss: zero-room target for env params (0-23)
        ir_target = torch.zeros_like(acoustic_params_pred[:, :N_IR_PARAMS])
        loss_ir = torch.nn.functional.mse_loss(
            acoustic_params_pred[:, :N_IR_PARAMS],
            ir_target,
        )

        # Voice source params (24-31): external distillation or zero regularization
        voice_source = acoustic_params_pred[:, N_IR_PARAMS:]
        if self.voice_source_loss is not None:
            # External distillation from pretrained estimator
            if self.voice_source_teacher is not None and not self._voice_source_on_device:
                self.voice_source_teacher.to(device)
                self._voice_source_on_device = True
            loss_voice = self.voice_source_loss(mel_target, voice_source)
        else:
            # Zero regularization (Phase A default)
            loss_voice = torch.nn.functional.mse_loss(
                voice_source,
                torch.zeros_like(voice_source),
            )
            loss_voice = self.config.lambda_voice * loss_voice

        loss_total = (
            loss_mel + self.config.lambda_ir * loss_ir + loss_voice
            + VQ_COMMITMENT_LAMBDA * vq_loss
        )

        losses = {
            "mel": loss_mel.item(),
            "ir": loss_ir.item(),
            "voice": loss_voice.item(),
            "total": loss_total.item(),
        }
        if vq_loss.item() > 0:
            losses["vq"] = vq_loss.item()

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

        # Student forward: content encoder → acoustic estimator → converter
        content_student = self.content_encoder(mel_target, f0)[0]
        vq_loss = self._vq_step(device)
        T = mel_target.shape[-1]
        if T >= IR_UPDATE_INTERVAL:
            mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
        else:
            mel_chunk = torch.nn.functional.pad(
                mel_target,
                (0, IR_UPDATE_INTERVAL - T),
            )
        acoustic_params = self.ir_estimator(mel_chunk)[0]
        pred_features = self.converter(content_student, spk_embed, acoustic_params)[0]

        # Convert to proper mel via vocoder + mel filterbank
        pred_mel_proxy = self._pred_to_mel(pred_features)  # [B, 80, T]
        if pred_mel_proxy.shape[-1] != mel_target.shape[-1]:
            pred_mel_proxy = torch.nn.functional.interpolate(
                pred_mel_proxy,
                size=mel_target.shape[-1],
                mode="linear",
                align_corners=False,
            )

        # Length mask for padding zeroing
        T = mel_target.shape[-1]
        lengths = batch.lengths.to(device)
        time_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        time_mask = time_mask.unsqueeze(1).float()  # [B, 1, T]

        # DMD loss: random timestep sampling (not fixed t=1)
        t = torch.rand(B, device=device)  # uniform [0, 1]
        noise = torch.randn_like(mel_target) * time_mask

        # Real data ODE interpolation: x_t = (1-t)*mel + t*noise
        t_expanded = t.view(B, 1, 1)
        x_t_real = (1 - t_expanded) * mel_target + t_expanded * noise
        with torch.no_grad():
            v_real = self.teacher(
                x_t_real, t, content_teacher, f0, spk_embed, acoustic_params,
            )

        # Student output ODE interpolation at same timestep
        x_t_fake = (1 - t_expanded) * pred_mel_proxy + t_expanded * noise
        v_fake = self.teacher(
            x_t_fake, t, content_teacher, f0, spk_embed, acoustic_params,
        )

        loss_dmd = ((v_fake - v_real) ** 2).mean()

        # Perceptual losses: use pred_mel_proxy (log-mel) directly
        pred_mean = pred_mel_proxy.mean(dim=-1)  # [B, 80]
        target_mean = mel_target.mean(dim=-1)  # [B, 80]
        loss_spk = self.spk_loss(pred_mean, target_mean)

        loss_total = (
            loss_dmd + self.config.lambda_spk * loss_spk
            + VQ_COMMITMENT_LAMBDA * vq_loss
        )
        losses = {
            "dmd": loss_dmd.item(),
            "spk": loss_spk.item(),
            "total": loss_total.item(),
        }
        if vq_loss.item() > 0:
            losses["vq"] = vq_loss.item()

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
        self,
        batch: TrainingBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run student forward pass, returning (pred_features, acoustic_params, mel_target)."""
        device = next(self.converter.parameters()).device
        mel_target = batch.mel_target.to(device)
        f0 = batch.f0.to(device)
        spk_embed = batch.spk_embed.to(device)

        content_student = self.content_encoder(mel_target, f0)[0]
        # VQ EMA update (commitment loss handled by caller if needed)
        if self.content_encoder.vq is not None:
            self.content_encoder.update_vq_ema()
        T = mel_target.shape[-1]
        if T >= IR_UPDATE_INTERVAL:
            mel_chunk = mel_target[:, :, :IR_UPDATE_INTERVAL]
        else:
            mel_chunk = nn.functional.pad(mel_target, (0, IR_UPDATE_INTERVAL - T))
        acoustic_params = self.ir_estimator(mel_chunk)[0]

        # Track voice source statistics
        if batch.speaker_ids:
            self.voice_source_tracker.update(acoustic_params, batch.speaker_ids)

        pred_features = self.converter(content_student, spk_embed, acoustic_params)[0]
        return pred_features, acoustic_params, mel_target

    def train_step_phase_b2(self, batch: TrainingBatch) -> dict[str, float]:
        """Phase B2: DMD2 — GAN-based distribution matching.

        Two time-scale update: discriminator updated ``disc_update_ratio`` times
        per student update.  The student forward is executed once and the
        detached fake mel is reused for all discriminator updates.
        """
        assert self.discriminator is not None, "Phase B2 requires a discriminator"
        assert self.disc_optimizer is not None, "Phase B2 requires a disc_optimizer"

        device = next(self.converter.parameters()).device
        mel_target = batch.mel_target.to(device)

        # --- Single student forward (reused for disc and gen) ---
        pred_features, acoustic_params, _ = self._student_forward(batch)
        vq_loss = getattr(
            self.content_encoder, "_vq_commitment_loss",
            torch.tensor(0.0, device=device),
        )
        pred_mel = self._pred_to_mel(pred_features)
        if pred_mel.shape[-1] != mel_target.shape[-1]:
            pred_mel = nn.functional.interpolate(
                pred_mel,
                size=mel_target.shape[-1],
                mode="linear",
                align_corners=False,
            )

        # --- Discriminator updates (disc_update_ratio times, detached fake) ---
        pred_mel_detached = pred_mel.detach()
        loss_disc_total = 0.0
        for _ in range(self.config.disc_update_ratio):
            logits_real = self.discriminator(mel_target.detach())
            logits_fake = self.discriminator(pred_mel_detached)
            _, loss_disc = self.dmd2_loss(logits_real, logits_fake)

            self.disc_optimizer.zero_grad()
            loss_disc.backward()
            self.disc_optimizer.step()
            loss_disc_total += loss_disc.item()

        # --- Generator (student) update (1 time, graph-connected pred_mel) ---
        logits_fake_for_gen = self.discriminator(pred_mel)
        loss_gen, _ = self.dmd2_loss(
            self.discriminator(mel_target.detach()).detach(),
            logits_fake_for_gen,
        )

        loss_total = (
            self.config.lambda_gan * loss_gen
            + VQ_COMMITMENT_LAMBDA * vq_loss
        )

        losses = {
            "gen": loss_gen.item(),
            "disc": loss_disc_total / max(self.config.disc_update_ratio, 1),
            "total": loss_total.item(),
        }
        if vq_loss.item() > 0:
            losses["vq"] = vq_loss.item()

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
        vq_loss = getattr(
            self.content_encoder, "_vq_commitment_loss",
            torch.tensor(0.0, device=device),
        )
        pred_mel = self._pred_to_mel(pred_features)  # [B, 80, T]
        if pred_mel.shape[-1] != mel_target.shape[-1]:
            pred_mel = nn.functional.interpolate(
                pred_mel,
                size=mel_target.shape[-1],
                mode="linear",
                align_corners=False,
            )

        # Multi-resolution STFT loss
        loss_stft = self.stft_loss(pred_mel, mel_target)

        # Speaker verification loss
        if self.speaker_encoder is not None:
            with torch.no_grad():
                pred_spk, _ = self.speaker_encoder(pred_mel)
                target_spk, _ = self.speaker_encoder(mel_target)
            loss_sv = self.sv_loss(pred_spk, target_spk)
        else:
            # Fallback: mel mean proxy
            pred_mean = pred_mel.mean(dim=-1)  # [B, 80]
            target_mean = mel_target.mean(dim=-1)  # [B, 80]
            loss_sv = self.sv_loss(pred_mean, target_mean)

        loss_total = (
            self.config.lambda_stft * loss_stft + self.config.lambda_sv * loss_sv
            + VQ_COMMITMENT_LAMBDA * vq_loss
        )

        losses = {
            "stft": loss_stft.item(),
            "sv": loss_sv.item(),
            "total": loss_total.item(),
        }
        if vq_loss.item() > 0:
            losses["vq"] = vq_loss.item()

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

        # Save voice source stats alongside checkpoint
        stats_path = ckpt_path.with_suffix(".voice_source_stats.json")
        self.voice_source_tracker.save(stats_path)

        return ckpt_path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.content_encoder.load_state_dict(ckpt["content_encoder"])
        self.converter.load_state_dict(ckpt["converter"])
        self.vocoder.load_state_dict(ckpt["vocoder"])
        self.ir_estimator.load_state_dict(ckpt["ir_estimator"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.discriminator is not None:
            self.discriminator.load_state_dict(ckpt["discriminator"])
        if self.disc_optimizer is not None:
            self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        self.global_step = ckpt["step"]
        logger.info(
            "Loaded distillation checkpoint from %s (step %d)", path, self.global_step
        )

        # Load voice source stats
        stats_path = Path(path).with_suffix(".voice_source_stats.json")
        if stats_path.exists():
            self.voice_source_tracker = VoiceSourceStatsTracker.load(stats_path)
