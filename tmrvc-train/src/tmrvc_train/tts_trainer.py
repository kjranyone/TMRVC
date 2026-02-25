"""TTSTrainer: training loop for TTS front-end models.

Trains TextEncoder, DurationPredictor, F0Predictor, and ContentSynthesizer
while keeping Converter and Vocoder frozen (using VC-trained weights).

When ``enable_ssl=True``, also trains Scene State Latent (SSL) modules:
SceneStateUpdate, DialogueHistoryEncoder, ProsodyStatsPredictor.

When ``enable_bpeh=True``, also trains the Breath-Pause Event Head (BPEH)
for explicit breath onset, duration, intensity, and pause prediction.
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

from tmrvc_core.constants import SSL_PROSODY_STATS_DIM, TOKENIZER_VOCAB_SIZE
from tmrvc_core.types import TTSBatch
from tmrvc_train.base_trainer import BaseTrainer, BaseTrainerConfig
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor, length_regulate
from tmrvc_train.models.text_encoder import TextEncoder

logger = logging.getLogger(__name__)


def extract_prosody_stats(
    f0: torch.Tensor,
    mel: torch.Tensor,
    frame_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract prosody statistics from a batch for SSL supervision.

    Returns ``[B, SSL_PROSODY_STATS_DIM]`` with:
        [0] pitch_mean, [1] pitch_std, [2] energy_mean, [3] energy_std,
        [4] speaking_rate (voiced ratio), [5] voiced_ratio,
        [6] pause_ratio (unvoiced ratio), [7] spectral_tilt.

    Args:
        f0: ``[B, 1, T]`` F0 in Hz.
        mel: ``[B, 80, T]`` log-mel spectrogram.
        frame_mask: ``[B, 1, T]`` valid frame mask.
    """
    B = f0.shape[0]
    device = f0.device
    stats = torch.zeros(B, SSL_PROSODY_STATS_DIM, device=device)

    mask_flat = frame_mask.squeeze(1)  # [B, T]
    n_valid = mask_flat.sum(dim=1).clamp(min=1)  # [B]

    # F0 statistics (voiced frames only)
    f0_flat = f0.squeeze(1)  # [B, T]
    voiced = (f0_flat > 10.0) & (mask_flat > 0)
    n_voiced = voiced.float().sum(dim=1).clamp(min=1)

    for b in range(B):
        v = f0_flat[b][voiced[b]]
        if v.numel() >= 2:
            stats[b, 0] = v.mean()
            stats[b, 1] = v.std()
        elif v.numel() == 1:
            stats[b, 0] = v[0]

    # Energy (mean mel energy per frame)
    energy = mel.mean(dim=1) * mask_flat  # [B, T]
    stats[:, 2] = energy.sum(dim=1) / n_valid
    stats[:, 3] = ((energy - stats[:, 2:3]) ** 2 * mask_flat).sum(dim=1).div(n_valid).sqrt()

    # Voiced / unvoiced ratios
    stats[:, 4] = n_voiced / n_valid  # speaking rate proxy
    stats[:, 5] = voiced.float().sum(dim=1) / n_valid
    stats[:, 6] = 1.0 - stats[:, 5]  # pause ratio

    # Spectral tilt: ratio of low-band (0-20) to high-band (60-80) mel energy
    low_band = mel[:, :20, :].mean(dim=1) * mask_flat
    high_band = mel[:, 60:, :].mean(dim=1) * mask_flat
    low_mean = low_band.sum(dim=1) / n_valid
    high_mean = high_band.sum(dim=1) / n_valid
    stats[:, 7] = (low_mean - high_mean)  # tilt (positive = more low energy)

    return stats


@dataclass
class TTSTrainerConfig(BaseTrainerConfig):
    """Configuration for TTS front-end training."""

    warmup_steps: int = 5000
    max_steps: int = 200_000
    checkpoint_dir: str = "checkpoints/tts"

    # Loss weights
    lambda_duration: float = 1.0
    lambda_f0: float = 0.5
    lambda_content: float = 1.0
    lambda_voiced: float = 0.2

    # SSL (Scene State Latent)
    enable_ssl: bool = False
    lambda_state_recon: float = 0.5
    lambda_state_cons: float = 0.3

    # BPEH (Breath-Pause Event Head)
    enable_bpeh: bool = False
    lambda_onset: float = 0.5
    lambda_dur: float = 0.3
    lambda_amp: float = 0.2

    text_frontend: str = "tokenizer"
    text_vocab_size: int = TOKENIZER_VOCAB_SIZE


class TTSTrainer(BaseTrainer):
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
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(optimizer, dataloader, config, lr_scheduler)
        self.text_encoder = text_encoder
        self.duration_predictor = duration_predictor
        self.f0_predictor = f0_predictor
        self.content_synthesizer = content_synthesizer
        if device is None:
            self.device = next(self.text_encoder.parameters()).device
        else:
            self.device = torch.device(device)

        # --- SSL modules (optional) ---
        self.ssl_state_update = None
        self.ssl_history_encoder = None
        self.ssl_prosody_predictor = None
        self.ssl_loss_fn = None
        if config.enable_ssl:
            from tmrvc_train.models.scene_state import (
                DialogueHistoryEncoder,
                ProsodyStatsPredictor,
                SceneStateLoss,
                SceneStateUpdate,
            )
            self.ssl_state_update = SceneStateUpdate().to(self.device)
            self.ssl_history_encoder = DialogueHistoryEncoder().to(self.device)
            self.ssl_prosody_predictor = ProsodyStatsPredictor().to(self.device)
            self.ssl_loss_fn = SceneStateLoss(
                lambda_recon=config.lambda_state_recon,
                lambda_cons=config.lambda_state_cons,
            )
            logger.info("SSL enabled: SceneStateUpdate + DialogueHistoryEncoder + ProsodyStatsPredictor")

        # --- BPEH modules (optional) ---
        self.bpeh_head = None
        self.bpeh_loss_fn = None
        if config.enable_bpeh:
            from tmrvc_train.models.breath_event_head import (
                BreathEventHead,
                BreathEventLoss,
            )
            self.bpeh_head = BreathEventHead().to(self.device)
            self.bpeh_loss_fn = BreathEventLoss(
                lambda_onset=config.lambda_onset,
                lambda_dur=config.lambda_dur,
                lambda_amp=config.lambda_amp,
            )
            logger.info("BPEH enabled: BreathEventHead")

    @property
    def _wandb_project(self) -> str:
        return "tts"

    @property
    def _wandb_prefix(self) -> str:
        return "tts"

    @property
    def _increment_step_in_train_step(self) -> bool:
        return True

    def _pre_train_setup(self) -> None:
        models = [
            self.text_encoder,
            self.duration_predictor,
            self.f0_predictor,
            self.content_synthesizer,
        ]
        for ssl_mod in [self.ssl_state_update, self.ssl_history_encoder, self.ssl_prosody_predictor]:
            if ssl_mod is not None:
                models.append(ssl_mod)
        if self.bpeh_head is not None:
            models.append(self.bpeh_head)
        for model in models:
            model.train()

    def _log_step(self, losses: dict[str, float]) -> None:
        lr = self.optimizer.param_groups[0]["lr"]
        ssl_str = ""
        if "state_total" in losses:
            ssl_str = f" ssl={losses['state_total']:.4f}"
        bpeh_str = ""
        if "event_total" in losses:
            bpeh_str = f" bpeh={losses['event_total']:.4f}"
        logger.info(
            "Step %d | total=%.4f dur=%.4f f0=%.4f content=%.4f voiced=%.4f%s%s lr=%.2e",
            self.global_step,
            losses["total"],
            losses["duration"],
            losses["f0"],
            losses["content"],
            losses["voiced"],
            ssl_str,
            bpeh_str,
            lr,
        )
        if self._wandb is not None:
            self._wandb.log(
                {f"tts/{k}": v for k, v in losses.items()},
                step=self.global_step,
            )

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

        # --- SSL (Scene State Latent) ---
        ssl_losses: dict[str, float] = {}
        ssl_total = torch.tensor(0.0, device=device)
        if self.ssl_state_update is not None:
            # u_t: mean-pooled text features as utterance encoding
            u_t = (text_features * phone_mask.unsqueeze(1)).sum(dim=2) / phone_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B, d]

            # h_t: zero history (single-utterance training; dialogue data added in WP2)
            from tmrvc_core.constants import D_HISTORY
            h_t = torch.zeros(B, D_HISTORY, device=device)

            # Scene state: start from zero (no cross-batch persistence in training)
            z_prev = self.ssl_state_update.initial_state(B, device)
            z_t = self.ssl_state_update(z_prev, u_t, h_t, batch.spk_embed)

            # Prosody stats: extract from GT and predict from scene state
            gt_prosody = extract_prosody_stats(batch.f0, batch.mel_target, frame_mask)
            pred_prosody = self.ssl_prosody_predictor(z_t)

            ssl_loss_dict = self.ssl_loss_fn(pred_prosody, gt_prosody.detach(), z_t, z_prev)
            ssl_total = ssl_loss_dict["state_total"]
            ssl_losses = {k: v.item() for k, v in ssl_loss_dict.items()}

        # --- BPEH (Breath-Pause Event Head) ---
        bpeh_losses: dict[str, float] = {}
        bpeh_total = torch.tensor(0.0, device=device)
        if self.bpeh_head is not None:
            pred_onset_logits, pred_breath_dur, pred_intensity, pred_pause_dur = (
                self.bpeh_head(expanded_features)
            )

            # GT event tensors from batch (zeros if None)
            gt_onset = batch.breath_onsets if batch.breath_onsets is not None else torch.zeros(B, T_target, device=device)
            gt_breath_dur = batch.breath_durations if batch.breath_durations is not None else torch.zeros(B, T_target, device=device)
            gt_intensity = batch.breath_intensity if batch.breath_intensity is not None else torch.zeros(B, T_target, device=device)
            gt_pause_dur = batch.pause_durations if batch.pause_durations is not None else torch.zeros(B, T_target, device=device)

            bpeh_loss_dict = self.bpeh_loss_fn(
                pred_onset_logits, pred_breath_dur, pred_intensity, pred_pause_dur,
                gt_onset, gt_breath_dur, gt_intensity, gt_pause_dur,
                frame_mask,
            )
            bpeh_total = bpeh_loss_dict["event_total"]
            bpeh_losses = {k: v.item() for k, v in bpeh_loss_dict.items()}

        # --- Total Loss ---
        total = (
            self.config.lambda_duration * dur_loss
            + self.config.lambda_f0 * f0_loss
            + self.config.lambda_content * content_loss
            + self.config.lambda_voiced * voiced_loss
            + ssl_total
            + bpeh_total
        )

        self._backward_step(total, self._trainable_params())

        self.global_step += 1

        losses = {
            "total": total.item(),
            "duration": dur_loss.item(),
            "f0": f0_loss.item(),
            "content": content_loss.item(),
            "voiced": voiced_loss.item(),
        }
        losses.update(ssl_losses)
        losses.update(bpeh_losses)
        return losses

    def _trainable_params(self) -> Iterator[nn.Parameter]:
        """Yield all trainable parameters."""
        for model in [
            self.text_encoder,
            self.duration_predictor,
            self.f0_predictor,
            self.content_synthesizer,
        ]:
            yield from model.parameters()
        # SSL modules
        for ssl_module in [
            self.ssl_state_update,
            self.ssl_history_encoder,
            self.ssl_prosody_predictor,
        ]:
            if ssl_module is not None:
                yield from ssl_module.parameters()
        # BPEH module
        if self.bpeh_head is not None:
            yield from self.bpeh_head.parameters()

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
            "text_frontend": self.config.text_frontend,
            "text_vocab_size": self.config.text_vocab_size,
            "enable_ssl": self.config.enable_ssl,
            "enable_bpeh": self.config.enable_bpeh,
        }
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.ssl_state_update is not None:
            checkpoint["ssl_state_update"] = self.ssl_state_update.state_dict()
            checkpoint["ssl_history_encoder"] = self.ssl_history_encoder.state_dict()
            checkpoint["ssl_prosody_predictor"] = self.ssl_prosody_predictor.state_dict()
        if self.bpeh_head is not None:
            checkpoint["bpeh_head"] = self.bpeh_head.state_dict()

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
        if self.ssl_state_update is not None and "ssl_state_update" in checkpoint:
            self.ssl_state_update.load_state_dict(checkpoint["ssl_state_update"])
            self.ssl_history_encoder.load_state_dict(checkpoint["ssl_history_encoder"])
            self.ssl_prosody_predictor.load_state_dict(checkpoint["ssl_prosody_predictor"])
        if self.bpeh_head is not None and "bpeh_head" in checkpoint:
            self.bpeh_head.load_state_dict(checkpoint["bpeh_head"])
        logger.info("Loaded TTS checkpoint from step %d: %s", self.global_step, path)
