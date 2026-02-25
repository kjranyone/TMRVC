"""Breath-Pause Event Head (BPEH) — predict breath/pause events from text features.

Predicts four frame-level outputs from length-regulated text features:
- Breath onset logits (binary classification)
- Breath duration (ms)
- Breath intensity (0-1)
- Pause duration (ms)

Losses:
- L_onset: focal loss for class-imbalanced onset detection
- L_dur: L1 on breath + pause durations (at event frames only)
- L_amp: L1 on breath intensity (at onset frames only)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import BPEH_D_HIDDEN, BPEH_N_BLOCKS, D_TEXT_ENCODER


class _ConvBlock(nn.Module):
    """Conv1d → ReLU → Conv1d residual block."""

    def __init__(self, d_hidden: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(d_hidden, d_hidden, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class BreathEventHead(nn.Module):
    """Predict breath/pause events from expanded text features.

    Args:
        d_input: Input feature dimension (D_TEXT_ENCODER).
        d_hidden: Hidden dimension for conv blocks.
        n_blocks: Number of residual conv blocks.
    """

    def __init__(
        self,
        d_input: int = D_TEXT_ENCODER,
        d_hidden: int = BPEH_D_HIDDEN,
        n_blocks: int = BPEH_N_BLOCKS,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(d_input, d_hidden, 1)
        self.blocks = nn.Sequential(*[_ConvBlock(d_hidden) for _ in range(n_blocks)])

        # 4 output heads
        self.onset_head = nn.Conv1d(d_hidden, 1, 1)      # logits
        self.breath_dur_head = nn.Conv1d(d_hidden, 1, 1)  # ms (Softplus)
        self.intensity_head = nn.Conv1d(d_hidden, 1, 1)   # 0-1 (Sigmoid)
        self.pause_dur_head = nn.Conv1d(d_hidden, 1, 1)   # ms (Softplus)

    def forward(
        self, expanded_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict breath/pause events.

        Args:
            expanded_features: ``[B, D_input, T]`` length-regulated text features.

        Returns:
            Tuple of:
            - onset_logits: ``[B, 1, T]`` raw logits for breath onset.
            - breath_durations: ``[B, 1, T]`` predicted breath duration (ms).
            - breath_intensity: ``[B, 1, T]`` predicted intensity (0-1).
            - pause_durations: ``[B, 1, T]`` predicted pause duration (ms).
        """
        h = self.input_proj(expanded_features)
        h = self.blocks(h)

        onset_logits = self.onset_head(h)
        breath_durations = F.softplus(self.breath_dur_head(h))
        breath_intensity = torch.sigmoid(self.intensity_head(h))
        pause_durations = F.softplus(self.pause_dur_head(h))

        return onset_logits, breath_durations, breath_intensity, pause_durations


class BreathEventLoss(nn.Module):
    """BPEH losses: L_onset + L_dur + L_amp.

    L_onset: Focal loss on breath onset (handles class imbalance).
    L_dur: L1 on breath and pause durations at event frames.
    L_amp: L1 on breath intensity at onset frames.

    Args:
        lambda_onset: Weight for onset loss.
        lambda_dur: Weight for duration loss.
        lambda_amp: Weight for intensity loss.
        focal_gamma: Focal loss gamma for onset classification.
    """

    def __init__(
        self,
        lambda_onset: float = 0.5,
        lambda_dur: float = 0.3,
        lambda_amp: float = 0.2,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.lambda_onset = lambda_onset
        self.lambda_dur = lambda_dur
        self.lambda_amp = lambda_amp
        self.focal_gamma = focal_gamma

    def forward(
        self,
        pred_onset_logits: torch.Tensor,
        pred_breath_dur: torch.Tensor,
        pred_intensity: torch.Tensor,
        pred_pause_dur: torch.Tensor,
        gt_onset: torch.Tensor,
        gt_breath_dur: torch.Tensor,
        gt_intensity: torch.Tensor,
        gt_pause_dur: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute BPEH losses.

        Args:
            pred_onset_logits: ``[B, 1, T]`` predicted onset logits.
            pred_breath_dur: ``[B, 1, T]`` predicted breath duration (ms).
            pred_intensity: ``[B, 1, T]`` predicted intensity (0-1).
            pred_pause_dur: ``[B, 1, T]`` predicted pause duration (ms).
            gt_onset: ``[B, T]`` ground-truth onset binary mask.
            gt_breath_dur: ``[B, T]`` ground-truth breath duration (ms).
            gt_intensity: ``[B, T]`` ground-truth intensity (0-1).
            gt_pause_dur: ``[B, T]`` ground-truth pause duration (ms).
            frame_mask: ``[B, 1, T]`` valid frame mask.

        Returns:
            Dict with ``event_onset``, ``event_dur``, ``event_amp``, ``event_total``.
        """
        # Reshape predictions to [B, T]
        pred_onset = pred_onset_logits.squeeze(1)  # [B, T]
        pred_bd = pred_breath_dur.squeeze(1)        # [B, T]
        pred_int = pred_intensity.squeeze(1)        # [B, T]
        pred_pd = pred_pause_dur.squeeze(1)         # [B, T]
        mask = frame_mask.squeeze(1)                # [B, T]

        # L_onset: Focal BCE
        bce = F.binary_cross_entropy_with_logits(
            pred_onset, gt_onset, reduction="none",
        )
        pt = torch.where(gt_onset > 0.5, torch.sigmoid(pred_onset), 1.0 - torch.sigmoid(pred_onset))
        focal_weight = (1.0 - pt) ** self.focal_gamma
        l_onset = (focal_weight * bce * mask).sum() / mask.sum().clamp(min=1)

        # L_dur: L1 on breath durations (at onset frames) + pause durations (at pause frames)
        breath_mask = gt_onset * mask  # only at breath onset frames
        pause_mask = (gt_pause_dur > 0).float() * mask  # only at pause frames

        n_breath = breath_mask.sum().clamp(min=1)
        n_pause = pause_mask.sum().clamp(min=1)

        l_breath_dur = (torch.abs(pred_bd - gt_breath_dur) * breath_mask).sum() / n_breath
        l_pause_dur = (torch.abs(pred_pd - gt_pause_dur) * pause_mask).sum() / n_pause
        l_dur = l_breath_dur + l_pause_dur

        # L_amp: L1 on intensity at onset frames
        l_amp = (torch.abs(pred_int - gt_intensity) * breath_mask).sum() / n_breath

        total = self.lambda_onset * l_onset + self.lambda_dur * l_dur + self.lambda_amp * l_amp

        return {
            "event_onset": l_onset,
            "event_dur": l_dur,
            "event_amp": l_amp,
            "event_total": total,
        }
