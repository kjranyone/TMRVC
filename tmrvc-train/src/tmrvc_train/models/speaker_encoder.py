"""SpeakerEncoderWithLoRA: ECAPA-TDNN backbone + spk_embed head + lora_delta head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import (
    D_SPEAKER,
    LORA_DELTA_SIZE,
)


class SpeakerEncoderWithLoRA(nn.Module):
    """Speaker encoder that produces both a speaker embedding and LoRA delta.

    Uses a simple backbone (trainable) that processes mel spectrograms.
    For pre-trained ECAPA-TDNN integration, use ``from_pretrained()``.

    Outputs:
    - spk_embed[B, 192]: L2-normalised speaker embedding.
    - lora_delta[B, LORA_DELTA_SIZE]: LoRA weight deltas for converter FiLM layers.
    """

    # ECAPA-TDNN intermediate dim (from SpeechBrain's attentive stat pooling)
    _BACKBONE_DIM = 1536

    def __init__(
        self,
        backbone_dim: int = _BACKBONE_DIM,
        d_speaker: int = D_SPEAKER,
        lora_delta_size: int = LORA_DELTA_SIZE,
    ) -> None:
        super().__init__()
        self.backbone_dim = backbone_dim

        # Simple backbone for standalone training
        # (replaced by ECAPA-TDNN in from_pretrained)
        self.backbone = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.backbone_proj = nn.Linear(512, backbone_dim)

        # Speaker embedding head
        self.spk_embed_head = nn.Linear(backbone_dim, d_speaker)

        # LoRA delta head
        self.lora_delta_head = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.SiLU(),
            nn.Linear(512, lora_delta_size),
        )

    def forward(
        self, mel_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            mel_ref: ``[B, 80, T_ref]`` reference mel spectrogram.

        Returns:
            Tuple of (spk_embed [B, 192], lora_delta [B, 24576]).
        """
        # Backbone features
        h = self.backbone(mel_ref)  # [B, 512, 1]
        h = h.squeeze(-1)  # [B, 512]
        h = self.backbone_proj(h)  # [B, backbone_dim]

        # Speaker embedding (L2 normalized)
        spk_embed = self.spk_embed_head(h)  # [B, d_speaker]
        spk_embed = F.normalize(spk_embed, p=2, dim=-1)

        # LoRA delta
        lora_delta = self.lora_delta_head(h)  # [B, lora_delta_size]

        return spk_embed, lora_delta
