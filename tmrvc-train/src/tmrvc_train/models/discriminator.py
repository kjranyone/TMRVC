"""MelDiscriminator: mel-domain CNN discriminator for DMD2 distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

from tmrvc_core.constants import N_MELS


class MelDiscriminator(nn.Module):
    """Mel-domain CNN discriminator for distribution matching (~0.5M params).

    4 conv blocks with spectral norm, input [B, 80, T] → logits [B, 1].
    Used in Phase B2 (DMD2) of distillation for GAN-based training.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256, 512]

        layers: list[nn.Module] = []
        in_ch = n_mels
        for out_ch in channels:
            layers.extend([
                spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            spectral_norm(nn.Linear(channels[-1], 1)),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Compute discriminator logits.

        Args:
            mel: ``[B, 80, T]`` mel spectrogram.

        Returns:
            ``[B, 1]`` discriminator logits (not activated).
        """
        h = self.features(mel)
        return self.classifier(h)
