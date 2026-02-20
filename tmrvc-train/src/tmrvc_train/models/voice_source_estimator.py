"""Voice Source parameter estimator for external distillation.

This module provides a lightweight CNN that estimates voice source parameters
(breathiness, tension, jitter, shimmer, formant_shift, roughness) from audio.
The estimator is trained separately and then frozen during TMRVC Phase 2
to provide ground-truth labels for voice source distillation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import (
    N_MELS,
    N_VOICE_SOURCE_PARAMS,
)

logger = logging.getLogger(__name__)


@dataclass
class VoiceSourceConfig:
    """Configuration for VoiceSourceEstimator."""

    n_mels: int = N_MELS
    hidden_dim: int = 128
    n_conv_layers: int = 4
    kernel_size: int = 3
    n_voice_params: int = N_VOICE_SOURCE_PARAMS
    dropout: float = 0.1


class VoiceSourceEstimator(nn.Module):
    """Lightweight CNN for voice source parameter estimation.

    Takes mel spectrogram as input and outputs 8 voice source parameters:
    - breathiness_low (0-1)
    - breathiness_high (0-1)
    - tension_low (-1 to 1)
    - tension_high (-1 to 1)
    - jitter (0-0.1)
    - shimmer (0-0.1)
    - formant_shift (-1 to 1)
    - roughness (0-1)

    Architecture:
        - 4 Conv1d layers with LayerNorm and GELU
        - Global average pooling
        - 2 FC layers with dropout
        - Output projection with parameter-specific activations

    Args:
        config: VoiceSourceConfig with model hyperparameters.
    """

    def __init__(self, config: VoiceSourceConfig | None = None) -> None:
        super().__init__()
        self.config = config or VoiceSourceConfig()
        cfg = self.config

        layers = []
        in_ch = cfg.n_mels
        for i in range(cfg.n_conv_layers):
            out_ch = cfg.hidden_dim * (2 ** min(i, 2))
            layers.extend(
                [
                    nn.Conv1d(
                        in_ch,
                        out_ch,
                        kernel_size=cfg.kernel_size,
                        padding=cfg.kernel_size // 2,
                    ),
                    nn.LayerNorm([out_ch]),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_ch, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.GELU(),
        )

        self.output_proj = nn.Linear(cfg.hidden_dim // 2, cfg.n_voice_params)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Estimate voice source parameters from mel spectrogram.

        Args:
            mel: Log-mel spectrogram [B, n_mels, T].

        Returns:
            Voice source parameters [B, 8] with the following layout:
            - [0]: breathiness_low (sigmoid)
            - [1]: breathiness_high (sigmoid)
            - [2]: tension_low (tanh)
            - [3]: tension_high (tanh)
            - [4]: jitter (sigmoid * 0.1)
            - [5]: shimmer (sigmoid * 0.1)
            - [6]: formant_shift (tanh)
            - [7]: roughness (sigmoid)
        """
        x = self.conv(mel)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        x = self.output_proj(x)
        x = self._apply_activations(x)
        return x

    def _apply_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-specific activations to enforce valid ranges."""
        out = x.clone()

        out[:, 0] = torch.sigmoid(x[:, 0])
        out[:, 1] = torch.sigmoid(x[:, 1])
        out[:, 2] = torch.tanh(x[:, 2])
        out[:, 3] = torch.tanh(x[:, 3])
        out[:, 4] = torch.sigmoid(x[:, 4]) * 0.1
        out[:, 5] = torch.sigmoid(x[:, 5]) * 0.1
        out[:, 6] = torch.tanh(x[:, 6])
        out[:, 7] = torch.sigmoid(x[:, 7])

        return out

    @classmethod
    def from_pretrained(
        cls, checkpoint_path: str, device: str = "cpu"
    ) -> "VoiceSourceEstimator":
        """Load a pretrained estimator from checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
            device: Device to load the model to.

        Returns:
            Loaded and frozen VoiceSourceEstimator.
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        config = VoiceSourceConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        logger.info(f"Loaded pretrained VoiceSourceEstimator from {checkpoint_path}")
        return model


class VoiceSourceDistillationLoss(nn.Module):
    """Loss function for voice source distillation from external estimator.

    Computes MSE between the IREstimator's voice source predictions
    and the pretrained VoiceSourceEstimator's predictions.

    Args:
        estimator: Pretrained VoiceSourceEstimator (frozen).
        lambda_voice: Weight for the distillation loss (default: 0.2).
    """

    def __init__(
        self,
        estimator: VoiceSourceEstimator,
        lambda_voice: float = 0.2,
    ) -> None:
        super().__init__()
        self.estimator = estimator
        self.lambda_voice = lambda_voice

        for param in self.estimator.parameters():
            param.requires_grad = False
        self.estimator.eval()

    def forward(
        self,
        mel: torch.Tensor,
        voice_source_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distillation loss.

        Args:
            mel: Mel spectrogram [B, n_mels, T] for ground truth estimation.
            voice_source_pred: Predicted voice source params [B, 8] from IREstimator.

        Returns:
            Weighted MSE loss scalar.
        """
        with torch.no_grad():
            voice_source_gt = self.estimator(mel)

        loss = F.mse_loss(voice_source_pred, voice_source_gt)
        return self.lambda_voice * loss


def create_voice_source_teacher(
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> VoiceSourceEstimator | None:
    """Factory function to create a voice source teacher for distillation.

    If no checkpoint is provided, returns None (will use zero regularization).

    Args:
        checkpoint_path: Optional path to pretrained checkpoint.
        device: Device to load the model to.

    Returns:
        VoiceSourceEstimator or None.
    """
    if checkpoint_path is None:
        logger.info("No voice source checkpoint provided, using zero regularization")
        return None

    try:
        return VoiceSourceEstimator.from_pretrained(checkpoint_path, device)
    except (FileNotFoundError, KeyError, RuntimeError) as e:
        logger.warning(f"Failed to load voice source estimator: {e}")
        return None
