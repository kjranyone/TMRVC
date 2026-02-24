"""DurationPredictor: FastSpeech2-style explicit duration prediction.

Predicts per-phoneme duration (in frames) from text features with
optional style conditioning via FiLM.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import D_STYLE, D_TEXT_ENCODER

from tmrvc_train.modules import FiLMConditioner


class _ConvBlock(nn.Module):
    """Conv1d + ReLU + Dropout (no LayerNorm on [B,C,T])."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.conv(x)))


class DurationPredictor(nn.Module):
    """Predict phoneme durations from text encoder output.

    Architecture::

        text_features[B, d, L] → Conv1d(d,d,k=3) + ReLU
            → Conv1d(d,d,k=3) + ReLU + FiLM(style)
            → Linear(d, 1) + Softplus
            → durations[B, L]  (positive frame counts)

    Args:
        d_input: Input feature dimension (default: D_TEXT_ENCODER=256).
        d_hidden: Hidden dimension (default: 256).
        d_style: Style conditioning dimension (default: D_STYLE=32).
        kernel_size: Conv kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_input: int = D_TEXT_ENCODER,
        d_hidden: int = 256,
        d_style: int = D_STYLE,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.conv1 = _ConvBlock(d_input, d_hidden, kernel_size, dropout)
        self.conv2 = _ConvBlock(d_hidden, d_hidden, kernel_size, dropout)

        self.film = FiLMConditioner(d_style, d_hidden)
        self.output_proj = nn.Linear(d_hidden, 1)
        self.softplus = nn.Softplus()

    def forward(
        self,
        text_features: torch.Tensor,
        style: torch.Tensor | None = None,
        phoneme_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict durations.

        Args:
            text_features: ``[B, d, L]`` text encoder output.
            style: ``[B, d_style]`` style conditioning (optional).
            phoneme_lengths: ``[B]`` for masking (unused in forward, used in loss).

        Returns:
            ``[B, L]`` predicted durations in frames (positive values).
        """
        x = self.conv1(text_features)  # [B, d_hidden, L]
        x = self.conv2(x)  # [B, d_hidden, L]

        if style is not None:
            x = self.film(x, style)

        # Project to scalar duration
        x = x.transpose(1, 2)  # [B, L, d_hidden]
        x = self.output_proj(x).squeeze(-1)  # [B, L]
        x = self.softplus(x)  # Ensure positive

        return x
