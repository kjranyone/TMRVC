"""DurationPredictor: predict phoneme durations for TTS.

Converts phoneme-level text features to log-durations (number of frames).
Used for length regulation in UCLM TTS mode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_MODEL  # 512


class DurationPredictor(nn.Module):
    """Predict phoneme durations in log domain.

    Architecture:
        - Input: [B, L, d_model] phoneme features
        - 3x CausalConv1d layers
        - Output: [B, L] log-durations

    Args:
        d_model: Input feature dimension.
        n_layers: Number of conv layers.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        d_model,
                        d_model,
                        kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

        self.proj = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Predict log-durations.

        Args:
            x: [B, L, d_model] phoneme-level features.
            mask: [B, L] padding mask (optional).

        Returns:
            log_durations: [B, L] predicted log-durations.
        """
        # Conv expects [B, d, L]
        x = x.transpose(1, 2)

        for layer in self.layers:
            x_res = layer[0](x)
            x_res = x_res.transpose(1, 2)
            x_res = layer[1](x_res)
            x_res = x_res.transpose(1, 2)
            x_res = layer[2](x_res)
            x_res = layer[3](x_res)
            x = x + x_res

        x = x.transpose(1, 2)
        log_dur = self.proj(x).squeeze(-1)  # [B, L]

        if mask is not None:
            log_dur = log_dur.masked_fill(mask, 0.0)

        return log_dur


def duration_loss(
    log_dur_pred: torch.Tensor,
    dur_target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss in log domain for durations.

    Args:
        log_dur_pred: [B, L] predicted log-durations.
        dur_target: [B, L] ground truth durations (frames).
        mask: [B, L] padding mask.
    """
    log_dur_target = torch.log(dur_target.float() + 1.0)

    if mask is not None:
        loss = F.mse_loss(log_dur_pred, log_dur_target, reduction="none")
        loss = loss.masked_select(~mask).mean()
    else:
        loss = F.mse_loss(log_dur_pred, log_dur_target)

    return loss
