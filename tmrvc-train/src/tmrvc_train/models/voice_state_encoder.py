"""VoiceStateEncoder: encode continuous voice state parameters.

Converts frame-level voice state parameters (breathiness, tension, etc.)
to continuous features that can be used for conditioning the CodecLM.

Voice state dimensions (8):
    0: breathiness [0, 1]
    1: tension [0, 1]
    2: arousal [0, 1]
    3: valence [-1, 1]
    4: roughness [0, 1]
    5: voicing [0, 1]
    6: energy [0, 1]
    7: rate [0.5, 2.0]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import D_MODEL  # 256


class VoiceStateEncoder(nn.Module):
    """Encode voice state parameters to continuous features.

    Architecture::

        voice_state[B, T, 8] → Linear(8, d_model)
            → CausalConv1d layers
            → state_features[B, T, d_model]

    Args:
        d_state: Input voice state dimension (default: 8).
        d_model: Output feature dimension (default: 256).
        n_layers: Number of conv layers.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_state: int = 8,
        d_model: int = D_MODEL,
        n_layers: int = 4,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_state = d_state
        self.d_model = d_model

        # Project to model dimension
        self.input_proj = nn.Linear(d_state, d_model)

        # Causal conv layers
        self.conv_layers = nn.ModuleList()
        for i in range(n_layers):
            self.conv_layers.append(
                nn.Sequential(
                    CausalConv1d(d_model, d_model, kernel_size=kernel_size),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )

    def forward(self, voice_state: torch.Tensor) -> torch.Tensor:
        """Encode voice state parameters.

        Args:
            voice_state: ``[B, T, d_state]`` voice state tensor.

        Returns:
            ``[B, T, d_model]`` encoded state features.
        """
        # Project to model dimension
        x = self.input_proj(voice_state)  # [B, T, d_model]

        # Apply causal conv layers
        for conv_layer in self.conv_layers:
            # Conv expects [B, d, T]
            x_t = x.transpose(1, 2)
            x_t = conv_layer[0](x_t)  # CausalConv1d
            x_t = x_t.transpose(1, 2)  # [B, T, d]
            x_t = conv_layer[1](x_t)  # LayerNorm
            x_t = conv_layer[2](x_t)  # GELU
            x_t = conv_layer[3](x_t)  # Dropout
            x = x + x_t  # Residual

        return x


class CausalConv1d(nn.Module):
    """Causal 1D convolution with left padding.

    Ensures that output at time t only depends on inputs at times <= t.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal convolution.

        Args:
            x: ``[B, C, T]`` input tensor.

        Returns:
            ``[B, C, T]`` output tensor (same length as input).
        """
        # Left pad for causality
        x = nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)
