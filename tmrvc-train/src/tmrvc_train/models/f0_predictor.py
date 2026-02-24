"""F0Predictor: style-conditioned fundamental frequency prediction.

Predicts frame-level F0 and voiced probability from length-regulated
text features, with FiLM style conditioning for emotion/prosody control.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import D_F0_PREDICTOR, D_STYLE, D_TEXT_ENCODER

from tmrvc_train.modules import CausalConvNeXtBlock, FiLMConditioner


def length_regulate(
    text_features: torch.Tensor,
    durations: torch.Tensor,
) -> torch.Tensor:
    """Expand text features according to phoneme durations.

    Args:
        text_features: ``[B, d, L]`` phoneme-level features.
        durations: ``[B, L]`` integer frame counts per phoneme.

    Returns:
        ``[B, d, T]`` frame-level features where T = sum(durations).
    """
    B, d, L = text_features.shape
    # Round durations to integers
    dur_int = durations.long()

    # Build expanded features per batch
    expanded = []
    for b in range(B):
        frames = []
        for l in range(L):
            n = dur_int[b, l].item()
            if n > 0:
                frames.append(text_features[b, :, l:l + 1].expand(-1, n))
        if frames:
            expanded.append(torch.cat(frames, dim=-1))  # [d, T_b]
        else:
            expanded.append(text_features.new_zeros(d, 1))

    # Pad to max T
    T_max = max(e.shape[-1] for e in expanded)
    padded = torch.zeros(B, d, T_max, device=text_features.device, dtype=text_features.dtype)
    for b, e in enumerate(expanded):
        padded[b, :, :e.shape[-1]] = e

    return padded


class F0Predictor(nn.Module):
    """Predict F0 and voiced probability from expanded text features.

    Architecture::

        text_features[B, d_text, T] → Conv1d(d_text, d_hidden, 1)
            → CausalConvNeXtBlock x 4 (dilation=[1,1,2,2]) + FiLM(style)
            → Conv1d(d_hidden, 2, 1)
            → f0[B,1,T], voiced_prob[B,1,T]

    Args:
        d_input: Input feature dimension (default: D_TEXT_ENCODER=256).
        d_hidden: Hidden dimension (default: D_F0_PREDICTOR=128).
        d_style: Style conditioning dimension (default: D_STYLE=32).
        n_blocks: Number of ConvNeXt blocks.
        kernel_size: Convolution kernel size.
        dilations: Per-block dilation factors.
    """

    def __init__(
        self,
        d_input: int = D_TEXT_ENCODER,
        d_hidden: int = D_F0_PREDICTOR,
        d_style: int = D_STYLE,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        dilations = dilations or [1, 1, 2, 2]
        assert len(dilations) == n_blocks

        self.input_proj = nn.Conv1d(d_input, d_hidden, kernel_size=1)

        self.blocks = nn.ModuleList([
            CausalConvNeXtBlock(d_hidden, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])
        self.films = nn.ModuleList([
            FiLMConditioner(d_style, d_hidden)
            for _ in range(n_blocks)
        ])

        # Output: f0 (Hz, positive) + voiced_prob (sigmoid)
        self.output_proj = nn.Conv1d(d_hidden, 2, kernel_size=1)

    def forward(
        self,
        text_features: torch.Tensor,
        style: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict F0 and voiced probability.

        Args:
            text_features: ``[B, d_input, T]`` length-regulated text features.
            style: ``[B, d_style]`` style conditioning (optional).

        Returns:
            Tuple of:
            - f0: ``[B, 1, T]`` predicted F0 in Hz (softplus → positive).
            - voiced_prob: ``[B, 1, T]`` voiced probability (sigmoid → [0,1]).
        """
        x = self.input_proj(text_features)  # [B, d_hidden, T]

        for block, film in zip(self.blocks, self.films):
            x, _ = block(x)  # Training mode, no state
            if style is not None:
                x = film(x, style)

        out = self.output_proj(x)  # [B, 2, T]
        f0_raw = out[:, 0:1, :]  # [B, 1, T]
        voiced_raw = out[:, 1:2, :]  # [B, 1, T]

        f0 = nn.functional.softplus(f0_raw)  # Positive Hz
        voiced_prob = torch.sigmoid(voiced_raw)  # [0, 1]

        return f0, voiced_prob
