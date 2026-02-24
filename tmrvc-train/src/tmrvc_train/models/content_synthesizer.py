"""ContentSynthesizer: text-to-content bridge for TTS.

Converts length-regulated text features to content features in the same
256-dimensional space as ContentEncoderStudent output, enabling the
Converter and Vocoder to be shared between VC and TTS pipelines.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import D_CONTENT, D_CONTENT_SYNTHESIZER, D_TEXT_ENCODER

from tmrvc_train.modules import CausalConvNeXtBlock


class ContentSynthesizer(nn.Module):
    """Synthesize content features from text features.

    Transforms length-regulated text encoder output into the same
    distribution as ContentEncoderStudent output (256d), so the
    downstream Converter receives compatible input regardless of
    VC or TTS mode.

    Architecture::

        text_features[B, d_text, T] → Conv1d(d_text, d_hidden, 1) + SiLU
            → CausalConvNeXtBlock x 4 (dilation=[1,1,2,4])
            → Conv1d(d_hidden, d_content, 1)
            → content[B, d_content, T]

    Training target: ``L_content = MSE(synthesized, content_encoder(mel, f0))``

    Args:
        d_input: Input feature dimension (default: D_TEXT_ENCODER=256).
        d_hidden: Hidden dimension (default: D_CONTENT_SYNTHESIZER=256).
        d_output: Output content dimension (default: D_CONTENT=256).
        n_blocks: Number of ConvNeXt blocks.
        kernel_size: Convolution kernel size.
        dilations: Per-block dilation factors.
    """

    def __init__(
        self,
        d_input: int = D_TEXT_ENCODER,
        d_hidden: int = D_CONTENT_SYNTHESIZER,
        d_output: int = D_CONTENT,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        dilations = dilations or [1, 1, 2, 4]
        assert len(dilations) == n_blocks

        self.input_proj = nn.Sequential(
            nn.Conv1d(d_input, d_hidden, kernel_size=1),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList([
            CausalConvNeXtBlock(d_hidden, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

        self.output_proj = nn.Conv1d(d_hidden, d_output, kernel_size=1)

    def forward(
        self,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Synthesize content features from text features.

        Args:
            text_features: ``[B, d_input, T]`` length-regulated text features.

        Returns:
            ``[B, d_content, T]`` synthesized content features.
        """
        x = self.input_proj(text_features)  # [B, d_hidden, T]

        for block in self.blocks:
            x, _ = block(x)  # Training mode, no state

        return self.output_proj(x)  # [B, d_content, T]
