"""ContentEncoderStudent: causal CNN distilled from ContentVec/WavLM."""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    CONTENT_ENCODER_STATE_FRAMES,
    D_CONTENT,
    N_MELS,
)
from tmrvc_train.modules import CausalConvNeXtBlock


class ContentEncoderStudent(nn.Module):
    """Lightweight causal content encoder.

    6 CausalConvNeXt blocks, d=256, k=3, dilation=[1,1,2,2,4,4].
    State context: (k-1)*d per block = 2+2+4+4+8+8 = 28 frames.
    Input: mel[B, 80, T] + f0[B, 1, T] concatenated → [B, 81, T].
    Output: content[B, 256, T].
    """

    def __init__(
        self,
        d_model: int = D_CONTENT,
        n_blocks: int = 6,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        dilations = dilations or [1, 1, 2, 2, 4, 4]
        assert len(dilations) == n_blocks

        # Input projection: mel(80) + f0(1) = 81 → d_model
        self.input_proj = nn.Sequential(
            nn.Conv1d(N_MELS + 1, d_model, kernel_size=1),
            nn.SiLU(),
        )

        # Causal ConvNeXt blocks
        self.blocks = nn.ModuleList([
            CausalConvNeXtBlock(d_model, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

        # Output projection
        self.output_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Precompute state slicing info
        self._state_sizes = [blk.context_size for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == CONTENT_ENCODER_STATE_FRAMES, (
            f"Expected total state {CONTENT_ENCODER_STATE_FRAMES}, got {self._total_state}"
        )

    def forward(
        self,
        mel: torch.Tensor,
        f0: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            mel: ``[B, 80, T]`` log-mel spectrogram.
            f0: ``[B, 1, T]`` log-F0.
            state_in: ``[B, d_model, total_state]`` streaming state or None.

        Returns:
            Tuple of ``(content [B, 256, T], state_out or None)``.
        """
        x = torch.cat([mel, f0], dim=1)  # [B, 81, T]
        x = self.input_proj(x)  # [B, d_model, T]

        # Split state for each block
        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states = [None] * len(self.blocks)

        new_states = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, s_in)
            new_states.append(s_out)

        x = self.output_proj(x)

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return x, state_out
        return x, None

    def _split_state(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Split concatenated state into per-block states."""
        states = []
        offset = 0
        for size in self._state_sizes:
            if size > 0:
                states.append(state[:, :, offset:offset + size])
            else:
                states.append(state[:, :, :0])  # empty tensor
            offset += size
        return states

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Create zero-initialized streaming state."""
        return torch.zeros(batch_size, self.d_model, self._total_state, device=device)
