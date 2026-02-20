"""ContentEncoderStudent: causal CNN distilled from ContentVec/WavLM with optional VQ."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    CONTENT_ENCODER_STATE_FRAMES,
    D_CONTENT,
    N_MELS,
)
from tmrvc_train.modules import CausalConvNeXtBlock

if TYPE_CHECKING:
    from tmrvc_train.models.vq_bottleneck import FactorizedVQBottleneck


class ContentEncoderStudent(nn.Module):
    """Lightweight causal content encoder.

    6 CausalConvNeXt blocks, d=256, k=3, dilation=[1,1,2,2,4,4].
    State context: (k-1)*d per block = 2+2+4+4+8+8 = 28 frames.
    Input: mel[B, 80, T] + f0[B, 1, T] concatenated → [B, 81, T].
    Output: content[B, 256, T].

    Optionally includes a VQ bottleneck for speaker leakage reduction.
    """

    def __init__(
        self,
        d_model: int = D_CONTENT,
        n_blocks: int = 6,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
        use_vq: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_vq = use_vq
        dilations = dilations or [1, 1, 2, 2, 4, 4]
        assert len(dilations) == n_blocks

        self.input_proj = nn.Sequential(
            nn.Conv1d(N_MELS + 1, d_model, kernel_size=1),
            nn.SiLU(),
        )

        self.blocks = nn.ModuleList(
            [
                CausalConvNeXtBlock(d_model, kernel_size=kernel_size, dilation=d)
                for d in dilations
            ]
        )

        self.output_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        self._state_sizes: list[int] = []
        total_state = 0
        for blk in self.blocks:
            sz = blk.context_size
            self._state_sizes.append(sz)
            total_state += sz
        self._total_state = total_state
        assert self._total_state == CONTENT_ENCODER_STATE_FRAMES, (
            f"Expected total state {CONTENT_ENCODER_STATE_FRAMES}, got {self._total_state}"
        )

        self.vq: FactorizedVQBottleneck | None = None
        if use_vq:
            from tmrvc_train.models.vq_bottleneck import FactorizedVQBottleneck

            self.vq = FactorizedVQBottleneck(d_input=d_model)

        # Pre-initialize so ONNX export doesn't warn about dynamic attribute
        self.register_buffer("_vq_commitment_loss", torch.tensor(0.0), persistent=False)

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
            If VQ is enabled, also tracks VQ indices for EMA updates.
        """
        x = torch.cat([mel, f0], dim=1)
        x = self.input_proj(x)

        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states: list[torch.Tensor | None] = [None] * len(self.blocks)

        new_states: list[torch.Tensor] = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, s_in)
            if s_out is not None:
                new_states.append(s_out)

        x = self.output_proj(x)

        if self.vq is not None:
            if self.training:
                self._vq_input = x  # Save pre-VQ input for EMA update
            x, vq_indices, commitment_loss = self.vq(x)
            self._vq_commitment_loss = commitment_loss
            if self.training:
                self._vq_indices = vq_indices
        else:
            self._vq_commitment_loss = torch.tensor(0.0, device=x.device)

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return x, state_out
        return x, None

    def _split_state(self, state: torch.Tensor) -> list[torch.Tensor]:
        """Split concatenated state into per-block states."""
        states: list[torch.Tensor] = []
        offset = 0
        for size in self._state_sizes:
            if size > 0:
                states.append(state[:, :, offset : offset + size])
            else:
                states.append(state[:, :, :0])
            offset += size
        return states

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Create zero-initialized streaming state."""
        return torch.zeros(batch_size, self.d_model, self._total_state, device=device)

    def update_vq_ema(self) -> None:
        """Update VQ codebook EMA after forward pass."""
        if self.vq is not None and hasattr(self, "_vq_input"):
            self.vq.update_ema(self._vq_input, self._vq_indices)
            del self._vq_input
            del self._vq_indices
