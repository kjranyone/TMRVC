"""VocoderStudent: iSTFT-based vocoder (MS-Wavehax style)."""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_CONTENT,
    D_VOCODER_FEATURES,
    VOCODER_STATE_FRAMES,
)
from tmrvc_train.modules import CausalConvNeXtBlock

_D_VOCODER_HIDDEN = D_CONTENT  # 256


class VocoderStudent(nn.Module):
    """Lightweight iSTFT-based causal vocoder.

    4 CausalConvNeXt blocks, d=256, k=3, dilation=[1,2,2,2].
    State context: (k-1)*d per block = 2+4+4+4 = 14 frames.
    Input: features[B, 513, T].
    Output: stft_mag[B, 513, T] (non-negative), stft_phase[B, 513, T] (radians).
    """

    def __init__(
        self,
        d_input: int = D_VOCODER_FEATURES,
        d_model: int = _D_VOCODER_HIDDEN,
        d_output: int = D_VOCODER_FEATURES,
        n_blocks: int = 4,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        dilations = dilations or [1, 2, 2, 2]
        assert len(dilations) == n_blocks

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.SiLU(),
        )

        # Causal ConvNeXt backbone
        self.blocks = nn.ModuleList([
            CausalConvNeXtBlock(d_model, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

        # Magnitude head (non-negative via ReLU)
        self.mag_head = nn.Sequential(
            nn.Conv1d(d_model, d_output, kernel_size=1),
            nn.ReLU(),
        )

        # Phase head: predict cos and sin separately, then atan2
        self.phase_cos_head = nn.Conv1d(d_model, d_output, kernel_size=1)
        self.phase_sin_head = nn.Conv1d(d_model, d_output, kernel_size=1)

        # State slicing
        self._state_sizes = [blk.context_size for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == VOCODER_STATE_FRAMES, (
            f"Expected total state {VOCODER_STATE_FRAMES}, got {self._total_state}"
        )

    def forward(
        self,
        features: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            features: ``[B, 513, T]`` STFT features from converter.
            state_in: ``[B, d_model, total_state]`` streaming state or None.

        Returns:
            Tuple of ``(stft_mag [B, 513, T], stft_phase [B, 513, T], state_out or None)``.
        """
        x = self.input_proj(features)  # [B, d_model, T]

        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states = [None] * len(self.blocks)

        new_states = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, s_in)
            new_states.append(s_out)

        # Magnitude (non-negative)
        mag = self.mag_head(x)  # [B, 513, T]

        # Phase via cos/sin parameterization
        cos_part = self.phase_cos_head(x)  # [B, 513, T]
        sin_part = self.phase_sin_head(x)  # [B, 513, T]
        phase = torch.atan2(sin_part, cos_part)  # [B, 513, T], range [-pi, pi]

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return mag, phase, state_out
        return mag, phase, None

    def _split_state(self, state: torch.Tensor) -> list[torch.Tensor]:
        states = []
        offset = 0
        for size in self._state_sizes:
            if size > 0:
                states.append(state[:, :, offset:offset + size])
            else:
                states.append(state[:, :, :0])
            offset += size
        return states

    def init_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.d_model, self._total_state, device=device)
