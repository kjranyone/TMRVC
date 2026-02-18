"""IREstimator: causal CNN for impulse response parameter estimation."""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    IR_ESTIMATOR_STATE_FRAMES,
    N_IR_PARAMS,
    N_MELS,
)
from tmrvc_train.modules import CausalConvNeXtBlock

_D_IR_HIDDEN = 128


class IREstimator(nn.Module):
    """Estimate IR parameters from mel chunks.

    3 CausalConvNeXt blocks (d=128, k=3, dilation=[1,1,1]) + AdaptiveAvgPool + MLP head.
    State context: (k-1)*d per block = 2+2+2 = 6 frames.
    Input: mel_chunk[B, 80, N] (N = ir_update_interval = 10).
    Output: ir_params[B, 24] (8 subbands x 3 params).
    """

    def __init__(
        self,
        d_input: int = N_MELS,
        d_model: int = _D_IR_HIDDEN,
        n_ir_params: int = N_IR_PARAMS,
        n_blocks: int = 3,
        kernel_size: int = 3,
        dilations: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        dilations = dilations or [1, 1, 1]
        assert len(dilations) == n_blocks

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv1d(d_input, d_model, kernel_size=1),
            nn.SiLU(),
        )

        # Causal ConvNeXt blocks
        self.blocks = nn.ModuleList([
            CausalConvNeXtBlock(d_model, kernel_size=kernel_size, dilation=d)
            for d in dilations
        ])

        # Temporal pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP head with range-constrained outputs
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.SiLU(),
            nn.Linear(64, n_ir_params),
        )

        # State slicing
        self._state_sizes = [blk.context_size for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == IR_ESTIMATOR_STATE_FRAMES, (
            f"Expected total state {IR_ESTIMATOR_STATE_FRAMES}, got {self._total_state}"
        )

        self.n_ir_params = n_ir_params

    def forward(
        self,
        mel_chunk: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            mel_chunk: ``[B, 80, N]`` accumulated mel frames.
            state_in: ``[B, d_model, total_state]`` streaming state or None.

        Returns:
            Tuple of ``(ir_params [B, 24], state_out or None)``.
        """
        x = self.input_proj(mel_chunk)  # [B, d_model, N]

        if state_in is not None:
            states = self._split_state(state_in)
        else:
            states = [None] * len(self.blocks)

        new_states = []
        for block, s_in in zip(self.blocks, states):
            x, s_out = block(x, s_in)
            new_states.append(s_out)

        # Temporal pooling → [B, d_model, 1] → [B, d_model]
        x = self.pool(x).squeeze(-1)

        # MLP → raw params [B, 24]
        raw = self.mlp(x)

        # Apply range constraints per parameter group
        # RT60 [0-7]: sigmoid * 2.95 + 0.05 → [0.05, 3.0]
        # DRR  [8-15]: sigmoid * 40 - 10 → [-10, 30]
        # Tilt [16-23]: tanh * 6 → [-6, 6]
        rt60 = torch.sigmoid(raw[:, :8]) * 2.95 + 0.05
        drr = torch.sigmoid(raw[:, 8:16]) * 40.0 - 10.0
        tilt = torch.tanh(raw[:, 16:24]) * 6.0

        ir_params = torch.cat([rt60, drr, tilt], dim=-1)  # [B, 24]

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return ir_params, state_out
        return ir_params, None

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
