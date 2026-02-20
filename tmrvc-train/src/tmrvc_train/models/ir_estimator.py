"""IREstimator: causal CNN for acoustic condition parameter estimation.

Estimates 32-dim acoustic parameters: 24 IR (environment) + 8 voice source.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_IR_ESTIMATOR_HIDDEN,
    IR_ESTIMATOR_STATE_FRAMES,
    N_ACOUSTIC_PARAMS,
    N_IR_PARAMS,
    N_MELS,
)
from tmrvc_train.modules import CausalConvNeXtBlock


class IREstimator(nn.Module):
    """Estimate acoustic condition parameters from mel chunks.

    3 CausalConvNeXt blocks (d=128, k=3, dilation=[1,1,1]) + AdaptiveAvgPool + MLP head.
    State context: (k-1)*d per block = 2+2+2 = 6 frames.
    Input: mel_chunk[B, 80, N] (N = ir_update_interval = 10).
    Output: acoustic_params[B, 32] (24 IR + 8 voice source).
    """

    def __init__(
        self,
        d_input: int = N_MELS,
        d_model: int = D_IR_ESTIMATOR_HIDDEN,
        n_acoustic_params: int = N_ACOUSTIC_PARAMS,
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
            nn.Linear(64, n_acoustic_params),
        )

        # State slicing
        self._state_sizes = [blk.context_size for blk in self.blocks]
        self._total_state = sum(self._state_sizes)
        assert self._total_state == IR_ESTIMATOR_STATE_FRAMES, (
            f"Expected total state {IR_ESTIMATOR_STATE_FRAMES}, got {self._total_state}"
        )

        self.n_acoustic_params = n_acoustic_params

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
            Tuple of ``(acoustic_params [B, 32], state_out or None)``.
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

        # MLP → raw params [B, 32]
        raw = self.mlp(x)

        # Apply range constraints per parameter group
        # --- IR parameters (0-23) ---
        # RT60 [0-7]: sigmoid * 2.95 + 0.05 → [0.05, 3.0]
        # DRR  [8-15]: sigmoid * 40 - 10 → [-10, 30]
        # Tilt [16-23]: tanh * 6 → [-6, 6]
        rt60 = torch.sigmoid(raw[:, :8]) * 2.95 + 0.05
        drr = torch.sigmoid(raw[:, 8:16]) * 40.0 - 10.0
        tilt = torch.tanh(raw[:, 16:24]) * 6.0

        # --- Voice source parameters (24-31) ---
        # breathiness_low/high [24-25]: sigmoid → [0, 1]
        breathiness = torch.sigmoid(raw[:, 24:26])
        # tension_low/high [26-27]: tanh → [-1, 1]
        tension = torch.tanh(raw[:, 26:28])
        # jitter [28]: sigmoid * 0.1 → [0, 0.1]
        jitter = torch.sigmoid(raw[:, 28:29]) * 0.1
        # shimmer [29]: sigmoid * 0.1 → [0, 0.1]
        shimmer = torch.sigmoid(raw[:, 29:30]) * 0.1
        # formant_shift [30]: tanh → [-1, 1]
        formant_shift = torch.tanh(raw[:, 30:31])
        # roughness [31]: sigmoid → [0, 1]
        roughness = torch.sigmoid(raw[:, 31:32])

        acoustic_params = torch.cat(
            [rt60, drr, tilt, breathiness, tension, jitter, shimmer, formant_shift, roughness],
            dim=-1,
        )  # [B, 32]

        if state_in is not None:
            state_out = torch.cat(new_states, dim=-1)
            return acoustic_params, state_out
        return acoustic_params, None

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
