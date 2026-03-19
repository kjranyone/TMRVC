import torch
import torch.nn as nn

from tmrvc_core.constants import D_MODEL


class VoiceStateFiLM(nn.Module):
    """Feature-wise Linear Modulation with voice_state.

    voice_state [B, 12] -> gamma, beta [B, d_model]
    y = gamma * x + beta

    Used in decoder to modulate features based on voice state parameters.
    """

    def __init__(self, d_voice_state: int = 12, d_model: int = D_MODEL):
        super().__init__()
        self.d_voice_state = d_voice_state
        self.d_model = d_model
        self.proj = nn.Linear(d_voice_state, d_model * 2)

    def forward(self, x: torch.Tensor, voice_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, d_model, T] or [B, T, d_model] feature tensor
            voice_state: [B, d_voice_state] or [B, T, d_voice_state]

        Returns:
            Modulated tensor with same shape as x
        """
        x_is_transposed = x.dim() == 3 and x.shape[1] == self.d_model

        if x_is_transposed:
            x = x.transpose(1, 2)

        if voice_state.dim() == 2:
            gamma_beta = self.proj(voice_state)
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1).expand(-1, x.shape[1], -1)
            beta = beta.unsqueeze(1).expand(-1, x.shape[1], -1)
        else:
            gamma_beta = self.proj(voice_state)
            gamma, beta = gamma_beta.chunk(2, dim=-1)

        result = gamma * x + beta

        if x_is_transposed:
            result = result.transpose(1, 2)

        return result


class MultiVoiceStateFiLM(nn.Module):
    """Multiple FiLM layers for different decoder stages."""

    def __init__(self, d_voice_state: int = 12, d_model: int = D_MODEL, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [VoiceStateFiLM(d_voice_state, d_model) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, voice_state: torch.Tensor, layer_idx: int = 0
    ) -> torch.Tensor:
        """Apply FiLM for specific layer."""
        return self.layers[layer_idx](x, voice_state)
