import torch
import torch.nn as nn


class VoiceStateFiLM(nn.Module):
    """Feature-wise Linear Modulation with voice_state.

    voice_state [B, 8] -> gamma, beta [B, d_model]
    y = gamma * x + beta

    Used in decoder to modulate features based on voice state parameters.
    """

    def __init__(self, d_voice_state: int = 8, d_model: int = 512):
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
        if voice_state.dim() == 2:
            voice_state = voice_state.unsqueeze(1)

        gamma_beta = self.proj(voice_state)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        if x.dim() == 3 and x.shape[1] != voice_state.shape[1]:
            if x.shape[1] == self.d_model:
                x = x.transpose(1, 2)
                gamma = gamma.transpose(1, 2)
                beta = beta.transpose(1, 2)
                result = gamma * x + beta
                return result.transpose(1, 2)
            else:
                gamma = gamma.expand(-1, x.shape[1], -1)
                beta = beta.expand(-1, x.shape[1], -1)
                return gamma * x + beta

        return gamma * x + beta


class MultiVoiceStateFiLM(nn.Module):
    """Multiple FiLM layers for different decoder stages."""

    def __init__(self, d_voice_state: int = 8, d_model: int = 512, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList(
            [VoiceStateFiLM(d_voice_state, d_model) for _ in range(n_layers)]
        )

    def forward(
        self, x: torch.Tensor, voice_state: torch.Tensor, layer_idx: int = 0
    ) -> torch.Tensor:
        """Apply FiLM for specific layer."""
        return self.layers[layer_idx](x, voice_state)
