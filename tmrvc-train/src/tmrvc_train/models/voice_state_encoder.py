"""Voice State Encoder for UCLM conditioning.

Combines 12-dim explicit parameters, 128-dim SSL features, and 12-dim delta state
into a single 512-dim state condition for the UCLM core.

Design reference: docs/design/onnx-contract.md Section 3.3
"""

import torch
import torch.nn as nn
from typing import Optional

from tmrvc_core.constants import D_MODEL

from .ssl_extractor import SSLProjection


class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial disentanglement.

    Forward: identity
    Backward: -alpha * gradient
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Module wrapper for Gradient Reversal Layer."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalLayer.apply(x, self.alpha)


class VoiceStateEncoder(nn.Module):
    """Encodes voice state for UCLM conditioning.

    Inputs:
        - explicit_state: [B, T, 12] or [B, 12] — manual/heuristic parameters
        - ssl_state: [B, T, 128] or [B, 128] — WavLM latent style
        - delta_state: [B, T, 12] or [B, 12] — voice_state_t - voice_state_{t-1}

    Output:
        - state_cond: [B, T, d_model] or [B, d_model] — fused condition

    Architecture:
        - explicit_proj: Linear(12, d_model // 3)
        - ssl_proj: Linear(128, d_model // 3)
        - delta_proj: Linear(12, d_model // 3)
        - fusion: Linear(d_model, d_model)
        - temporal_conv: CausalConv1d for temporal smoothing
        - GRL adversarial classifier for disentanglement
    """

    def __init__(
        self,
        d_voice_state_explicit: int = 12,
        d_voice_state_ssl: int = 128,
        d_voice_state_delta: int = 12,
        d_model: int = D_MODEL,
        num_speakers: int = 0,
        num_phonemes: int = 0,
        use_grl: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.use_grl = use_grl

        third = d_model // 3

        self.explicit_proj = nn.Sequential(
            nn.Linear(d_voice_state_explicit, third),
            nn.GELU(),
        )

        self.ssl_proj = nn.Sequential(
            nn.Linear(d_voice_state_ssl, third),
            nn.GELU(),
        )

        self.delta_proj = nn.Sequential(
            nn.Linear(d_voice_state_delta, third),
            nn.GELU(),
        )

        self.fusion = nn.Sequential(
            nn.Linear(third * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Causal temporal conv: left-pad only (no lookahead into future frames)
        self.temporal_conv = nn.Sequential(
            nn.ConstantPad1d((4, 0), 0.0),  # left-pad 4 for kernel_size=5
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=0),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(d_model)

        if use_grl and (num_speakers > 0 or num_phonemes > 0):
            self.adversarial_classifier = nn.Sequential(
                GradientReversal(alpha=1.0),
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_speakers + num_phonemes),
            )
        else:
            self.adversarial_classifier = None

    def forward(
        self,
        explicit_state: torch.Tensor,
        ssl_state: Optional[torch.Tensor] = None,
        delta_state: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward pass.

        Args:
            explicit_state: [B, T, 12] or [B, 12]
            ssl_state: [B, T, 128] or [B, 128], optional
            delta_state: [B, T, 12] or [B, 12], optional

        Returns:
            state_cond: [B, T, d_model] or [B, d_model]
            adv_logits: [B, num_classes] or None (if no GRL)
        """
        B = explicit_state.shape[0]
        T = explicit_state.shape[1] if explicit_state.ndim == 3 else None
        device = explicit_state.device

        # fallbacks for missing signals (Worker 02 / Worker 04)
        if ssl_state is None:
            shape = (B, T, 128) if T is not None else (B, 128)
            ssl_state = torch.zeros(shape, device=device)
        if delta_state is None:
            shape = (B, T, 12) if T is not None else (B, 12)
            delta_state = torch.zeros(shape, device=device)

        x_exp = self.explicit_proj(explicit_state)
        x_ssl = self.ssl_proj(ssl_state)
        x_delta = self.delta_proj(delta_state)

        x = torch.cat([x_exp, x_ssl, x_delta], dim=-1)
        x = self.fusion(x)

        if x.dim() == 3:
            x = x.transpose(1, 2)
            x = self.temporal_conv(x)
            x = x.transpose(1, 2)
        else:
            x = x.unsqueeze(-1)  # [B, 512] -> [B, 512, 1]
            x = self.temporal_conv(x)
            x = x.squeeze(-1)  # [B, 512, 1] -> [B, 512]

        state_cond = self.norm(x)

        adv_logits = None
        if self.adversarial_classifier is not None:
            adv_logits = self.adversarial_classifier(state_cond)

        return state_cond, adv_logits


class VoiceStateEncoderForStreaming(nn.Module):
    """Optimized VoiceStateEncoder for streaming inference.

    Simplified version without GRL for real-time use.
    Outputs single state_cond per frame.
    """

    def __init__(
        self,
        d_voice_state_explicit: int = 12,
        d_voice_state_ssl: int = 128,
        d_voice_state_delta: int = 12,
        d_model: int = D_MODEL,
    ):
        super().__init__()

        self.explicit_proj = nn.Linear(d_voice_state_explicit, d_model // 4)
        self.ssl_proj = nn.Linear(d_voice_state_ssl, d_model // 2)
        self.delta_proj = nn.Linear(d_voice_state_delta, d_model // 4)

        self.fusion = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        delta_state: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for single frame.

        Args:
            explicit_state: [B, 12]
            ssl_state: [B, 128]
            delta_state: [B, 12]

        Returns:
            state_cond: [B, d_model]
        """
        x_exp = self.explicit_proj(explicit_state)
        x_ssl = self.ssl_proj(ssl_state)
        x_delta = self.delta_proj(delta_state)

        x = torch.cat([x_exp, x_ssl, x_delta], dim=-1)
        x = self.fusion(x)

        return self.norm(x)


def create_voice_state_encoder(
    d_model: int = D_MODEL,
    for_streaming: bool = False,
    use_grl: bool = True,
    num_speakers: int = 0,
    num_phonemes: int = 0,
) -> nn.Module:
    """Factory function to create voice state encoder.

    Args:
        d_model: Output dimension
        for_streaming: Use streaming-optimized version
        use_grl: Use Gradient Reversal Layer (training only)
        num_speakers: Number of speakers for adversarial loss
        num_phonemes: Number of phonemes for adversarial loss

    Returns:
        VoiceStateEncoder module
    """
    if for_streaming:
        return VoiceStateEncoderForStreaming(d_model=d_model)

    return VoiceStateEncoder(
        d_model=d_model,
        use_grl=use_grl,
        num_speakers=num_speakers,
        num_phonemes=num_phonemes,
    )
