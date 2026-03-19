"""Acting texture latent encoder and predictor for v4.

The acting latent captures non-physical residual acting qualities that cannot
be represented by the 12-D explicit physical controls alone.

Architecture:
- ActingLatentEncoder: Extracts acting latent from reference audio features
- ActingLatentPredictor: Predicts acting latent prior from text + context
- ActingMacroProjector: Maps 6 user-facing macro controls to 24-D latent space

Rules:
- Latent must be separate from physical controls (different tensor)
- Must support same-speaker replay and cross-speaker reuse
- Must not degenerate to zero usage (regularization required)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_VOICE_STATE_SSL, D_ACTING_LATENT, D_ACTING_MACRO, D_MODEL


class ActingLatentEncoder(nn.Module):
    """Extracts acting texture latent from audio features.

    Used during training to extract the "ground truth" acting latent
    from reference audio, which the predictor then learns to approximate.

    Architecture: SSL features -> MLP -> bottleneck -> latent
    """

    def __init__(
        self,
        d_input: int = D_VOICE_STATE_SSL,
        d_hidden: int = 256,
        d_latent: int = D_ACTING_LATENT,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent

        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # Bottleneck to prevent latent from carrying too much info
        self.mu_proj = nn.Linear(d_hidden, d_latent)
        self.logvar_proj = nn.Linear(d_hidden, d_latent)

    def forward(
        self, ssl_features: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode SSL features to acting latent.

        Args:
            ssl_features: [B, T, d_input] SSL voice state features
            mask: [B, T] bool mask for valid frames

        Returns:
            latent: [B, d_latent] sampled latent (reparameterized)
            mu: [B, d_latent] mean
            logvar: [B, d_latent] log variance
        """
        h = self.encoder(ssl_features)  # [B, T, d_hidden]

        # Pool over time (masked mean)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            h = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            h = h.mean(dim=1)  # [B, d_hidden]

        mu = self.mu_proj(h)         # [B, d_latent]
        logvar = self.logvar_proj(h)  # [B, d_latent]
        # Clamp logvar to prevent extremely large/small variance values
        logvar = logvar.clamp(-10.0, 2.0)

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std
        else:
            latent = mu

        return latent, mu, logvar


class ActingLatentPredictor(nn.Module):
    """Predicts acting latent prior from text features and context.

    Used at inference time when no reference audio is available.
    Takes text encoder output + optional dialogue context and predicts
    what the acting latent should be.
    """

    def __init__(
        self,
        d_text: int = D_MODEL,
        d_context: int = D_MODEL,
        d_latent: int = D_ACTING_LATENT,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(d_text + d_context, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent),
        )

    def forward(
        self,
        text_features: torch.Tensor,      # [B, L, d_text] or [B, d_text]
        context_features: torch.Tensor | None = None,  # [B, d_context]
    ) -> torch.Tensor:
        """Predict acting latent from text and context.

        Returns:
            latent_prior: [B, d_latent]
        """
        # Pool text if needed
        if text_features.ndim == 3:
            text_pooled = text_features.mean(dim=1)  # [B, d_text]
        else:
            text_pooled = text_features

        if context_features is not None:
            combined = torch.cat([text_pooled, context_features], dim=-1)
        else:
            # Zero-pad context
            B = text_pooled.size(0)
            device = text_pooled.device
            zero_ctx = torch.zeros(
                B,
                self.predictor[0].in_features - text_pooled.size(-1),
                device=device,
            )
            combined = torch.cat([text_pooled, zero_ctx], dim=-1)

        return self.predictor(combined)  # [B, d_latent]


class ActingMacroProjector(nn.Module):
    """Projects 6 user-facing macro controls to 24-D latent space.

    Macro controls: intensity, instability, tenderness, tension, spontaneity, reference_mix
    These are NOT raw latent axes -- they are learned projections.
    """

    def __init__(
        self,
        d_macro: int = D_ACTING_MACRO,
        d_latent: int = D_ACTING_LATENT,
        d_hidden: int = 64,
    ):
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(d_macro, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_latent),
        )

    def forward(self, macro_controls: torch.Tensor) -> torch.Tensor:
        """Project macro controls to latent space.

        Args:
            macro_controls: [B, 6] macro control values

        Returns:
            latent_bias: [B, d_latent] bias to add to latent
        """
        return self.projector(macro_controls)


class ActingLatentConditioner(nn.Module):
    """Projects acting latent to model conditioning dimension.

    Takes the 24-D acting latent and projects it to d_model for injection
    into the transformer.
    """

    def __init__(
        self,
        d_latent: int = D_ACTING_LATENT,
        d_model: int = D_MODEL,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(d_latent, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Project latent to model space.

        Args:
            latent: [B, d_latent] or [B, 1, d_latent]

        Returns:
            condition: [B, d_model] or [B, 1, d_model]
        """
        return self.proj(latent)
