"""Acting-specific loss functions for v4 training.

Covers:
- Acting latent regularization (KL divergence, prevent collapse)
- Disentanglement loss (physical vs latent separation)
- Semantic alignment loss (latent should align with semantic annotations)
- Residual usage penalty (latent must capture non-physical residuals)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def acting_latent_kl_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_nats: float = 2.0,
) -> torch.Tensor:
    """KL divergence regularization for acting latent.

    Prevents latent collapse while maintaining expressiveness.
    Uses free-nats threshold to allow some information in the latent.

    Args:
        mu: [B, d_latent] latent mean
        logvar: [B, d_latent] latent log variance
        free_nats: Minimum KL before penalty kicks in

    Returns:
        Scalar KL loss
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl_per_dim.sum(dim=-1).mean()

    # Free nats: don't penalize KL below this threshold
    kl = torch.clamp(kl - free_nats, min=0.0)

    return kl


def acting_latent_usage_loss(
    latent: torch.Tensor,
    min_variance: float = 0.01,
) -> torch.Tensor:
    """Penalize latent collapse (zero usage).

    The latent should have non-trivial variance across the batch.
    If all latents converge to the same point, this loss increases.

    Args:
        latent: [B, d_latent] latent vectors
        min_variance: Minimum acceptable variance per dimension

    Returns:
        Scalar loss (high when latent is collapsed)
    """
    # Variance across batch for each dimension
    # Use unbiased=False to avoid NaN when B=1 (Bessel's correction divides by 0)
    var_per_dim = latent.var(dim=0, unbiased=False)  # [d_latent]

    # Penalize dimensions with variance below threshold
    collapse_penalty = F.relu(min_variance - var_per_dim).mean()

    return collapse_penalty


def disentanglement_loss(
    physical_pred: torch.Tensor,
    acting_latent: torch.Tensor,
    d_physical: int = 12,
) -> torch.Tensor:
    """Encourage separation between physical controls and acting latent.

    The acting latent should NOT duplicate information already in the physical path.
    Uses correlation penalty: minimize mutual information proxy.

    Args:
        physical_pred: [B, T, d_physical] predicted physical controls
        acting_latent: [B, d_latent] acting latent (broadcast over T)

    Returns:
        Scalar disentanglement loss
    """
    # Pool physical over time — handle both [B, T, 12] and [B, 12]
    if physical_pred.ndim == 2:
        physical_pooled = physical_pred  # already [B, d_physical]
    else:
        physical_pooled = physical_pred.mean(dim=1)  # [B, d_physical]

    # Cross-correlation matrix
    # Normalize both
    p_norm = (physical_pooled - physical_pooled.mean(0)) / (physical_pooled.std(0) + 1e-8)
    l_norm = (acting_latent - acting_latent.mean(0)) / (acting_latent.std(0) + 1e-8)

    # [d_physical, d_latent] cross-correlation
    if p_norm.size(0) > 1:
        cross_corr = (p_norm.T @ l_norm) / (p_norm.size(0) - 1)
        # Penalize high cross-correlation
        loss = (cross_corr ** 2).mean()
    else:
        loss = torch.tensor(0.0, device=physical_pred.device)

    return loss


def semantic_alignment_loss(
    acting_latent: torch.Tensor,
    target_latent: torch.Tensor,
) -> torch.Tensor:
    """Alignment loss between predicted and target acting latent.

    During training, the target is the encoder output from reference audio.
    The predictor should learn to approximate this.

    Args:
        acting_latent: [B, d_latent] predicted latent (from predictor)
        target_latent: [B, d_latent] target latent (from encoder)

    Returns:
        Scalar MSE loss
    """
    return F.mse_loss(acting_latent, target_latent.detach())
