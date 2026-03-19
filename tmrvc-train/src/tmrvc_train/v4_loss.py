"""v4 unified loss composition.

All v4 loss terms from master plan section 8.6:
1. codec token prediction loss
2. control token prediction loss
3. pointer progression loss
4. explicit physical supervision loss (12-D)
5. acting latent regularization loss
6. disentanglement loss
7. speaker consistency loss
8. prosody prediction loss
9. semantic alignment loss

Plus biological constraint losses from track_training §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class V4LossConfig:
    """Configuration for v4 loss composition."""
    # Core losses
    lambda_codec: float = 1.0
    lambda_control: float = 1.0
    lambda_pointer: float = 0.5
    lambda_progress: float = 0.3
    lambda_boundary: float = 0.3

    # Physical supervision
    lambda_physical: float = 1.0
    lambda_delta_physical: float = 0.5

    # Acting latent
    lambda_acting_kl: float = 0.01
    lambda_acting_usage: float = 0.1
    lambda_disentanglement: float = 0.1
    lambda_semantic_align: float = 0.5

    # Biological constraints
    lambda_bio_covariance: float = 0.1
    lambda_bio_transition: float = 0.1
    lambda_bio_implausibility: float = 0.05

    # Speaker / prosody
    lambda_speaker: float = 0.5
    lambda_prosody: float = 0.5
    lambda_contrastive: float = 0.1

    # VQ
    lambda_vq: float = 0.1

    # Adversarial
    lambda_adversarial: float = 0.1


@dataclass
class V4LossResult:
    """Holds all individual loss terms and total."""
    total: torch.Tensor = None

    # Core
    codec_loss: torch.Tensor = None
    control_loss: torch.Tensor = None
    pointer_loss: torch.Tensor = None
    progress_loss: torch.Tensor = None
    boundary_loss: torch.Tensor = None

    # Physical
    physical_loss: torch.Tensor = None
    delta_physical_loss: torch.Tensor = None

    # Acting
    acting_kl_loss: torch.Tensor = None
    acting_usage_loss: torch.Tensor = None
    disentanglement_loss: torch.Tensor = None
    semantic_align_loss: torch.Tensor = None

    # Biological
    bio_covariance_loss: torch.Tensor = None
    bio_transition_loss: torch.Tensor = None
    bio_implausibility_loss: torch.Tensor = None

    # Speaker / prosody
    speaker_loss: torch.Tensor = None
    prosody_loss: torch.Tensor = None
    contrastive_loss: torch.Tensor = None

    # VQ / adversarial
    vq_loss: torch.Tensor = None
    adversarial_loss: torch.Tensor = None

    def to_dict(self) -> dict:
        """Convert to dict for logging, skipping None values."""
        result = {}
        for k, v in vars(self).items():
            if v is not None and isinstance(v, torch.Tensor):
                result[k] = v.detach().item()
        return result


def compute_v4_total_loss(
    result: V4LossResult,
    config: V4LossConfig,
    tier_weights: Optional[dict] = None,
) -> torch.Tensor:
    """Compute weighted sum of all loss terms.

    Args:
        result: V4LossResult with individual losses populated
        config: Weight configuration
        tier_weights: Optional supervision tier weight overrides

    Returns:
        Total loss tensor
    """
    device = None
    # Find a non-None loss to get device
    for v in vars(result).values():
        if isinstance(v, torch.Tensor):
            device = v.device
            break

    if device is None:
        return torch.tensor(0.0)

    total = torch.tensor(0.0, device=device)
    tw = tier_weights or {}

    def _add(loss, lambda_val, tier_key=None):
        nonlocal total
        if loss is not None:
            w = lambda_val
            if tier_key and tier_key in tw:
                w *= tw[tier_key]
            total = total + w * loss

    _add(result.codec_loss, config.lambda_codec, "codec_loss")
    _add(result.control_loss, config.lambda_control, "control_loss")
    _add(result.pointer_loss, config.lambda_pointer, "pointer_loss")
    _add(result.progress_loss, config.lambda_progress)
    _add(result.boundary_loss, config.lambda_boundary)
    _add(result.physical_loss, config.lambda_physical, "physical_loss")
    _add(result.delta_physical_loss, config.lambda_delta_physical)
    _add(result.acting_kl_loss, config.lambda_acting_kl, "acting_latent_loss")
    _add(result.acting_usage_loss, config.lambda_acting_usage, "acting_latent_loss")
    _add(result.disentanglement_loss, config.lambda_disentanglement, "disentanglement_loss")
    _add(result.semantic_align_loss, config.lambda_semantic_align, "semantic_loss")
    _add(result.bio_covariance_loss, config.lambda_bio_covariance)
    _add(result.bio_transition_loss, config.lambda_bio_transition)
    _add(result.bio_implausibility_loss, config.lambda_bio_implausibility)
    _add(result.speaker_loss, config.lambda_speaker, "speaker_loss")
    _add(result.prosody_loss, config.lambda_prosody, "prosody_loss")
    _add(result.contrastive_loss, config.lambda_contrastive)
    _add(result.vq_loss, config.lambda_vq)
    _add(result.adversarial_loss, config.lambda_adversarial)

    result.total = total
    return total
