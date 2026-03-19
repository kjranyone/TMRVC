"""Biological constraint regularization for v4.

Enforces physically plausible co-occurrence and temporal dynamics
in the 12-D physical control space.

Constraints:
- Low-rank covariance prior: physical controls covary in low-rank structure
- Frame-to-frame transition prior: smooth causal transitions
- Physically implausible combination penalty

Rules:
- No future-frame smoothing (causal only)
- Constraints must produce non-zero gradients
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_VOICE_STATE, BIO_COVARIANCE_RANK, BIO_TRANSITION_PENALTY_WEIGHT


class BiologicalConstraintRegularizer(nn.Module):
    """Regularizes physical control predictions for biological plausibility.

    Uses:
    1. Low-rank covariance prior (learned)
    2. Frame-to-frame transition penalty (causal)
    3. Implausible combination detection
    """

    def __init__(
        self,
        d_physical: int = D_VOICE_STATE,
        covariance_rank: int = BIO_COVARIANCE_RANK,
        transition_weight: float = BIO_TRANSITION_PENALTY_WEIGHT,
    ):
        super().__init__()
        self.d_physical = d_physical
        self.transition_weight = transition_weight

        # Low-rank covariance prior: L @ L^T approximates the learned covariance
        # Physical controls should covary along this low-rank structure
        self.cov_factor = nn.Parameter(
            torch.randn(d_physical, covariance_rank) * 0.1
        )
        # Diagonal component
        self.cov_diag = nn.Parameter(
            torch.ones(d_physical) * 0.5
        )

        # Implausible combination detector
        # Learned scoring function: high score = implausible
        self.implausibility_net = nn.Sequential(
            nn.Linear(d_physical, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def compute_covariance_loss(
        self, physical_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Low-rank covariance prior loss.

        Encourages predicted physical controls to follow the learned
        covariance structure.

        Args:
            physical_pred: [B, T, 12] predicted physical controls

        Returns:
            Scalar loss
        """
        B, T, D = physical_pred.shape

        # Compute sample covariance
        centered = physical_pred - physical_pred.mean(dim=1, keepdim=True)
        # [B, D, D]
        sample_cov = torch.bmm(centered.transpose(1, 2), centered) / max(T - 1, 1)

        # Learned low-rank covariance: L @ L^T + diag
        learned_cov = (
            self.cov_factor @ self.cov_factor.T
            + torch.diag(F.softplus(self.cov_diag))
        )  # [D, D]

        # Frobenius norm between sample and learned covariance
        loss = F.mse_loss(sample_cov, learned_cov.unsqueeze(0).expand(B, -1, -1))

        return loss

    def compute_transition_loss(
        self, physical_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Frame-to-frame transition smoothness penalty (causal only).

        Penalizes large jumps between consecutive frames.
        Only uses past->current direction (no future smoothing).

        Args:
            physical_pred: [B, T, 12] predicted physical controls

        Returns:
            Scalar loss
        """
        if physical_pred.size(1) < 2:
            return (physical_pred * 0.0).sum()

        # Causal: only penalize t -> t+1 transitions
        deltas = physical_pred[:, 1:, :] - physical_pred[:, :-1, :]  # [B, T-1, 12]

        # L2 penalty on transitions
        transition_loss = (deltas ** 2).mean()

        return transition_loss * self.transition_weight

    @staticmethod
    def _generate_implausible_samples(
        physical_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Generate synthetic biologically implausible physical control combinations.

        Creates negative examples by producing extreme co-occurrences that violate
        known physical constraints (e.g., maximum breathiness + maximum vocal effort,
        extreme pitch + extreme jaw closure).

        Args:
            physical_pred: [B, T, 12] reference plausible controls (for shape/device)

        Returns:
            [B, T, 12] implausible synthetic samples
        """
        B, T, D = physical_pred.shape
        device = physical_pred.device

        # Start from random permutations of the real data to break natural covariance
        idx = torch.randint(0, B, (B,), device=device)
        neg = physical_pred[idx].clone()

        # Strategy 1: push correlated dimensions to opposite extremes
        # breathiness (dim 3) high + vocal_effort (dim 10) high is implausible
        mask1 = torch.rand(B, T, 1, device=device) < 0.5
        neg[:, :, 3:4] = torch.where(mask1, torch.ones_like(neg[:, :, 3:4]), neg[:, :, 3:4])
        neg[:, :, 10:11] = torch.where(mask1, torch.ones_like(neg[:, :, 10:11]), neg[:, :, 10:11])

        # Strategy 2: random extreme values on multiple dimensions simultaneously
        mask2 = torch.rand(B, T, D, device=device) < 0.3
        extremes = torch.where(
            torch.rand(B, T, D, device=device) < 0.5,
            torch.zeros(B, T, D, device=device),
            torch.ones(B, T, D, device=device),
        )
        neg = torch.where(mask2, extremes, neg)

        return neg.detach()

    def compute_implausibility_loss(
        self, physical_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Penalty for physically implausible parameter combinations.

        Uses a learned scoring function trained with both positive (plausible,
        from model predictions) and negative (implausible, synthetically generated)
        examples. Without negative examples the loss collapses to 0.

        Args:
            physical_pred: [B, T, 12] predicted physical controls

        Returns:
            Scalar loss (binary cross-entropy for the classifier)
        """
        # Score real (plausible) samples — target label 0 (not implausible)
        scores_real = self.implausibility_net(physical_pred)  # [B, T, 1]

        # Generate and score synthetic implausible samples — target label 1
        neg_samples = self._generate_implausible_samples(physical_pred)
        scores_neg = self.implausibility_net(neg_samples)  # [B, T, 1]

        # Binary cross-entropy: real->0, negative->1
        loss_real = F.binary_cross_entropy_with_logits(
            scores_real, torch.zeros_like(scores_real),
        )
        loss_neg = F.binary_cross_entropy_with_logits(
            scores_neg, torch.ones_like(scores_neg),
        )

        # Also penalize the model's own predictions that look implausible
        implausibility_penalty = torch.sigmoid(scores_real).mean()

        return 0.5 * (loss_real + loss_neg) + implausibility_penalty

    def forward(
        self, physical_pred: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all biological constraint losses.

        Args:
            physical_pred: [B, T, 12] predicted physical controls

        Returns:
            Dict with individual loss components and total
        """
        cov_loss = self.compute_covariance_loss(physical_pred)
        trans_loss = self.compute_transition_loss(physical_pred)
        implaus_loss = self.compute_implausibility_loss(physical_pred)

        total = cov_loss + trans_loss + implaus_loss

        return {
            "bio_covariance_loss": cov_loss,
            "bio_transition_loss": trans_loss,
            "bio_implausibility_loss": implaus_loss,
            "bio_total_loss": total,
        }
