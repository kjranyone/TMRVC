"""Latency-Conditioned Distillation (LCD) — losses and q conditioning.

LCD injects a latency budget ``q ∈ [0, 1]`` during distillation:
- ``q = 0``: maximum latency (80ms, HQ lookahead mode)
- ``q = 1``: minimum latency (20ms, live causal mode)

The student learns to produce quality proportional to the latency budget.

Loss terms:
- ``L_latency``: penalizes when estimated runtime exceeds budget(q)
- ``L_mono``: enforces quality monotonically increases with more latency (lower q)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Budget function: linear interpolation between HQ and Live latency
HQ_MODE_MS = 80.0   # q=0 → 80ms
LIVE_MODE_MS = 20.0  # q=1 → 20ms


def latency_budget(q: torch.Tensor) -> torch.Tensor:
    """Compute latency budget from q.

    Args:
        q: ``[B]`` latency parameter in [0, 1].

    Returns:
        ``[B]`` latency budget in ms.
    """
    return HQ_MODE_MS * (1.0 - q) + LIVE_MODE_MS * q


class LatencyConditioner(nn.Module):
    """Embed scalar q into a conditioning vector for the student.

    Produces a FiLM-style modulation that scales the student input
    based on latency budget.

    Args:
        d_output: Output embedding dimension.
    """

    def __init__(self, d_output: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_output),
            nn.SiLU(),
            nn.Linear(d_output, d_output),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Embed q scalar into conditioning vector.

        Args:
            q: ``[B]`` latency parameter.

        Returns:
            ``[B, d_output]`` latency embedding.
        """
        return self.net(q.unsqueeze(-1))  # [B, 1] → [B, d_output]


class LatencyLoss(nn.Module):
    """L_latency: penalty when estimated processing time exceeds budget.

    ``L_latency = max(0, estimated_runtime - budget(q))^2``

    The estimated runtime is modeled as a function of the student's
    intermediate activations (proxy for computational cost).

    Args:
        base_runtime_ms: Base runtime of student (no overhead).
        overhead_scale: Scale factor for activation-based overhead estimate.
    """

    def __init__(
        self,
        base_runtime_ms: float = 15.0,
        overhead_scale: float = 5.0,
    ) -> None:
        super().__init__()
        self.base_runtime_ms = base_runtime_ms
        self.overhead_scale = overhead_scale

    def forward(
        self,
        q: torch.Tensor,
        activation_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute latency penalty.

        Args:
            q: ``[B]`` latency parameter.
            activation_norm: ``[B]`` L2 norm of student activations (proxy for cost).

        Returns:
            Scalar latency penalty.
        """
        budget = latency_budget(q)  # [B]
        estimated = self.base_runtime_ms + self.overhead_scale * activation_norm  # [B]
        excess = F.relu(estimated - budget)  # [B]
        return (excess ** 2).mean()


class MonotonicityLoss(nn.Module):
    """L_mono: enforce quality monotonically increases with latency.

    For ``q_low < q_high`` (less latency budget), quality should be lower:
    ``Q(y(q_low)) <= Q(y(q_high)) - margin``

    Quality is measured as negative mel MSE (higher = better).

    Args:
        margin: Minimum quality improvement margin.
    """

    def __init__(self, margin: float = 0.05) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        quality_low_q: torch.Tensor,
        quality_high_q: torch.Tensor,
    ) -> torch.Tensor:
        """Compute monotonicity penalty.

        Both inputs represent quality metrics (higher = better).
        Penalizes when ``quality_high_q + margin < quality_low_q``
        (i.e. more latency gives worse quality).

        Args:
            quality_low_q: ``[B]`` quality at lower latency budget (higher q).
            quality_high_q: ``[B]`` quality at higher latency budget (lower q).

        Returns:
            Scalar monotonicity penalty.
        """
        violation = F.relu(quality_low_q - quality_high_q + self.margin)
        return violation.mean()
