"""RL phase configuration for v4 UCLM fine-tuning.

Contains PPO hyperparameters, reward weights, and safety guards that
control early-stopping when plain-text TTS quality degrades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardWeights:
    """Weights for the four reward components.

    The total reward is a weighted sum:
        R = w_if * R_instruction + w_pc * R_physical
          + w_int * R_intelligibility + w_nat * R_naturalness
    """

    instruction_following: float = 1.0
    physical_compliance: float = 0.5
    intelligibility: float = 0.3
    naturalness: float = 0.2

    def total_weight(self) -> float:
        return (
            self.instruction_following
            + self.physical_compliance
            + self.intelligibility
            + self.naturalness
        )

    def normalised(self) -> "RewardWeights":
        """Return a copy where weights sum to 1.0."""
        t = self.total_weight()
        if t <= 0:
            return RewardWeights(0.25, 0.25, 0.25, 0.25)
        return RewardWeights(
            instruction_following=self.instruction_following / t,
            physical_compliance=self.physical_compliance / t,
            intelligibility=self.intelligibility / t,
            naturalness=self.naturalness / t,
        )


@dataclass
class SafetyGuards:
    """Safety constraints for RL fine-tuning.

    The RL loop checks these after every evaluation interval:
    - plain-text TTS quality must not degrade by more than ``max_plain_text_degradation``
    - physical control monotonicity must stay above ``min_monotonicity``
    - if either guard trips, training halts immediately
    """

    max_plain_text_degradation: float = 0.05  # 5%
    min_monotonicity: float = 0.8
    eval_interval_steps: int = 200
    patience: int = 3  # consecutive violations before hard stop


@dataclass
class RLPhaseConfig:
    """Full configuration for the RL fine-tuning phase.

    Attributes:
        ppo_clip_epsilon: PPO clipping parameter for the surrogate objective.
        ppo_epochs: Number of PPO mini-batch passes per rollout batch.
        ppo_mini_batch_size: Samples per PPO mini-batch.
        gamma: Discount factor for GAE return estimation.
        gae_lambda: Lambda for generalised advantage estimation.
        max_grad_norm: Gradient clipping threshold.
        lr: Learning rate for the RL optimiser (AdamW).
        kl_penalty_coeff: Coefficient for KL divergence penalty against
            the frozen reference model.
        value_loss_coeff: Coefficient for the value function loss in PPO.
        entropy_coeff: Entropy bonus coefficient to encourage exploration.
        rollout_batch_size: Number of samples generated per rollout.
        max_rollout_frames: Maximum codec frames per generated sample.
        max_steps: Total RL training steps. 0 = unlimited (rely on guards).
        reward_weights: Component reward weights.
        safety: Safety guard configuration.
        baseline_naturalness_score: Pre-RL naturalness score measured on
            a held-out plain-text evaluation set.  Set this after supervised
            training finishes.
        asr_model_name: HuggingFace model ID for the rich-transcription
            ASR used in instruction-following reward.
        seed: Random seed for reproducibility.
    """

    # PPO hyperparameters
    ppo_clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    ppo_mini_batch_size: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0

    # Optimiser
    lr: float = 1e-5
    weight_decay: float = 0.01

    # Loss coefficients
    kl_penalty_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01

    # Rollout
    rollout_batch_size: int = 8
    max_rollout_frames: int = 1000

    # Schedule
    max_steps: int = 5000

    # Reward / safety
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    safety: SafetyGuards = field(default_factory=SafetyGuards)

    # Baseline (set after supervised phase)
    baseline_naturalness_score: Optional[float] = None

    # ASR reward model
    asr_model_name: str = "Qwen/Qwen3-ASR-1.7B"

    # Reproducibility
    seed: int = 42

    def validate(self) -> None:
        """Raise ``ValueError`` on invalid configuration."""
        if self.ppo_clip_epsilon <= 0 or self.ppo_clip_epsilon >= 1:
            raise ValueError(
                f"ppo_clip_epsilon must be in (0, 1), got {self.ppo_clip_epsilon}"
            )
        if self.ppo_epochs < 1:
            raise ValueError(f"ppo_epochs must be >= 1, got {self.ppo_epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.safety.max_plain_text_degradation <= 0:
            raise ValueError("max_plain_text_degradation must be positive")
        if not 0.0 < self.safety.min_monotonicity <= 1.0:
            raise ValueError("min_monotonicity must be in (0, 1]")
