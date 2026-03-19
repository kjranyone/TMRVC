"""RL fine-tuning package for v4 instruction-following.

This package implements Phase 3-5 of the v4 training pipeline:
PPO-based RL fine-tuning where the UCLM codec token policy is
refined using multi-objective rewards after supervised training
converges.

Submodules:
    config      - RLPhaseConfig with PPO hyperparameters, reward weights, safety guards
    reward      - Four reward classes for multi-objective optimisation
    trainer_rl  - RLTrainer with PPO update loop and early stopping
"""

from tmrvc_train.rl.config import RLPhaseConfig, RewardWeights, SafetyGuards
from tmrvc_train.rl.reward import (
    InstructionFollowingReward,
    PhysicalComplianceReward,
    IntelligibilityReward,
    NaturalnessGuard,
    RewardResult,
)
from tmrvc_train.rl.trainer_rl import RLTrainer

__all__ = [
    "RLPhaseConfig",
    "RewardWeights",
    "SafetyGuards",
    "InstructionFollowingReward",
    "PhysicalComplianceReward",
    "IntelligibilityReward",
    "NaturalnessGuard",
    "RewardResult",
    "RLTrainer",
]
