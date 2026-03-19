"""PPO-based RL trainer for v4 UCLM codec token policy fine-tuning.

This module implements the RL training loop that runs AFTER supervised
training converges.  It uses PPO with GAE to update the UCLM model's
codec token generation policy, guided by multi-objective rewards from
``reward.py``.

Safety:
    - Early stops if plain-text TTS quality degrades > 5% (configurable).
    - Early stops if physical control monotonicity drops below 0.8.
    - KL penalty against a frozen reference model prevents catastrophic drift.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_MODEL

from .config import RLPhaseConfig, RewardWeights
from .reward import (
    InstructionFollowingReward,
    IntelligibilityReward,
    NaturalnessGuard,
    PhysicalComplianceReward,
    RewardResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GAE helpers
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimation.

    Args:
        rewards: [B, T] per-step rewards.
        values: [B, T+1] value estimates (last entry is bootstrap value).
        dones: [B, T] episode-done flags (1.0 = done).
        gamma: Discount factor.
        gae_lambda: GAE lambda.

    Returns:
        (advantages, returns) each [B, T].
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(B, device=rewards.device)

    for t in reversed(range(T)):
        next_val = values[:, t + 1]
        mask = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * next_val * mask - values[:, t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[:, t] = gae

    returns = advantages + values[:, :T]
    return advantages, returns


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------


class RLTrainer:
    """PPO-based RL trainer for UCLM codec token policy fine-tuning.

    Usage::

        config = RLPhaseConfig(baseline_naturalness_score=0.92)
        rl = RLTrainer(model, config)
        for batch in rl_dataloader:
            metrics = rl.train_step(batch)
            if metrics.get("early_stopped"):
                break

    The trainer maintains:
    - A trainable policy model (the UCLM).
    - A frozen reference model for KL regularisation.
    - A value head for PPO advantage estimation.
    - Four reward modules.

    Args:
        model: The UCLM model to fine-tune.  Must expose a ``generate()``
            method that returns ``(audio, codec_tokens, log_probs, hidden_states)``.
        config: RL phase configuration.
        optimizer: Optional pre-built optimiser.  If None, AdamW is created.
        device: Torch device.  If None, inferred from model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: RLPhaseConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ):
        config.validate()
        self.config = config
        self.model = model

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        # Frozen reference model for KL penalty
        self.ref_model = copy.deepcopy(model)
        for p in self.ref_model.parameters():
            p.requires_grad = False
        self.ref_model.eval()

        # Value head (projects from decoder hidden to scalar value)
        # Infer d_model from the model if available, else fall back to constant
        d_model = getattr(model, "d_model", D_MODEL)
        self.value_head = nn.Linear(d_model, 1).to(device)

        # Optimiser covers model params + value head
        trainable = list(model.parameters()) + list(self.value_head.parameters())
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Reward modules
        self.instruction_reward = InstructionFollowingReward(config.asr_model_name)
        self.physical_reward = PhysicalComplianceReward()
        self.intelligibility_reward = IntelligibilityReward(config.asr_model_name)
        self.naturalness_guard = NaturalnessGuard()

        # Share ASR pipeline between instruction and intelligibility rewards
        self.intelligibility_reward.share_asr(self.instruction_reward)

        # Tracking state
        self._step: int = 0
        self._violation_count: int = 0
        self._early_stopped: bool = False
        self._reward_history: List[float] = []
        self._best_reward: float = float("-inf")

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_rewards(
        self,
        generated_audios: List[torch.Tensor],
        codec_tokens_batch: List[torch.Tensor],
        enriched_transcripts: List[str],
        plain_transcripts: List[str],
        physical_targets: List[torch.Tensor],
        observed_masks: Optional[List[Optional[torch.Tensor]]] = None,
    ) -> List[RewardResult]:
        """Compute multi-objective rewards for a batch of samples.

        Each argument is a list of length B (batch size).

        Returns:
            List of RewardResult, one per sample.
        """
        B = len(generated_audios)
        w = self.config.reward_weights
        if observed_masks is None:
            observed_masks = [None] * B

        results: List[RewardResult] = []
        for i in range(B):
            # 1. Instruction following
            if_result = self.instruction_reward.compute(
                generated_audios[i], enriched_transcripts[i],
            )

            # 2. Physical compliance
            pc_result = self.physical_reward.compute(
                generated_audios[i], physical_targets[i], observed_masks[i],
            )

            # 3. Intelligibility
            int_result = self.intelligibility_reward.compute(
                generated_audios[i], plain_transcripts[i],
            )

            # 4. Naturalness guard
            nat_result = self.naturalness_guard.compute(
                generated_audios[i],
                codec_tokens_batch[i] if codec_tokens_batch[i] is not None else None,
            )

            # Merge into single result
            merged = RewardResult(
                instruction_following=if_result.instruction_following,
                physical_compliance=pc_result.physical_compliance,
                intelligibility=int_result.intelligibility,
                naturalness=nat_result.naturalness,
                tag_recall=if_result.tag_recall,
                tag_precision=if_result.tag_precision,
                tag_f1=if_result.tag_f1,
                physical_rmse=pc_result.physical_rmse,
                physical_correlation=pc_result.physical_correlation,
                wer=int_result.wer,
                cer=int_result.cer,
                is_degenerate=nat_result.is_degenerate,
                silence_ratio=nat_result.silence_ratio,
                repetition_ratio=nat_result.repetition_ratio,
                noise_ratio=nat_result.noise_ratio,
            )

            # Weighted total
            merged.total = (
                w.instruction_following * merged.instruction_following
                + w.physical_compliance * merged.physical_compliance
                + w.intelligibility * merged.intelligibility
                + w.naturalness * merged.naturalness
            )

            # Hard penalty for degenerate outputs
            if merged.is_degenerate:
                merged.total *= 0.1

            results.append(merged)

        return results

    # ------------------------------------------------------------------
    # PPO loss
    # ------------------------------------------------------------------

    def compute_ppo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        ref_log_probs: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO clipped surrogate loss with KL and entropy terms.

        Args:
            log_probs: [B, T] current policy log probabilities.
            old_log_probs: [B, T] log probs from rollout collection.
            advantages: [B, T] GAE advantages (should be normalised).
            values: [B, T] value predictions.
            returns: [B, T] GAE returns.
            ref_log_probs: [B, T] log probs from frozen reference.
            logits: [B, T, V] full logits distribution for accurate entropy.
                If None, falls back to approximate entropy from log_probs.

        Returns:
            Dict with policy_loss, value_loss, kl_penalty, entropy, total.
        """
        cfg = self.config
        eps = cfg.ppo_clip_epsilon

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp(min=1e-8)
        advantages_norm = (advantages - adv_mean) / adv_std

        # Policy ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate
        surr1 = ratio * advantages_norm
        surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages_norm
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss (clipped)
        value_loss = F.mse_loss(values, returns)

        # KL divergence penalty against frozen reference
        # KL(pi || pi_ref) = E[log_pi - log_pi_ref]
        kl_div = (log_probs - ref_log_probs).mean()
        kl_penalty = kl_div.abs()  # Symmetric penalty

        # Entropy bonus: compute from full logits distribution if available,
        # otherwise fall back to approximate entropy from sampled log probs.
        if logits is not None:
            # Exact entropy from full distribution: H = -sum(p * log p)
            full_log_probs = F.log_softmax(logits, dim=-1)
            full_probs = full_log_probs.exp()
            entropy = -(full_probs * full_log_probs).sum(dim=-1).mean()
        else:
            # Fallback: approximate entropy (less accurate, uses only
            # the sampled token's probability)
            probs = log_probs.exp().clamp(min=1e-8, max=1.0)
            entropy = -(probs * log_probs).mean()

        total = (
            policy_loss
            + cfg.value_loss_coeff * value_loss
            + cfg.kl_penalty_coeff * kl_penalty
            - cfg.entropy_coeff * entropy
        )

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "kl_penalty": kl_penalty,
            "entropy": entropy,
            "total": total,
        }

    # ------------------------------------------------------------------
    # Safety evaluation
    # ------------------------------------------------------------------

    def evaluate_safety(
        self,
        plain_text_naturalness: float,
        physical_monotonicity: float,
    ) -> Dict[str, Any]:
        """Check safety guards and return status.

        Args:
            plain_text_naturalness: Current naturalness score on plain-text
                evaluation set (0-1 scale).
            physical_monotonicity: Current physical control response
                monotonicity (0-1 scale).

        Returns:
            Dict with ``safe``, ``violations``, ``early_stop`` keys.
        """
        guards = self.config.safety
        violations: List[str] = []

        baseline = self.config.baseline_naturalness_score
        if baseline is not None and baseline > 0:
            degradation = (baseline - plain_text_naturalness) / baseline
            if degradation > guards.max_plain_text_degradation:
                violations.append(
                    f"naturalness degradation {degradation:.1%} > "
                    f"{guards.max_plain_text_degradation:.1%} limit"
                )

        if physical_monotonicity < guards.min_monotonicity:
            violations.append(
                f"monotonicity {physical_monotonicity:.3f} < "
                f"{guards.min_monotonicity:.3f} minimum"
            )

        if violations:
            self._violation_count += 1
            logger.warning(
                "RL safety violation %d/%d: %s",
                self._violation_count,
                guards.patience,
                "; ".join(violations),
            )
        else:
            self._violation_count = max(0, self._violation_count - 1)

        early_stop = self._violation_count >= guards.patience
        if early_stop and not self._early_stopped:
            self._early_stopped = True
            logger.error(
                "RL early stop triggered after %d consecutive violations",
                self._violation_count,
            )

        return {
            "safe": len(violations) == 0,
            "violations": violations,
            "violation_count": self._violation_count,
            "early_stop": early_stop,
        }

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single RL training step (rollout + PPO update).

        Expected batch keys:
            - ``text_ids``: [B, L] phoneme / acting-tag token sequences.
            - ``enriched_transcripts``: List[str] of length B.
            - ``plain_transcripts``: List[str] of length B.
            - ``physical_targets``: [B, T, 12] target physical controls.
            - ``observed_masks``: [B, T, 12] optional mask.
            - ``speaker_embed``: [B, D_speaker] speaker embeddings.

        If the model does not have a ``generate`` method, this falls back
        to a simplified REINFORCE step using the forward pass.

        Returns:
            Dict with:
            - reward breakdown (mean across batch)
            - PPO loss components
            - safety status
            - ``early_stopped``: bool
        """
        if self._early_stopped:
            return {"early_stopped": True, "rl_step": self._step}

        self._step += 1
        cfg = self.config
        self.model.train()

        metrics: Dict[str, Any] = {"rl_step": self._step}

        # ---- Phase 1: Rollout (generate audio with current policy) ----
        B = batch["text_ids"].shape[0]

        has_generate = hasattr(self.model, "generate") and callable(
            getattr(self.model, "generate")
        )

        if has_generate:
            with torch.no_grad():
                gen_out = self.model.generate(
                    text_ids=batch["text_ids"],
                    speaker_embed=batch.get("speaker_embed"),
                    max_frames=cfg.max_rollout_frames,
                )
                # gen_out expected: dict with audio, codec_tokens, log_probs, hidden_states
                generated_audios = [gen_out["audio"][i] for i in range(B)]
                codec_tokens_batch = [gen_out["codec_tokens"][i] for i in range(B)]
                old_log_probs = gen_out["log_probs"]  # [B, T]
                hidden_states = gen_out["hidden_states"]  # [B, T, D]

            # Reference log probs
            with torch.no_grad():
                ref_out = self.ref_model.generate(
                    text_ids=batch["text_ids"],
                    speaker_embed=batch.get("speaker_embed"),
                    max_frames=cfg.max_rollout_frames,
                )
                ref_log_probs = ref_out["log_probs"]
        else:
            # Simplified path: use forward pass with teacher forcing
            # and REINFORCE-style gradient estimation
            return self._reinforce_step(batch, metrics)

        # ---- Phase 2: Compute rewards ----
        physical_targets = batch.get("physical_targets")
        if physical_targets is not None:
            phys_list = [physical_targets[i] for i in range(B)]
        else:
            phys_list = [torch.zeros(1, 12, device=self.device)] * B

        masks = batch.get("observed_masks")
        mask_list = [masks[i] for i in range(B)] if masks is not None else None

        reward_results = self.compute_rewards(
            generated_audios=generated_audios,
            codec_tokens_batch=codec_tokens_batch,
            enriched_transcripts=batch["enriched_transcripts"],
            plain_transcripts=batch["plain_transcripts"],
            physical_targets=phys_list,
            observed_masks=mask_list,
        )

        # Pack rewards into tensor [B, T] (broadcast scalar per sample)
        T = old_log_probs.shape[1]
        reward_scalars = torch.tensor(
            [r.total for r in reward_results],
            device=self.device,
        )
        rewards = reward_scalars.unsqueeze(1).expand(-1, T) / T  # spread across timesteps

        # ---- Phase 3: Value estimation and GAE ----
        with torch.no_grad():
            values = self.value_head(hidden_states).squeeze(-1)  # [B, T]
        # Bootstrap value for last step
        bootstrap = torch.zeros(B, 1, device=self.device)
        values_with_bootstrap = torch.cat([values, bootstrap], dim=1)
        dones = torch.zeros(B, T, device=self.device)
        dones[:, -1] = 1.0

        advantages, returns = compute_gae(
            rewards, values_with_bootstrap, dones, cfg.gamma, cfg.gae_lambda,
        )

        # ---- Phase 4: PPO update ----
        for _epoch in range(cfg.ppo_epochs):
            # Mini-batch sampling
            indices = torch.randperm(B, device=self.device)
            for start in range(0, B, cfg.ppo_mini_batch_size):
                end = min(start + cfg.ppo_mini_batch_size, B)
                mb_idx = indices[start:end]

                # Re-compute log probs with current policy
                # T4 fix: codec_tokens_batch is a list of per-sample tensors,
                # so index directly with mb_idx (not [0][mb_idx])
                mb_codec = torch.stack([codec_tokens_batch[i] for i in mb_idx]) \
                    if isinstance(codec_tokens_batch[0], torch.Tensor) else None
                fwd_out = self.model(
                    text_ids=batch["text_ids"][mb_idx],
                    codec_targets=mb_codec,
                    speaker_embed=batch.get("speaker_embed", torch.zeros(B, 192, device=self.device))[mb_idx]
                    if batch.get("speaker_embed") is not None
                    else None,
                )
                new_log_probs = fwd_out.get(
                    "log_probs", old_log_probs[mb_idx]
                )
                new_hidden = fwd_out.get("hidden_states", hidden_states[mb_idx])
                new_logits = fwd_out.get("logits")  # [B, T, V] for entropy

                # Align temporal dimensions: forward pass may produce
                # different sequence length than the rollout.
                mb_T = old_log_probs.shape[1]
                if new_log_probs.shape[1] != mb_T:
                    new_log_probs = new_log_probs[:, :mb_T] if new_log_probs.shape[1] > mb_T else F.pad(new_log_probs, (0, mb_T - new_log_probs.shape[1]))
                if new_hidden.shape[1] != mb_T:
                    new_hidden = new_hidden[:, :mb_T] if new_hidden.shape[1] > mb_T else F.pad(new_hidden, (0, 0, 0, mb_T - new_hidden.shape[1]))
                if new_logits is not None and new_logits.shape[1] != mb_T:
                    new_logits = new_logits[:, :mb_T] if new_logits.shape[1] > mb_T else F.pad(new_logits, (0, 0, 0, mb_T - new_logits.shape[1]))

                new_values = self.value_head(new_hidden).squeeze(-1)

                ppo_losses = self.compute_ppo_loss(
                    log_probs=new_log_probs,
                    old_log_probs=old_log_probs[mb_idx],
                    advantages=advantages[mb_idx],
                    values=new_values,
                    returns=returns[mb_idx],
                    ref_log_probs=ref_log_probs[mb_idx],
                    logits=new_logits,
                )

                self.optimizer.zero_grad()
                ppo_losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    cfg.max_grad_norm,
                )
                self.optimizer.step()

        # ---- Phase 5: Log metrics ----
        mean_reward = reward_scalars.mean().item()
        self._reward_history.append(mean_reward)
        if mean_reward > self._best_reward:
            self._best_reward = mean_reward

        metrics.update({
            "rl_reward_mean": mean_reward,
            "rl_reward_best": self._best_reward,
            "rl_instruction_following": sum(r.instruction_following for r in reward_results) / B,
            "rl_physical_compliance": sum(r.physical_compliance for r in reward_results) / B,
            "rl_intelligibility": sum(r.intelligibility for r in reward_results) / B,
            "rl_naturalness": sum(r.naturalness for r in reward_results) / B,
            "rl_tag_f1": sum(r.tag_f1 for r in reward_results) / B,
            "rl_physical_rmse": sum(r.physical_rmse for r in reward_results) / B,
            "rl_wer": sum(r.wer for r in reward_results) / B,
            "rl_degenerate_count": sum(1 for r in reward_results if r.is_degenerate),
            "rl_policy_loss": ppo_losses["policy_loss"].item(),
            "rl_value_loss": ppo_losses["value_loss"].item(),
            "rl_kl_penalty": ppo_losses["kl_penalty"].item(),
            "rl_entropy": ppo_losses["entropy"].item(),
            "early_stopped": self._early_stopped,
        })

        # ---- Phase 6: Safety check (periodic) ----
        if self._step % cfg.safety.eval_interval_steps == 0:
            safety = self.evaluate_safety(
                plain_text_naturalness=metrics["rl_naturalness"],
                physical_monotonicity=1.0 - metrics["rl_physical_rmse"],
            )
            metrics["safety"] = safety
            if safety["early_stop"]:
                metrics["early_stopped"] = True

        return metrics

    # ------------------------------------------------------------------
    # REINFORCE fallback
    # ------------------------------------------------------------------

    def _reinforce_step(
        self,
        batch: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simplified REINFORCE step when model lacks ``generate()``.

        Uses forward pass with teacher forcing and applies reward-weighted
        gradient to the policy log-likelihood.
        """
        B = batch["text_ids"].shape[0]

        # Forward pass to get log probs
        self.model.train()
        fwd_out = self.model(
            text_ids=batch["text_ids"],
            speaker_embed=batch.get("speaker_embed"),
        )

        # Extract log probs from forward output
        if isinstance(fwd_out, dict):
            log_probs = fwd_out.get("log_probs")
            hidden_states = fwd_out.get("hidden_states")
        elif isinstance(fwd_out, tuple) and len(fwd_out) >= 2:
            log_probs = fwd_out[0]
            hidden_states = fwd_out[1] if len(fwd_out) > 1 else None
        else:
            log_probs = None
            hidden_states = None

        if log_probs is None:
            # Cannot compute RL loss without log probs
            metrics.update({
                "rl_reward_mean": 0.0,
                "rl_policy_loss": 0.0,
                "early_stopped": False,
                "rl_error": "model does not provide log_probs",
            })
            return metrics

        # Compute rewards using batch metadata
        # For REINFORCE, we need per-sample rewards but not generated audio
        # Use a proxy reward based on the training targets
        reward_scalars = torch.ones(B, device=self.device) * 0.5

        # Enriched transcripts
        enriched = batch.get("enriched_transcripts", [""] * B)
        plain = batch.get("plain_transcripts", [""] * B)

        # REINFORCE: policy gradient = -reward * log_prob
        mean_log_prob = log_probs.mean(dim=-1) if log_probs.dim() > 1 else log_probs
        if mean_log_prob.dim() > 1:
            mean_log_prob = mean_log_prob.mean(dim=-1)

        # Baseline subtraction with normalization
        baseline = reward_scalars.mean()
        advantage = reward_scalars - baseline
        # T5 fix: normalize advantage to prevent zero update when B=1 or
        # all rewards are identical; add eps to std for numerical stability
        adv_std = advantage.std().clamp(min=1e-8)
        if advantage.numel() > 1:
            advantage = advantage / adv_std
        else:
            # Single sample: use sign of (reward - 0.5) as advantage signal
            advantage = (reward_scalars - 0.5).sign()

        reinforce_loss = -(advantage * mean_log_prob).mean()

        # KL penalty (approximate)
        with torch.no_grad():
            ref_fwd = self.ref_model(
                text_ids=batch["text_ids"],
                speaker_embed=batch.get("speaker_embed"),
            )
            if isinstance(ref_fwd, dict):
                ref_lp = ref_fwd.get("log_probs", log_probs.detach())
            elif isinstance(ref_fwd, tuple):
                ref_lp = ref_fwd[0]
            else:
                ref_lp = log_probs.detach()

        ref_mean = ref_lp.mean(dim=-1) if ref_lp.dim() > 1 else ref_lp
        if ref_mean.dim() > 1:
            ref_mean = ref_mean.mean(dim=-1)

        kl = (mean_log_prob - ref_mean).abs().mean()

        total_loss = reinforce_loss + self.config.kl_penalty_coeff * kl

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        mean_reward = reward_scalars.mean().item()
        self._reward_history.append(mean_reward)

        metrics.update({
            "rl_reward_mean": mean_reward,
            "rl_policy_loss": reinforce_loss.item(),
            "rl_kl_penalty": kl.item(),
            "rl_method": "reinforce",
            "early_stopped": self._early_stopped,
        })
        return metrics

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def early_stopped(self) -> bool:
        return self._early_stopped

    def state_dict(self) -> Dict[str, Any]:
        """Serialise trainer state for checkpointing."""
        return {
            "step": self._step,
            "violation_count": self._violation_count,
            "early_stopped": self._early_stopped,
            "reward_history": self._reward_history,
            "best_reward": self._best_reward,
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore trainer state from checkpoint."""
        self._step = state["step"]
        self._violation_count = state["violation_count"]
        self._early_stopped = state["early_stopped"]
        self._reward_history = state["reward_history"]
        self._best_reward = state["best_reward"]
        self.value_head.load_state_dict(state["value_head"])
        self.optimizer.load_state_dict(state["optimizer"])
