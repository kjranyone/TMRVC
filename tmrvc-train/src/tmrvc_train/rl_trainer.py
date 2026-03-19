#!/usr/bin/env python3
"""RL fine-tuning CLI entry point for v4 UCLM.

Invoked by dev.py as:
    python -m tmrvc_train.rl_trainer --base-checkpoint <path> ...

Loads a supervised checkpoint, builds an RLTrainer, and runs the PPO loop
with multi-objective rewards and safety guards.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    N_ACTING_TAGS, N_CODEBOOKS, RVQ_VOCAB_SIZE,
    CONTROL_VOCAB_SIZE, PHONEME_VOCAB_SIZE, CONTROL_SLOTS,
)  # noqa: F401 — CONTROL_SLOTS used in build_dataloader

from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.rl import RLPhaseConfig, RewardWeights, SafetyGuards, RLTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_trainer")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="v4 RL fine-tuning (PPO) for instruction-following",
    )
    p.add_argument("--base-checkpoint", required=True, help="Supervised checkpoint path")
    p.add_argument("--output-dir", default="checkpoints/rl", help="Output directory")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl-coeff", type=float, default=0.01)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--w-instruction", type=float, default=1.0)
    p.add_argument("--w-physical", type=float, default=0.5)
    p.add_argument("--w-intelligibility", type=float, default=0.3)
    p.add_argument("--w-naturalness", type=float, default=0.2)
    p.add_argument("--max-degradation", type=float, default=0.05,
                   help="Maximum plain-text quality degradation (fraction)")
    p.add_argument("--resume", default=None, help="Resume from RL checkpoint")
    p.add_argument("--additional-steps", type=int, default=0,
                   help="Extra steps when resuming (added to checkpoint step)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=200)
    return p.parse_args()


def load_model(checkpoint_path: str, device: str) -> DisentangledUCLM:
    """Load supervised checkpoint into a DisentangledUCLM model."""
    model = DisentangledUCLM(
        d_model=D_MODEL,
        d_explicit=D_VOICE_STATE,
        d_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER,
        n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE,
        control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE,
        num_speakers=100,
        acting_tag_vocab_size=N_ACTING_TAGS,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    logger.info("Loaded supervised checkpoint from %s", checkpoint_path)
    return model


def build_dataloader(batch_size: int):
    """Build a DataLoader for RL training from cached v4 data."""
    from tmrvc_data.v4_dataset import V4UCLMDataset, v4_collate_fn

    root = Path(__file__).resolve().parent.parent.parent.parent
    cache_dir = root / "data" / "cache"

    dataset = V4UCLMDataset(
        cache_dir=str(cache_dir),
        max_frames=1000,
        min_frames=10,
        use_enriched_transcript=True,
        enriched_transcript_prob=0.5,
    )

    logger.info("RL dataset: %d samples", len(dataset))

    def _collate(samples):
        """Collate V4 samples and map to RLTrainer expected keys."""
        raw = v4_collate_fn(samples)
        B = raw["codec_tokens_a"].shape[0] if raw.get("codec_tokens_a") is not None else 1
        T = raw["codec_tokens_a"].shape[-1] if raw.get("codec_tokens_a") is not None else 1

        d = {
            # RLTrainer expects text_ids, not phoneme_ids
            "text_ids": raw.get("phoneme_ids", torch.zeros(B, 1, dtype=torch.long)),
            "speaker_embed": raw.get("speaker_embed", torch.zeros(B, D_SPEAKER)),
            "physical_targets": raw.get("physical_targets"),
            "observed_masks": raw.get("physical_observed_mask"),
            # Enriched/plain transcripts as lists
            "enriched_transcripts": raw.get("enriched_transcript", [""] * B),
            "plain_transcripts": raw.get("text", [""] * B),
            # Also pass through keys for supervised path fallback
            "target_a": raw["codec_tokens_a"],
            "target_b": raw["codec_tokens_b"] if raw.get("codec_tokens_b") is not None else torch.zeros(B, CONTROL_SLOTS, T, dtype=torch.long),
        }
        return d

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate, drop_last=True,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # --- Load model ---
    if args.resume:
        logger.info("Resuming RL from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_step = ckpt.get("step", 0)
        max_steps = start_step + (args.additional_steps or args.max_steps)
        base_path = ckpt.get("base_checkpoint", args.base_checkpoint)
        model = load_model(base_path, device)
        # Restore RL-updated weights
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        _resume_ckpt = ckpt  # saved for ref_model restoration below
    else:
        model = load_model(args.base_checkpoint, device)
        start_step = 0
        max_steps = args.max_steps
        _resume_ckpt = None

    # --- RL config ---
    config = RLPhaseConfig(
        lr=args.lr,
        kl_penalty_coeff=args.kl_coeff,
        max_steps=max_steps,
        reward_weights=RewardWeights(
            instruction_following=args.w_instruction,
            physical_compliance=args.w_physical,
            intelligibility=args.w_intelligibility,
            naturalness=args.w_naturalness,
        ),
        safety=SafetyGuards(
            max_plain_text_degradation=args.max_degradation,
        ),
    )

    # --- Build RL trainer ---
    rl_trainer = RLTrainer(model=model, config=config, device=torch.device(device))

    # Restore ref_model and trainer state on resume
    if _resume_ckpt is not None:
        if "ref_model" in _resume_ckpt:
            rl_trainer.ref_model.load_state_dict(_resume_ckpt["ref_model"], strict=False)
            logger.info("Restored ref_model from checkpoint")
        if "rl_trainer_state" in _resume_ckpt:
            rl_trainer.load_state_dict(_resume_ckpt["rl_trainer_state"])
            logger.info("Restored RL trainer state (step=%d)", rl_trainer.step)

    # --- Dataloader ---
    dataloader = build_dataloader(args.batch_size)
    if len(dataloader.dataset) == 0:
        logger.error("No RL data available. Exiting.")
        sys.exit(1)

    # --- Training loop ---
    logger.info("=" * 60)
    logger.info("RL fine-tuning: steps %d -> %d", start_step, max_steps)
    logger.info("  lr=%g  kl_coeff=%g  max_degradation=%.1f%%",
                args.lr, args.kl_coeff, args.max_degradation * 100)
    logger.info("  reward weights: instr=%.1f phys=%.1f intel=%.1f nat=%.1f",
                args.w_instruction, args.w_physical,
                args.w_intelligibility, args.w_naturalness)
    logger.info("=" * 60)

    step = start_step
    epoch = 0
    running = {}
    t0 = time.time()
    metrics_history_path = output_dir / "rl_metrics_history.jsonl"

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            if step >= max_steps:
                break

            metrics = rl_trainer.train_step(batch)
            step += 1

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + v

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: v / args.log_every for k, v in running.items()}
                reward = avg.get("reward_mean", 0)
                logger.info(
                    "step %d/%d | reward=%.4f | %.2f s/step | %s",
                    step, max_steps, reward, elapsed / max(step - start_step, 1),
                    " ".join(f"{k}={avg[k]:.4f}" for k in sorted(avg)
                             if k != "reward_mean" and isinstance(avg[k], float))[:120],
                )

                # Write metrics history
                with open(metrics_history_path, "a") as f:
                    f.write(json.dumps({"step": step, **avg}) + "\n")

                running = {}

            if step % args.save_every == 0:
                ckpt_path = output_dir / f"rl_step_{step}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "ref_model": rl_trainer.ref_model.state_dict(),
                    "rl_trainer_state": rl_trainer.state_dict(),
                    "step": step,
                    "base_checkpoint": str(args.base_checkpoint),
                }, ckpt_path)
                logger.info("Saved: %s", ckpt_path)

            # Early stopping check
            if metrics.get("early_stopped"):
                logger.warning("RL early stopped at step %d (safety guard triggered)", step)
                break

        if metrics.get("early_stopped"):
            break

    # --- Final save ---
    elapsed = time.time() - t0
    logger.info("RL training done: %d steps, %d epochs, %.0fs", step, epoch, elapsed)

    final_metrics = {
        "step": step,
        "reward_mean": running.get("reward_mean", 0),
        "instruction_following": running.get("instruction_following", 0),
        "physical_compliance": running.get("physical_compliance", 0),
        "plain_text_degradation": running.get("plain_text_degradation", 0),
    }
    (output_dir / "rl_metrics.json").write_text(
        json.dumps(final_metrics, indent=2), encoding="utf-8",
    )

    torch.save({
        "model": model.state_dict(),
        "ref_model": rl_trainer.ref_model.state_dict(),
        "rl_trainer_state": rl_trainer.state_dict(),
        "step": step,
        "base_checkpoint": str(args.base_checkpoint),
    }, output_dir / "rl_final.pt")
    logger.info("Final checkpoint: %s", output_dir / "rl_final.pt")


if __name__ == "__main__":
    main()
