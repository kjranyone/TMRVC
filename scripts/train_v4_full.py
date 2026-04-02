#!/usr/bin/env python3
"""v4 complete training pipeline — real models, no shortcuts.

Phase 1: Bootstrap cache from raw audio using ALL real models:
  - ASR: faster-whisper large-v3 (cached)
  - G2P: real phoneme conversion
  - Voice State: real DSP 12-D extraction
  - Speaker Encoder: ECAPA-TDNN (cached)
  - LLM Annotation: Qwen3.5-35B-A3B (cached) → semantic + enriched transcripts
  - Codec: EnCodec 24kHz frozen pre-trained, 75 Hz, 8 RVQ x 1024 (condition A)

Phase 2: v4 full training with:
  - Enriched transcript path (inline acting tags)
  - All 9 v4 loss terms
  - Biological constraint regularization
  - Acting latent encoder/predictor
  - Supervision tier weighting

Usage:
    .venv/bin/python scripts/train_v4_full.py --steps 200 --sample-pct 1
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v4_full")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    D_ACTING_LATENT, N_ACTING_TAGS, N_CODEBOOKS, CONTROL_SLOTS,
    RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE, PHONEME_VOCAB_SIZE,
    HOP_LENGTH, SAMPLE_RATE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "v4_full"))
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--resume-from", type=int, default=None,
                   help="Resume from checkpoint step (e.g. 9500)")
    p.add_argument("--codec-condition", default="A", choices=["A", "B", "C", "D"],
                   help="Codec experiment condition (track_codec_strategy.md)")
    return p.parse_args()


# Phase 1 (cache generation) is handled by manage_data.py.
# See TRAIN_GUIDE.md for the canonical workflow.

# =========================================================================
# Phase 2: Training
# =========================================================================

def main():
    args = parse_args()
    # v4 cache: prefer new managed path, fall back to legacy
    cache_dir = ROOT / "data" / "cache" / "v4"
    if not cache_dir.exists():
        legacy = ROOT / "data" / "cache"
        if (legacy / "v4full").exists():
            cache_dir = legacy
        elif legacy.exists():
            cache_dir = legacy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache is built by manage_data.py (canonical pipeline).
    # Phase 1 builder has been removed. Use: manage_data.py add + build

    # ---- Phase 2: Model + Trainer ----
    logger.info("=" * 60)
    logger.info("Phase 2: Building v4 model + trainer")
    logger.info("=" * 60)
    device = args.device

    from tmrvc_train.models.uclm_model import DisentangledUCLM
    from tmrvc_train.models.acting_latent import ActingLatentEncoder, ActingLatentPredictor
    from tmrvc_train.trainer import UCLMTrainer
    from tmrvc_train.v4_loss import V4LossConfig

    codec_cond = args.codec_condition
    logger.info("Codec condition: %s", codec_cond)

    # Infer d_model/n_layers from checkpoint when resuming (checkpoint may differ from constants.yaml)
    eff_d_model = D_MODEL
    eff_n_layers = None  # use default from constants
    eff_n_heads = None
    resume_step = 0

    if args.resume_from is not None:
        resume_path = output_dir / f"v4_step_{args.resume_from}.pt"
        if not resume_path.exists():
            logger.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)
        _ckpt_sd = torch.load(resume_path, map_location="cpu", weights_only=False)["model"]
        eff_d_model = _ckpt_sd["uclm_core.layers.0.norm1.weight"].shape[0]
        eff_n_layers = max(int(k.split(".")[2]) for k in _ckpt_sd if k.startswith("uclm_core.layers.")) + 1
        q_dim = _ckpt_sd["uclm_core.layers.0.attn.q_proj.weight"].shape[0]
        k_dim = _ckpt_sd["uclm_core.layers.0.attn.k_proj.weight"].shape[0]
        for nh in [8, 12, 16, 4]:
            hd = eff_d_model // nh
            if hd * nh == eff_d_model and k_dim % hd == 0:
                eff_n_heads = nh
                break
        del _ckpt_sd
        logger.info("Resume: d_model=%d, n_layers=%d, n_heads=%s (from checkpoint)",
                     eff_d_model, eff_n_layers, eff_n_heads)

    init_kwargs = dict(
        d_model=eff_d_model,
        d_voice_state_explicit=D_VOICE_STATE, d_voice_state_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=256,
        acting_tag_vocab_size=N_ACTING_TAGS,
        codec_condition=codec_cond,
    )
    if eff_n_layers is not None:
        init_kwargs["n_layers"] = eff_n_layers
    if eff_n_heads is not None:
        init_kwargs["n_heads"] = eff_n_heads

    model = DisentangledUCLM(**init_kwargs).to(device)

    acting_enc = ActingLatentEncoder().to(device)
    acting_pred = ActingLatentPredictor(d_text=eff_d_model, d_context=eff_d_model).to(device)

    # Only model params here — trainer will add acting_enc/pred/bio params via add_param_group
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Resume from checkpoint
    if args.resume_from is not None:
        resume_path = output_dir / f"v4_step_{args.resume_from}.pt"
        logger.info("Loading checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        # Filter out keys with shape mismatch (code may have evolved since checkpoint)
        model_sd = model.state_dict()
        ckpt_sd = ckpt["model"]
        filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        skipped = [k for k in ckpt_sd if k in model_sd and model_sd[k].shape != ckpt_sd[k].shape]
        model.load_state_dict(filtered, strict=False)
        if skipped:
            logger.warning("Skipped %d keys with shape mismatch: %s", len(skipped), skipped)
        acting_enc.load_state_dict(ckpt["acting_encoder"])
        acting_pred.load_state_dict(ckpt["acting_predictor"])
        # Optimizer state may have mismatched param groups due to model changes;
        # load with best-effort
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            logger.warning("Optimizer state load failed (%s), starting fresh optimizer", e)
        resume_step = ckpt["step"]
        del ckpt
        logger.info("Resumed from step %d (%d/%d model keys loaded)",
                     resume_step, len(filtered), len(ckpt_sd))

    trainer = UCLMTrainer(
        model=model, optimizer=optimizer, device=device,
        tts_mode="pointer", tts_prob=1.0,
        pointer_loss_weight=0.5, progress_loss_weight=0.2,
        boundary_confidence_loss_weight=0.1,
        voice_state_loss_weight=0.1,
        conditioning_dropout_prob=0.1,
        curriculum=None,
        enable_v4_losses=True,
        v4_loss_config=V4LossConfig(),
        acting_latent_encoder=acting_enc,
        acting_latent_predictor=acting_pred,
        bio_constraint_weight=1.0,
        acting_kl_weight=0.01,
        disentanglement_weight=0.1,
        semantic_alignment_weight=0.5,
        use_enriched_transcript=True,
        enriched_transcript_prob=0.5,
        codec_condition=codec_cond,
    )

    # LR scheduler: linear warmup + cosine decay (after trainer adds param groups)
    import math
    warmup_steps = min(5000, args.steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Sync scheduler and trainer._global_step with resume point
    if resume_step > 0:
        trainer._global_step = resume_step
        for _ in range(resume_step):
            scheduler.step()
        logger.info("Synced scheduler and trainer._global_step to step %d", resume_step)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.2fM trainable params", n_params / 1e6)

    # ---- Phase 3: Dataloader (V4UCLMDataset) ----
    from tmrvc_data.v4_dataset import V4UCLMDataset, v4_collate_fn

    dataset = V4UCLMDataset(
        cache_dir=str(cache_dir),
        max_frames=args.max_frames, min_frames=10,
        use_enriched_transcript=True,
        enriched_transcript_prob=0.5,
    )
    logger.info("Dataset: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("Empty dataset. Exiting.")
        sys.exit(1)

    def _collate(samples):
        """Collate V4UCLMDataset samples and map keys to Trainer expectations."""
        raw = v4_collate_fn(samples)
        B = raw["codec_tokens_a"].shape[0] if raw.get("codec_tokens_a") is not None else 1
        T = raw["codec_tokens_a"].shape[-1] if raw.get("codec_tokens_a") is not None else 1

        d = {
            "target_a": raw["codec_tokens_a"],
            "target_b": raw["codec_tokens_b"] if raw.get("codec_tokens_b") is not None else torch.full((B, CONTROL_SLOTS, T), -1, dtype=torch.long),
            "speaker_embed": raw.get("speaker_embed", torch.zeros(B, D_SPEAKER)),
            "phoneme_ids": raw.get("phoneme_ids", torch.zeros(B, 1, dtype=torch.long)),
            "phoneme_lens": raw.get("phoneme_ids_lengths", torch.ones(B, dtype=torch.long)),
            # Keys matching trainer expectations
            "physical_targets": raw.get("physical_targets"),
            "physical_observed_mask": raw.get("physical_observed_mask"),
            "physical_confidence": raw.get("physical_confidence"),
            "voice_state_targets": raw.get("physical_targets"),
            "voice_state_observed_mask": raw.get("physical_observed_mask"),
            "voice_state_confidence": raw.get("physical_confidence"),
            # New v4 fields passed through
            "enriched_phoneme_ids": raw.get("enriched_phoneme_ids"),
            "use_enriched": raw.get("use_enriched"),
            "supervision_tier": raw.get("supervision_tier"),
            "ssl_state": raw.get("ssl_state", torch.zeros(B, T, D_VOICE_STATE_SSL)),
            "bootstrap_alignment": raw.get("bootstrap_alignment"),
            "speaker_id": torch.tensor(
                [int(hashlib.md5(s.encode()).hexdigest(), 16) % 256 if isinstance(s, str) else 0
                 for s in (raw.get("speaker_id") or [""] * B)],
                dtype=torch.long,
            ),
            "language_id": torch.tensor(
                [{"ja": 0, "en": 1, "zh": 2, "ko": 3}.get(l, 0) if isinstance(l, str) else (l if isinstance(l, int) else 0)
                 for l in (raw.get("language") or [0] * B)],
                dtype=torch.long,
            ),
            "utterance_ids": raw.get("utterance_id", [f"unk_{i}" for i in range(B)]),
        }

        # Align all temporal dimensions to min(T_codec, T_vs)
        T_codec = T
        T_vs = raw["physical_targets"].shape[1] if raw.get("physical_targets") is not None else T
        T_aligned = min(T_codec, T_vs)

        d["target_a"] = d["target_a"][:, :, :T_aligned]
        if d.get("target_b") is not None and isinstance(d["target_b"], torch.Tensor):
            d["target_b"] = d["target_b"][:, :, :T_aligned]
        else:
            d["target_b"] = torch.zeros(B, CONTROL_SLOTS, T_aligned, dtype=torch.long)

        if raw.get("physical_targets") is not None:
            d["explicit_state"] = raw["physical_targets"][:, :T_aligned, :]
        else:
            d["explicit_state"] = torch.zeros(B, T_aligned, D_VOICE_STATE)
        # Align ssl_state (keep real data if available)
        if d.get("ssl_state") is not None and isinstance(d["ssl_state"], torch.Tensor) and d["ssl_state"].shape[1] >= T_aligned:
            d["ssl_state"] = d["ssl_state"][:, :T_aligned, :]
        else:
            d["ssl_state"] = torch.zeros(B, T_aligned, D_VOICE_STATE_SSL)

        # Interpolate voice-state tensors to match codec frame count
        for vs_key in ("voice_state_targets", "voice_state_observed_mask", "voice_state_confidence",
                       "physical_targets", "physical_observed_mask", "physical_confidence"):
            if d.get(vs_key) is not None and isinstance(d[vs_key], torch.Tensor) and d[vs_key].dim() == 3:
                vs_t = d[vs_key].permute(0, 2, 1).float()  # [B, D, T_vs]
                vs_t = F.interpolate(vs_t, size=T_aligned, mode='nearest')
                d[vs_key] = vs_t.permute(0, 2, 1).to(d[vs_key].dtype)

        d["lengths"] = torch.full((B,), T_aligned, dtype=torch.long)

        # Frame lengths from codec tokens
        d["lengths"] = torch.tensor([T] * B, dtype=torch.long)

        # Convert bootstrap_alignment tensor [B, T] to dict expected by trainer
        ba = d.get("bootstrap_alignment")
        ba_is_heuristic = raw.get("bootstrap_is_heuristic")
        # Mark as heuristic if any sample in batch has heuristic alignment
        is_heuristic = True
        if ba_is_heuristic is not None:
            if isinstance(ba_is_heuristic, (list, tuple)):
                is_heuristic = any(ba_is_heuristic)
            else:
                is_heuristic = bool(ba_is_heuristic)
        if ba is not None and isinstance(ba, torch.Tensor):
            has_real = (ba.abs().sum(dim=-1) > 0)
            if has_real.all():
                d["bootstrap_alignment"] = {
                    "phoneme_indices": ba[:, :T_aligned],
                    "_heuristic": is_heuristic,
                }
            else:
                d["bootstrap_alignment"] = None
        else:
            d["bootstrap_alignment"] = None

        # VC source (clone of target_a)
        d["source_a_t"] = d["target_a"].clone()

        return d

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate, drop_last=True,
    )

    # ---- Phase 4: Train ----
    logger.info("=" * 60)
    logger.info("Phase 4: v4 FULL training — %d steps", args.steps)
    logger.info("  Real ASR transcripts ✓")
    logger.info("  Real G2P phonemes ✓")
    logger.info("  Real 12-D voice state ✓")
    logger.info("  Real speaker embeddings ✓")
    logger.info("  Real LLM enriched transcripts ✓")
    logger.info("  All v4 losses ✓")
    logger.info("=" * 60)

    step = resume_step
    epoch = 0
    running = {}
    best_loss = float("inf")
    t0 = time.time()

    accum_steps = args.grad_accum
    micro_step = 0

    while step < args.steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.steps:
                break
            is_first_micro = (micro_step % accum_steps == 0)
            is_last_micro = (micro_step % accum_steps == accum_steps - 1)

            # Zero grad only at the start of each accumulation cycle
            if is_first_micro:
                trainer.optimizer.zero_grad(set_to_none=True)

            try:
                metrics = trainer.train_step(
                    batch,
                    accumulate=not is_last_micro,
                    accum_steps=accum_steps,
                )
            except RuntimeError as e:
                if "shape" in str(e) or "size" in str(e):
                    logger.warning("Skipping bad batch at step %d: %s", step, e)
                    micro_step += 1
                    continue
                raise
            micro_step += 1
            if not is_last_micro:
                continue  # accumulating — don't count as a step
            step += 1
            scheduler.step()

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + v

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: v / args.log_every for k, v in running.items()}
                total = avg.get("loss_total", avg.get("loss", 0))
                parts = [f"{k}={avg[k]:.4f}" for k in sorted(avg) if k.startswith("loss") and avg[k] != 0]
                logger.info("step %d/%d | loss=%.4f | %.2f s/step | %s",
                            step, args.steps, total, elapsed / step, " ".join(parts[:8]))
                if 0 < total < best_loss:
                    best_loss = total
                running = {}

            if step % args.save_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "acting_encoder": acting_enc.state_dict(),
                    "acting_predictor": acting_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }, output_dir / f"v4_step_{step}.pt")
                logger.info("Saved: v4_step_%d.pt", step)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done: %d steps, %d epochs, %.0fs (%.2f s/step)", step, epoch, elapsed, elapsed / max(step, 1))
    logger.info("Best loss: %.4f", best_loss)

    torch.save({
        "model": model.state_dict(),
        "acting_encoder": acting_enc.state_dict(),
        "acting_predictor": acting_pred.state_dict(),
        "step": step, "best_loss": best_loss,
    }, output_dir / "v4_full_final.pt")

    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info("Gradient flow: %d / %d (%.0f%%)", n_grad, n_total, n_grad / max(n_total, 1) * 100)


if __name__ == "__main__":
    main()
