#!/usr/bin/env python3
"""Overfit single-utterance test.

Trains the v4 textless model from scratch on ONE cached utterance for ~1500 steps.
If the model can reproduce the training audio at the end, the architecture/loss
pipeline is sound and the issue is convergence/generalization on the full dataset.
If it cannot, there is a fundamental bug.

Usage: .venv/bin/python scripts/overfit_test.py
"""
from __future__ import annotations
import logging
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("overfit")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    N_CODEBOOKS, CONTROL_SLOTS, CONTROL_VOCAB_SIZE, PHONEME_VOCAB_SIZE,
    N_ACTING_TAGS, SAMPLE_RATE,
)
from tmrvc_data.v4_dataset import V4UCLMDataset, v4_collate_fn
from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.models.acting_latent import ActingLatentEncoder, ActingLatentPredictor
from tmrvc_train.trainer import UCLMTrainer
from tmrvc_train.v4_loss import V4LossConfig

CACHE = ROOT / "data" / "cache" / "v4d_overfit"
OUT_DIR = ROOT / "checkpoints" / "v4_overfit"


def _collate_for_trainer(samples, speaker_to_int):
    """Mirror of train_v4_full.py::_collate (key remapping for Trainer)."""
    raw = v4_collate_fn(samples)
    B = raw["codec_tokens_a"].shape[0]
    T = raw["codec_tokens_a"].shape[-1]

    d = {
        "target_a": raw["codec_tokens_a"],
        "target_b": raw["codec_tokens_b"] if raw.get("codec_tokens_b") is not None else torch.full((B, CONTROL_SLOTS, T), -1, dtype=torch.long),
        "speaker_embed": raw.get("speaker_embed", torch.zeros(B, D_SPEAKER)),
        "phoneme_ids": raw.get("phoneme_ids", torch.zeros(B, 1, dtype=torch.long)),
        "phoneme_lens": raw.get("phoneme_ids_lengths", torch.ones(B, dtype=torch.long)),
        "physical_targets": raw.get("physical_targets"),
        "physical_observed_mask": raw.get("physical_observed_mask"),
        "physical_confidence": raw.get("physical_confidence"),
        "voice_state_targets": raw.get("physical_targets"),
        "voice_state_observed_mask": raw.get("physical_observed_mask"),
        "voice_state_confidence": raw.get("physical_confidence"),
        "ssl_state": raw.get("ssl_state", torch.zeros(B, T, D_VOICE_STATE_SSL)),
        "bootstrap_alignment": None,
        "speaker_id": torch.tensor(
            [speaker_to_int.get(s, 0) if isinstance(s, str) else 0
             for s in (raw.get("speaker_id") or [""] * B)],
            dtype=torch.long,
        ),
        "language_id": torch.tensor(
            [{"ja": 0, "en": 1, "zh": 2, "ko": 3}.get(l, 0) if isinstance(l, str) else 0
             for l in (raw.get("language") or [0] * B)],
            dtype=torch.long,
        ),
        "utterance_ids": raw.get("utterance_id", [f"unk_{i}" for i in range(B)]),
    }

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

    if d.get("ssl_state") is not None and isinstance(d["ssl_state"], torch.Tensor) and d["ssl_state"].shape[1] >= T_aligned:
        d["ssl_state"] = d["ssl_state"][:, :T_aligned, :]
    else:
        d["ssl_state"] = torch.zeros(B, T_aligned, D_VOICE_STATE_SSL)

    for vs_key in ("voice_state_targets", "voice_state_observed_mask", "voice_state_confidence"):
        if d.get(vs_key) is not None and isinstance(d[vs_key], torch.Tensor) and d[vs_key].dim() == 3:
            vs_t = d[vs_key].permute(0, 2, 1).float()
            vs_t = F.interpolate(vs_t, size=T_aligned, mode='nearest')
            d[vs_key] = vs_t.permute(0, 2, 1).to(d[vs_key].dtype)

    d["lengths"] = torch.full((B,), T_aligned, dtype=torch.long)
    d["source_a_t"] = d["target_a"].clone()
    return d


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = V4UCLMDataset(
        cache_dir=str(CACHE), max_frames=400, min_frames=10,
        use_enriched_transcript=False,
    )
    logger.info("Dataset: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("Empty overfit cache: %s", CACHE)
        sys.exit(1)

    spk_id = dataset.utterances[0].get("speaker_id", "spk0")
    speaker_to_int = {spk_id: 0}

    # Build a single fixed batch (B=2, duplicates of the same utterance)
    fixed_batch = _collate_for_trainer([dataset[0], dataset[0]], speaker_to_int)
    logger.info("Batch: phoneme_ids=%s, target_a=%s", fixed_batch["phoneme_ids"].shape, fixed_batch["target_a"].shape)

    # Model (textless arch matching v4_full)
    model = DisentangledUCLM(
        d_model=D_MODEL, n_layers=8, n_heads=8,
        d_voice_state_explicit=D_VOICE_STATE, d_voice_state_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=1, rvq_vocab_size=4096,
        control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=1,
        acting_tag_vocab_size=N_ACTING_TAGS,
        codec_condition="D",
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.2fM params", n_params / 1e6)

    acting_enc = ActingLatentEncoder().to(device)
    acting_pred = ActingLatentPredictor(d_text=D_MODEL, d_context=D_MODEL).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

    trainer = UCLMTrainer(
        model=model, optimizer=optimizer, device=device,
        tts_mode="pointer", tts_prob=1.0,
        pointer_loss_weight=0.5, progress_loss_weight=0.2,
        boundary_confidence_loss_weight=0.1,
        voice_state_loss_weight=0.0,
        conditioning_dropout_prob=0.0,
        curriculum=None,
        enable_v4_losses=False,
        v4_loss_config=V4LossConfig(),
        acting_latent_encoder=acting_enc,
        acting_latent_predictor=acting_pred,
        bio_constraint_weight=0.0,
        acting_kl_weight=0.0,
        disentanglement_weight=0.0,
        semantic_alignment_weight=0.0,
        use_enriched_transcript=False,
        codec_condition="D",
    )

    n_steps = 1500
    t0 = time.time()
    best_loss_a = float("inf")
    for step in range(1, n_steps + 1):
        losses = trainer.train_step(fixed_batch)
        trainer._global_step = step
        if step % 25 == 0 or step == 1:
            la = float(losses.get("loss_a", torch.tensor(0.0)))
            l_align = float(losses.get("loss_align", torch.tensor(0.0)))
            l = float(losses["loss"])
            elapsed = time.time() - t0
            best_loss_a = min(best_loss_a, la)
            logger.info("step %d/%d | loss=%.3f loss_a=%.3f loss_align=%.3f | best_a=%.3f | %.1fs",
                        step, n_steps, l, la, l_align, best_loss_a, elapsed)

    logger.info("Overfit best loss_a: %.4f", best_loss_a)

    ckpt_path = OUT_DIR / "v4_overfit_final.pt"
    torch.save({
        "model": model.state_dict(),
        "acting_encoder": acting_enc.state_dict(),
        "acting_predictor": acting_pred.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": n_steps,
        "best_loss": best_loss_a,
    }, ckpt_path)
    logger.info("Saved %s", ckpt_path)


if __name__ == "__main__":
    main()
