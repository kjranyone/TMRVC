#!/usr/bin/env python3
"""v4 full training smoke test — 1% sampling.

Generates training cache from raw audio, then runs the full v4 pipeline
using the existing UCLMTrainer with all loss terms enabled.

Usage:
    .venv/bin/python scripts/train_v4_smoke.py [--device cpu] [--steps 200] [--sample-pct 1]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v4_smoke")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    D_ACTING_LATENT, N_ACTING_TAGS, N_CODEBOOKS, CONTROL_SLOTS,
    RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE, PHONEME_VOCAB_SIZE,
    HOP_LENGTH, SAMPLE_RATE, BIO_COVARIANCE_RANK,
    BIO_TRANSITION_PENALTY_WEIGHT,
)


def parse_args():
    p = argparse.ArgumentParser(description="v4 training smoke test")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--sample-pct", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "v4_smoke"))
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--skip-cache-gen", action="store_true")
    return p.parse_args()


# =========================================================================
# Phase 1: Cache generation
# =========================================================================

def discover_raw_audio(sample_pct: float) -> list[Path]:
    all_wavs = list((ROOT / "data" / "raw").rglob("*.wav")) if (ROOT / "data" / "raw").exists() else []
    if not all_wavs:
        return []
    n = max(1, int(len(all_wavs) * sample_pct / 100))
    random.seed(42)
    sampled = random.sample(all_wavs, min(n, len(all_wavs)))
    logger.info("Sampled %d / %d raw audio (%.1f%%)", len(sampled), len(all_wavs), sample_pct)
    return sampled


def generate_cache(wav_paths: list[Path], cache_dir: Path, device: str, max_frames: int) -> int:
    from tmrvc_data.preprocessing import load_and_resample

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Lazy models
    spk_encoder = None
    vs_estimator = None

    n = 0
    for i, wav_path in enumerate(wav_paths):
        utt_id = f"smoke_{i:06d}"
        speaker_id = f"spk_{wav_path.parent.name.replace(' ', '_')[:20]}"
        utt_dir = cache_dir / "smoke" / "train" / speaker_id / utt_id
        if (utt_dir / "meta.json").exists():
            n += 1
            continue

        try:
            waveform, sr = load_and_resample(str(wav_path), target_sr=SAMPLE_RATE)
            if waveform is None:
                continue
            waveform_np = waveform.squeeze().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform).squeeze()
            if waveform_np.ndim == 0 or len(waveform_np) == 0:
                continue

            n_samples = len(waveform_np)
            n_frames = n_samples // HOP_LENGTH
            if n_frames < 20 or n_frames > max_frames:
                continue

            # Codec tokens (synthetic — no checkpoint)
            codec_tokens = np.random.randint(0, RVQ_VOCAB_SIZE, (N_CODEBOOKS, n_frames), dtype=np.int64)

            # Voice state (real DSP if available)
            try:
                if vs_estimator is None:
                    from tmrvc_data.voice_state import VoiceStateEstimator
                    vs_estimator = VoiceStateEstimator(device=device)
                wt = torch.from_numpy(waveform_np).float().unsqueeze(0)
                from tmrvc_data.preprocessing import compute_mel
                mel = compute_mel(wt).to(device)
                f0 = torch.zeros(1, 1, mel.shape[-1], device=torch.device(device))
                vs = vs_estimator.estimate(mel, f0)
                vs = vs.squeeze(0).cpu().numpy()[:n_frames] if isinstance(vs, torch.Tensor) else np.zeros((n_frames, D_VOICE_STATE), dtype=np.float32)
            except Exception:
                vs = np.clip(np.random.randn(n_frames, D_VOICE_STATE) * 0.2 + 0.5, 0, 1).astype(np.float32)

            vs = np.clip(vs, 0, 1).astype(np.float32)
            if vs.shape[0] != n_frames:
                vs = np.resize(vs, (n_frames, D_VOICE_STATE))

            # Speaker embed
            try:
                if spk_encoder is None:
                    from tmrvc_data.speaker import SpeakerEncoder
                    spk_encoder = SpeakerEncoder(device=device)
                wt = torch.from_numpy(waveform_np).float().unsqueeze(0)
                se = spk_encoder.extract(wt, sample_rate=SAMPLE_RATE)
                se = se.cpu().numpy().flatten().astype(np.float32) if isinstance(se, torch.Tensor) else np.zeros(D_SPEAKER, dtype=np.float32)
            except Exception:
                se = np.random.randn(D_SPEAKER).astype(np.float32)
                se /= np.linalg.norm(se) + 1e-8

            # Phonemes
            n_phonemes = max(5, n_frames // 4)
            phoneme_ids = np.random.randint(1, PHONEME_VOCAB_SIZE, (n_phonemes,), dtype=np.int64)

            # Write
            utt_dir.mkdir(parents=True, exist_ok=True)
            np.save(utt_dir / "codec_tokens.npy", codec_tokens)
            np.save(utt_dir / "voice_state.npy", vs)
            np.save(utt_dir / "spk_embed.npy", se)
            np.save(utt_dir / "phoneme_ids.npy", phoneme_ids)
            np.save(utt_dir / "text_suprasegmentals.npy", np.zeros((n_phonemes, 4), dtype=np.float32))
            np.save(utt_dir / "voice_state_targets.npy", vs)
            mask = np.ones((n_frames, D_VOICE_STATE), dtype=bool)
            mask[:, 8:] = False
            np.save(utt_dir / "voice_state_observed_mask.npy", mask)
            conf = np.ones((n_frames, D_VOICE_STATE), dtype=np.float32) * 0.8
            conf[:, 8:] = 0.1
            np.save(utt_dir / "voice_state_confidence.npy", conf)

            with open(utt_dir / "meta.json", "w") as f:
                json.dump({
                    "utterance_id": utt_id, "speaker_id": speaker_id,
                    "n_frames": int(n_frames), "text": "", "language_id": 0,
                    "duration_sec": n_samples / SAMPLE_RATE,
                    "enriched_transcript": "",
                    "supervision_tier": "tier_b",
                    "quality_score": 0.5,
                }, f)
            n += 1
            if (i + 1) % 20 == 0:
                logger.info("Cache: %d / %d", i + 1, len(wav_paths))
        except Exception as e:
            logger.debug("Skip %s: %s", wav_path.name, e)
    logger.info("Cache: %d utterances in %s", n, cache_dir)
    return n


# =========================================================================
# Phase 2: Training
# =========================================================================

def main():
    args = parse_args()
    cache_dir = ROOT / "data" / "cache"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Cache ---
    if not args.skip_cache_gen:
        logger.info("=== Phase 1: Cache generation (%.1f%%) ===", args.sample_pct)
        wavs = discover_raw_audio(args.sample_pct)
        if wavs:
            n = generate_cache(wavs, cache_dir, args.device, args.max_frames)
        else:
            logger.warning("No raw audio, using synthetic data")
            n = generate_cache([], cache_dir, args.device, args.max_frames)
        if n == 0:
            logger.error("No cache data. Exiting.")
            sys.exit(1)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Phase 2: Build model + trainer ---
    logger.info("=== Phase 2: Model + Trainer ===")
    device = args.device

    from tmrvc_train.models.uclm_model import DisentangledUCLM
    from tmrvc_train.models.acting_latent import ActingLatentEncoder, ActingLatentPredictor
    from tmrvc_train.models.biological_constraints import BiologicalConstraintRegularizer
    from tmrvc_train.trainer import UCLMTrainer, CurriculumScheduler
    from tmrvc_train.v4_loss import V4LossConfig

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
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.2fM params", n_params / 1e6)

    # v4 modules
    acting_enc = ActingLatentEncoder().to(device)
    acting_pred = ActingLatentPredictor().to(device)

    # All params
    all_params = list(model.parameters()) + list(acting_enc.parameters()) + list(acting_pred.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)

    trainer = UCLMTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        tts_mode="pointer",
        tts_prob=1.0,  # TTS only for smoke test (no VC source data)
        curriculum=None,  # No curriculum — TTS only, all steps use same config
        pointer_loss_weight=0.5,
        progress_loss_weight=0.2,
        boundary_confidence_loss_weight=0.1,
        voice_state_loss_weight=0.1,
        conditioning_dropout_prob=0.1,
        # v4 params
        enable_v4_losses=True,
        v4_loss_config=V4LossConfig(),
        acting_latent_encoder=acting_enc,
        acting_latent_predictor=acting_pred,
        bio_constraint_weight=1.0,
        acting_kl_weight=0.01,
        disentanglement_weight=0.1,
        semantic_alignment_weight=0.5,
    )

    # --- Phase 3: Dataloader (V4UCLMDataset) ---
    logger.info("=== Phase 3: Dataloader ===")
    from tmrvc_data.v4_dataset import V4UCLMDataset, v4_collate_fn

    dataset = V4UCLMDataset(
        cache_dir=str(cache_dir),
        max_frames=args.max_frames,
        min_frames=10,
        use_enriched_transcript=True,
        enriched_transcript_prob=0.5,
    )
    logger.info("Dataset: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("Empty dataset. Exiting.")
        sys.exit(1)

    def _collate_to_dict(samples):
        """Collate V4UCLMDataset samples and map keys to Trainer expectations."""
        raw = v4_collate_fn(samples)
        B = raw["codec_tokens_a"].shape[0] if raw.get("codec_tokens_a") is not None else 1
        T = raw["codec_tokens_a"].shape[-1] if raw.get("codec_tokens_a") is not None else 1

        d = {
            "target_a": raw["codec_tokens_a"],
            "target_b": raw["codec_tokens_b"] if raw.get("codec_tokens_b") is not None else torch.zeros(B, CONTROL_SLOTS, T, dtype=torch.long),
            "speaker_embed": raw.get("speaker_embed", torch.zeros(B, D_SPEAKER)),
            "phoneme_ids": raw.get("phoneme_ids", torch.zeros(B, 1, dtype=torch.long)),
            "phoneme_lens": raw.get("phoneme_ids_lengths", torch.ones(B, dtype=torch.long)),
            "voice_state_targets": raw.get("physical_targets"),
            "voice_state_observed_mask": raw.get("physical_observed_mask"),
            "voice_state_confidence": raw.get("physical_confidence"),
            # New v4 fields passed through
            "enriched_phoneme_ids": raw.get("enriched_phoneme_ids"),
            "use_enriched": raw.get("use_enriched"),
            "supervision_tier": raw.get("supervision_tier"),
            # Generate missing keys
            "ssl_state": torch.zeros(B, T, D_VOICE_STATE_SSL),
            "speaker_id": torch.zeros(B, dtype=torch.long),
            "language_id": torch.zeros(B, dtype=torch.long),
            "utterance_ids": raw.get("utterance_id", [f"unk_{i}" for i in range(B)]),
        }

        # explicit_state: use physical_targets as voice state
        if raw.get("physical_targets") is not None:
            d["explicit_state"] = raw["physical_targets"]
        else:
            d["explicit_state"] = torch.zeros(B, T, D_VOICE_STATE)

        # Frame lengths from codec tokens
        d["lengths"] = torch.tensor([T] * B, dtype=torch.long)

        # VC source (clone of target_a)
        d["source_a_t"] = d["target_a"].clone()

        return d

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_to_dict,
        drop_last=True,
    )

    # --- Phase 4: Train ---
    logger.info("=== Phase 4: Training (%d steps) ===", args.steps)
    logger.info("Loss: codec + control + pointer + physical(12-D) + acting_latent + bio_constraints + disentanglement + semantic_align")

    step = 0
    epoch = 0
    running = {}
    best_loss = float("inf")
    t0 = time.time()

    while step < args.steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.steps:
                break

            metrics = trainer.train_step(batch)
            step += 1

            # Accumulate metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + v

            # Log
            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: v / args.log_every for k, v in running.items()}
                total = avg.get("loss_total", avg.get("loss", 0))

                parts = []
                for k in sorted(avg):
                    if k.startswith("loss") and avg[k] != 0:
                        parts.append(f"{k}={avg[k]:.4f}")

                logger.info(
                    "step %d/%d | total=%.4f | %.1f steps/s | stage=%s | %s",
                    step, args.steps, total, step / elapsed,
                    avg.get("curriculum_stage", "?"),
                    " ".join(parts[:8]),
                )

                if total < best_loss and total > 0:
                    best_loss = total
                running = {}

            # Save
            if step % args.save_every == 0:
                ckpt = {
                    "model": model.state_dict(),
                    "acting_encoder": acting_enc.state_dict(),
                    "acting_predictor": acting_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }
                path = output_dir / f"v4_step_{step}.pt"
                torch.save(ckpt, path)
                logger.info("Saved: %s", path)

    # --- Summary ---
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done: %d steps, %d epochs, %.1fs (%.1f steps/s)",
                step, epoch, elapsed, step / max(elapsed, 1))
    logger.info("Best loss: %.4f", best_loss)

    # Final save
    torch.save({
        "model": model.state_dict(),
        "acting_encoder": acting_enc.state_dict(),
        "acting_predictor": acting_pred.state_dict(),
        "step": step, "best_loss": best_loss,
    }, output_dir / "v4_smoke_final.pt")

    # Gradient check
    n_has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info("Gradient flow: %d / %d params (%.0f%%)", n_has_grad, n_total, n_has_grad / max(n_total, 1) * 100)


if __name__ == "__main__":
    main()
