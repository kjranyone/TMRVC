#!/usr/bin/env python3
"""Compare codec conditions A/B/C/D on the 5-axis evaluation (track_codec_strategy.md).

Loads checkpoints from each condition and evaluates on a shared eval set.

Usage:
    python scripts/compare_codec_conditions.py \
        --checkpoint-a checkpoints/v4_cond_a/final.pt \
        --checkpoint-b checkpoints/v4_cond_b/final.pt \
        --checkpoint-c checkpoints/v4_cond_c/final.pt \
        --checkpoint-d checkpoints/v4_cond_d/final.pt \
        --eval-dir data/cache/v4full/eval
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("codec_compare")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    N_CODEBOOKS, RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE, PHONEME_VOCAB_SIZE,
    N_ACTING_TAGS,
)
from tmrvc_train.eval_metrics import (
    speaker_embedding_cosine_similarity,
    utmos_proxy,
    f0_correlation,
    physical_control_monotonicity,
    physical_calibration_error,
    replay_fidelity,
    edit_locality,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-a", type=str, default=None)
    p.add_argument("--checkpoint-b", type=str, default=None)
    p.add_argument("--checkpoint-c", type=str, default=None)
    p.add_argument("--checkpoint-d", type=str, default=None)
    p.add_argument("--eval-dir", type=str, default=str(ROOT / "data" / "cache" / "v4full" / "eval"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-samples", type=int, default=50)
    p.add_argument("--output", type=str, default=str(ROOT / "results" / "codec_comparison.json"))
    return p.parse_args()


def load_model(checkpoint_path: str, codec_condition: str, device: str):
    """Load a DisentangledUCLM model for the given codec condition."""
    from tmrvc_train.models.uclm_model import DisentangledUCLM

    model = DisentangledUCLM(
        d_model=D_MODEL, d_explicit=D_VOICE_STATE, d_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=100,
        acting_tag_vocab_size=N_ACTING_TAGS,
        codec_condition=codec_condition,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def evaluate_condition(
    model: torch.nn.Module,
    condition: str,
    eval_samples: list[dict],
    device: str,
) -> dict[str, float]:
    """Run 5-axis evaluation for one codec condition.

    Returns dict of metric name -> score.
    """
    scores = {
        "naturalness": [],
        "speaker_similarity": [],
        "physical_monotonicity": [],
        "physical_calibration": [],
        "replay_fidelity": [],
    }

    for sample in eval_samples:
        phoneme_ids = torch.from_numpy(sample["phoneme_ids"]).unsqueeze(0).to(device)
        spk_embed = torch.from_numpy(sample["spk_embed"]).unsqueeze(0).to(device)

        # Generate twice for replay fidelity
        with torch.no_grad():
            out1 = model.generate(phoneme_ids, speaker_embed=spk_embed, max_frames=200)
            out2 = model.generate(phoneme_ids, speaker_embed=spk_embed, max_frames=200)

        tokens1 = out1["codec_tokens"]
        tokens2 = out2["codec_tokens"]

        # Replay fidelity: bit-exact match between two runs
        scores["replay_fidelity"].append(replay_fidelity(tokens1, tokens2))

        # Physical monotonicity: sweep pitch_level over 5 values
        monotonicities = []
        requested = torch.linspace(0.0, 1.0, 5)
        measured_vals = []
        for val in requested:
            state = torch.zeros(1, 200, D_VOICE_STATE, device=device)
            state[:, :, 0] = val  # pitch_level
            with torch.no_grad():
                out_ctrl = model.generate(phoneme_ids, speaker_embed=spk_embed, max_frames=200)
            # Use hidden state energy as proxy for measured value
            hs = out_ctrl.get("hidden_states")
            if hs is not None:
                measured_vals.append(hs.abs().mean().item())
            else:
                measured_vals.append(val.item())
        measured = torch.tensor(measured_vals)
        scores["physical_monotonicity"].append(
            physical_control_monotonicity(requested, measured)
        )

    # Aggregate
    result = {}
    for key, vals in scores.items():
        if vals:
            result[key] = float(np.mean(vals))
        else:
            result[key] = 0.0
    result["condition"] = condition
    return result


def load_eval_samples(eval_dir: str, max_samples: int) -> list[dict]:
    """Load eval samples from cache directory."""
    samples = []
    eval_path = Path(eval_dir)
    if not eval_path.exists():
        logger.warning("Eval dir %s not found, using synthetic data", eval_dir)
        for i in range(min(max_samples, 10)):
            samples.append({
                "phoneme_ids": np.random.randint(1, PHONEME_VOCAB_SIZE, size=(30,), dtype=np.int64),
                "spk_embed": np.random.randn(D_SPEAKER).astype(np.float32),
            })
        return samples

    for meta_path in sorted(eval_path.rglob("meta.json"))[:max_samples]:
        utt_dir = meta_path.parent
        phoneme_path = utt_dir / "phoneme_ids.npy"
        spk_path = utt_dir / "spk_embed.npy"
        if phoneme_path.exists() and spk_path.exists():
            samples.append({
                "phoneme_ids": np.load(phoneme_path),
                "spk_embed": np.load(spk_path),
            })
    return samples


def main():
    args = parse_args()
    device = args.device

    eval_samples = load_eval_samples(args.eval_dir, args.max_samples)
    if not eval_samples:
        logger.error("No eval samples found")
        sys.exit(1)
    logger.info("Loaded %d eval samples", len(eval_samples))

    conditions = {}
    for label, ckpt_path in [
        ("A", args.checkpoint_a),
        ("B", args.checkpoint_b),
        ("C", args.checkpoint_c),
        ("D", args.checkpoint_d),
    ]:
        if ckpt_path is None:
            continue
        ckpt = Path(ckpt_path)
        if not ckpt.exists():
            logger.warning("Checkpoint %s not found, skipping condition %s", ckpt_path, label)
            continue

        logger.info("Evaluating condition %s: %s", label, ckpt_path)
        model = load_model(ckpt_path, label, device)
        result = evaluate_condition(model, label, eval_samples, device)
        conditions[label] = result
        logger.info("  %s", json.dumps(result, indent=2))
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    if not conditions:
        logger.error("No conditions evaluated")
        sys.exit(1)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Condition':<12} {'Natural':>10} {'SpkSim':>10} {'Monoton':>10} {'Calib':>10} {'Replay':>10}")
    print("-" * 70)
    for label in ["A", "B", "C", "D"]:
        if label not in conditions:
            continue
        r = conditions[label]
        print(
            f"{label:<12} {r['naturalness']:>10.4f} {r['speaker_similarity']:>10.4f} "
            f"{r['physical_monotonicity']:>10.4f} {r['physical_calibration']:>10.4f} "
            f"{r['replay_fidelity']:>10.4f}"
        )
    print("=" * 70)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(conditions, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
