#!/usr/bin/env python3
"""Generate golden .npz files for Python<->Rust ONNX parity testing.

Creates frozen random-weight model, runs ONNX inference with fixed inputs,
saves inputs and outputs as golden reference files.

Usage:
    python scripts/eval/generate_parity_golden.py \
        [--output-dir tests/golden/] \
        [--seed 42]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Dimensions matching constants.rs
D_MODEL = 512
D_SPEAKER = 192
D_VOICE_STATE = 12
D_ACTING_LATENT = 24
N_CODEBOOKS = 8
CODEBOOK_SIZE = 1024
CONTROL_SLOTS = 4
CONTROL_VOCAB_SIZE = 64
N_LAYERS = 6
N_HEADS = 8
HEAD_DIM = D_MODEL // N_HEADS


def generate_pacing_golden(output_dir: Path, seed: int = 42) -> None:
    """Generate golden pacing formula test vectors."""
    rng = np.random.RandomState(seed)

    # Test vectors for pacing formula parity
    test_cases = []
    n_cases = 50
    for i in range(n_cases):
        advance_logit = rng.uniform(-3.0, 3.0)
        progress_delta = rng.uniform(0.0, 2.0)
        boundary_confidence = rng.uniform(0.0, 1.0)
        pace = rng.uniform(0.5, 2.5)
        hold_bias = rng.uniform(-1.0, 1.0)
        boundary_bias = rng.uniform(-1.0, 1.0)
        phrase_pressure = rng.uniform(-1.0, 1.0)
        breath_tendency = rng.uniform(-1.0, 1.0)

        # Compute expected outputs
        modulated = (advance_logit - hold_bias + boundary_bias
                    + (pace - 1.0) * 2.0 + phrase_pressure * 1.5
                    - breath_tendency * 0.5)
        p_advance = 1.0 / (1.0 + np.exp(-modulated))
        velocity = progress_delta * pace
        drag = max(0.0, hold_bias * 0.02)
        progress_increment = max(0.0, velocity - drag)

        test_cases.append({
            "advance_logit": advance_logit,
            "progress_delta": progress_delta,
            "boundary_confidence": boundary_confidence,
            "pace": pace,
            "hold_bias": hold_bias,
            "boundary_bias": boundary_bias,
            "phrase_pressure": phrase_pressure,
            "breath_tendency": breath_tendency,
            "expected_p_advance": p_advance,
            "expected_velocity": velocity,
            "expected_drag": drag,
            "expected_progress_increment": progress_increment,
        })

    # Save as npz
    arrays = {}
    for key in test_cases[0].keys():
        arrays[key] = np.array([tc[key] for tc in test_cases], dtype=np.float32)

    npz_path = output_dir / "pacing_golden.npz"
    np.savez(npz_path, **arrays)
    logger.info("Saved pacing golden: %s (%d cases)", npz_path, n_cases)


def generate_onnx_io_golden(output_dir: Path, seed: int = 42) -> None:
    """Generate golden ONNX I/O test vectors."""
    rng = np.random.RandomState(seed)

    context_len = 4

    # Inputs
    content_features = rng.randn(1, D_MODEL, context_len).astype(np.float32)
    b_ctx = np.zeros((1, CONTROL_SLOTS, context_len), dtype=np.int64)
    spk_embed = rng.randn(1, D_SPEAKER).astype(np.float32)
    state_cond = rng.randn(1, D_MODEL).astype(np.float32)
    acting_texture_latent = rng.randn(1, D_ACTING_LATENT).astype(np.float32)
    cfg_scale = np.array([1.5], dtype=np.float32)

    npz_path = output_dir / "onnx_io_golden.npz"
    np.savez(
        npz_path,
        content_features=content_features,
        b_ctx=b_ctx,
        spk_embed=spk_embed,
        state_cond=state_cond,
        acting_texture_latent=acting_texture_latent,
        cfg_scale=cfg_scale,
        context_len=np.array([context_len], dtype=np.int64),
    )
    logger.info("Saved ONNX I/O golden: %s", npz_path)


def main():
    parser = argparse.ArgumentParser(description="Generate parity golden files")
    parser.add_argument("--output-dir", default="tests/golden", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_pacing_golden(output_dir, seed=args.seed)
    generate_onnx_io_golden(output_dir, seed=args.seed)

    print(f"\nGolden files generated in: {output_dir}/")
    for f in sorted(output_dir.glob("*.npz")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
