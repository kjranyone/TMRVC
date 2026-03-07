#!/usr/bin/env python3
"""Latency benchmark for UCLM v3 pointer-based TTS inference.

Usage:
    python scripts/benchmark_latency.py --device cpu
    python scripts/benchmark_latency.py --device cuda --warmup 5 --iterations 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch


def benchmark_pointer_step(device: str, warmup: int = 3, iterations: int = 10) -> dict:
    """Benchmark a single pointer inference step."""
    from tmrvc_train.models import DisentangledUCLM

    d_model = 256
    model = DisentangledUCLM(d_model=d_model).to(device).eval()

    # Simulate single-frame inputs
    content_features = torch.randn(1, 1, d_model, device=device)
    b_ctx = torch.zeros(1, 4, 1, dtype=torch.long, device=device)
    speaker_embed = torch.randn(1, 192, device=device)
    state_cond = torch.randn(1, 1, d_model, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_streaming(
                content_features=content_features,
                b_ctx=b_ctx,
                speaker_embed=speaker_embed,
                state_cond=state_cond,
                cfg_scale=1.0,
            )
    if device == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.forward_streaming(
                content_features=content_features,
                b_ctx=b_ctx,
                speaker_embed=speaker_embed,
                state_cond=state_cond,
                cfg_scale=1.0,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "device": device,
        "iterations": iterations,
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "budget_10ms": statistics.mean(times) < 10.0,
    }


def benchmark_full_tts(device: str, num_phonemes: int = 20, warmup: int = 2, iterations: int = 5) -> dict:
    """Benchmark full TTS generation with pointer mode."""
    from tmrvc_core.dialogue_types import StyleParams
    from tmrvc_serve.uclm_engine import UCLMEngine

    engine = UCLMEngine(device=device)
    # Skip if no checkpoint available
    try:
        engine.load_models()
    except Exception:
        return {"skipped": True, "reason": "No checkpoint available"}

    phonemes = torch.randint(1, 50, (1, num_phonemes))
    speaker = torch.randn(1, 192)
    style = StyleParams.neutral()

    # Warmup
    for _ in range(warmup):
        engine.tts(phonemes=phonemes, speaker_embed=speaker, style=style)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        audio, metrics = engine.tts(phonemes=phonemes, speaker_embed=speaker, style=style)
        times.append((time.perf_counter() - t0) * 1000)

    audio_dur_ms = audio.shape[-1] / 24.0  # 24kHz
    mean_gen = statistics.mean(times)

    return {
        "device": device,
        "num_phonemes": num_phonemes,
        "iterations": iterations,
        "mean_gen_ms": mean_gen,
        "audio_duration_ms": audio_dur_ms,
        "rtf": mean_gen / max(audio_dur_ms, 0.01),
    }


def main():
    parser = argparse.ArgumentParser(description="UCLM v3 latency benchmark")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    print(f"=== UCLM v3 Latency Benchmark (device={args.device}) ===\n")

    print("1. Single pointer step benchmark:")
    step_results = benchmark_pointer_step(args.device, args.warmup, args.iterations)
    for k, v in step_results.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")

    print(f"\n   Within 10ms budget: {'YES' if step_results['budget_10ms'] else 'NO'}")

    print("\n2. Full TTS generation benchmark:")
    tts_results = benchmark_full_tts(args.device, warmup=args.warmup, iterations=args.iterations)
    for k, v in tts_results.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.3f}")
        else:
            print(f"   {k}: {v}")


if __name__ == "__main__":
    main()
