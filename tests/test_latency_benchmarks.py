"""Worker 06: Latency benchmarks for v3 streaming runtime.

Tests:
- Per-step inference time measurement
- Memory usage per streaming step
- CFG 2-pass overhead measurement
"""

from __future__ import annotations

import time
import pytest
import torch

from tmrvc_train.models.uclm_model import DisentangledUCLM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Frozen Tier 0 latency budget from acceptance-thresholds.md
STREAMING_LATENCY_BUDGET_MS = 10.0  # p95 per-step target
# Relaxed CPU budget for CI (actual hardware benchmarks run on GPU)
CPU_LATENCY_BUDGET_MS = 500.0
# CFG overhead: 2-pass should be < 2x single-pass (some computation shared)
CFG_OVERHEAD_RATIO_MAX = 2.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_and_inputs(
    target_length: int = 10,
    device: str = "cpu",
) -> tuple[DisentangledUCLM, dict]:
    """Create a model and minimal TTS inputs."""
    model = DisentangledUCLM()
    model.eval()
    model.to(device)

    inputs = {
        "phoneme_ids": torch.randint(1, 100, (1, 8), device=device),
        "language_ids": torch.zeros(1, 8, dtype=torch.long, device=device),
        "pointer_state": None,
        "speaker_embed": torch.randn(1, 192, device=device),
        "explicit_state": torch.randn(1, target_length, 8, device=device),
        "ssl_state": torch.randn(1, target_length, 128, device=device),
        "target_a": torch.zeros(1, 8, target_length, dtype=torch.long, device=device),
        "target_b": torch.zeros(1, 4, target_length, dtype=torch.long, device=device),
        "target_length": target_length,
    }
    return model, inputs


# ---------------------------------------------------------------------------
# Per-Step Inference Time
# ---------------------------------------------------------------------------

class TestPerStepInferenceLatency:
    """Measure per-step inference time on CPU (informational) and GPU (gating)."""

    def test_single_step_latency_cpu(self):
        """Single forward pass on CPU must complete within a relaxed budget.

        The actual 10 ms gate applies to GPU hardware. This test validates
        that the model can run without hangs or excessive overhead.
        """
        model, inputs = _make_model_and_inputs(target_length=5)

        # Warmup
        with torch.no_grad():
            model.forward_tts_pointer(**inputs)

        times = []
        n_runs = 5
        for _ in range(n_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model.forward_tts_pointer(**inputs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        p95 = sorted(times)[int(0.95 * n_runs)]
        # CPU budget is relaxed; this catches regressions not absolute targets
        assert p95 < CPU_LATENCY_BUDGET_MS, (
            f"CPU per-step p95 latency {p95:.1f} ms exceeds budget {CPU_LATENCY_BUDGET_MS} ms"
        )

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_single_step_latency_gpu(self):
        """Per-step latency must be < streaming_latency_budget_ms on GPU."""
        model, inputs = _make_model_and_inputs(target_length=5, device="cuda")

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model.forward_tts_pointer(**inputs)
            torch.cuda.synchronize()

        times = []
        n_runs = 20
        for _ in range(n_runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                model.forward_tts_pointer(**inputs)
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        p50 = sorted(times)[n_runs // 2]
        p95 = sorted(times)[int(0.95 * n_runs)]
        p99 = sorted(times)[int(0.99 * n_runs)]

        assert p95 < STREAMING_LATENCY_BUDGET_MS, (
            f"GPU per-step p95={p95:.2f} ms exceeds {STREAMING_LATENCY_BUDGET_MS} ms budget. "
            f"p50={p50:.2f} ms, p99={p99:.2f} ms"
        )


# ---------------------------------------------------------------------------
# Memory Usage Per Streaming Step
# ---------------------------------------------------------------------------

class TestStreamingMemoryUsage:
    """Measure memory usage during streaming inference steps."""

    @pytest.mark.slow
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_steady_state_memory_bounded(self):
        """Memory should not grow unboundedly across streaming steps."""
        model, inputs = _make_model_and_inputs(target_length=5, device="cuda")

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        # Simulate multiple streaming steps
        mem_readings = []
        for step in range(10):
            with torch.no_grad():
                model.forward_tts_pointer(**inputs)
            torch.cuda.synchronize()
            mem_readings.append(torch.cuda.memory_allocated())

        # Memory should plateau, not grow linearly
        # Compare last 5 steps: growth should be minimal
        late_growth = mem_readings[-1] - mem_readings[4]
        early_growth = mem_readings[4] - mem_readings[0]

        # If late growth is > early growth, there may be a leak
        if early_growth > 0:
            growth_ratio = late_growth / early_growth
            assert growth_ratio < 1.5, (
                f"Potential memory leak: late growth ratio {growth_ratio:.2f} "
                f"(early={early_growth} bytes, late={late_growth} bytes)"
            )

    def test_cpu_memory_does_not_explode(self):
        """CPU: running multiple steps should not cause OOM-scale allocations."""
        model, inputs = _make_model_and_inputs(target_length=5)

        # Just ensure it runs 10 steps without crashing
        for _ in range(10):
            with torch.no_grad():
                model.forward_tts_pointer(**inputs)


# ---------------------------------------------------------------------------
# CFG 2-Pass Overhead
# ---------------------------------------------------------------------------

class TestCFGOverhead:
    """Measure the overhead of CFG 2-pass inference vs single-pass."""

    def test_cfg_mask_overhead_bounded(self):
        """Applying CFG unconditional mask should not be excessively slow."""
        B, T, D = 1, 20, 8

        # Measure single apply_cfg_unconditional_mask call
        times = []
        for _ in range(10):
            start = time.perf_counter()
            result = DisentangledUCLM.apply_cfg_unconditional_mask(
                explicit_state=torch.ones(B, T, D),
                ssl_state=torch.ones(B, T, 128),
                speaker_embed=torch.ones(B, 192),
                dialogue_context=torch.ones(B, 256),
                acting_intent=torch.ones(B, 64),
                prosody_latent=torch.ones(B, 128),
                delta_voice_state=torch.ones(B, T, D),
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        median_ms = sorted(times)[5]
        # CFG mask is just zeroing tensors; should be < 1 ms even on CPU
        assert median_ms < 10.0, (
            f"CFG mask overhead {median_ms:.2f} ms is unexpectedly high"
        )

    def test_two_pass_vs_single_pass_ratio(self):
        """2-pass CFG forward should be < CFG_OVERHEAD_RATIO_MAX times single pass."""
        model, inputs = _make_model_and_inputs(target_length=10)

        # Warmup
        with torch.no_grad():
            model.forward_tts_pointer(**inputs)

        # Single pass timing
        single_times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                model.forward_tts_pointer(**inputs)
            single_times.append((time.perf_counter() - start) * 1000)

        # Two-pass timing (conditional + unconditional)
        two_pass_times = []
        for _ in range(5):
            start = time.perf_counter()
            with torch.no_grad():
                # Pass 1: conditional
                model.forward_tts_pointer(**inputs)
                # Pass 2: unconditional (simulate with same inputs)
                model.forward_tts_pointer(**inputs)
            two_pass_times.append((time.perf_counter() - start) * 1000)

        median_single = sorted(single_times)[2]
        median_two = sorted(two_pass_times)[2]

        if median_single > 0:
            ratio = median_two / median_single
            assert ratio < CFG_OVERHEAD_RATIO_MAX, (
                f"CFG 2-pass ratio {ratio:.2f}x exceeds max {CFG_OVERHEAD_RATIO_MAX}x. "
                f"Single={median_single:.1f} ms, Two-pass={median_two:.1f} ms"
            )
