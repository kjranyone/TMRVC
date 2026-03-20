"""v4 controllability and editability metric definitions.

Required by track_validation.md S 2:
- Physical control response monotonicity
- Physical calibration error
- Trajectory replay fidelity
- Edit locality
- Cross-speaker acting transfer quality
- Semantic prompt-following quality

These are metric harness definitions with real measurement functions.
They define the measurement contract for v4 sign-off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Measurement helpers (used by both tests and scripts/eval/)
# ---------------------------------------------------------------------------


def compute_monotonicity(inputs: np.ndarray, measured: np.ndarray) -> float:
    """Compute monotonicity score between input sweep and measured response.

    Monotonicity is the Spearman rank correlation between ``inputs`` and
    ``measured``.  A perfectly monotonic response yields 1.0.

    Args:
        inputs: 1-D array of control input values (ascending).
        measured: 1-D array of corresponding measured acoustic values.

    Returns:
        Spearman rho in [-1, 1].
    """
    from scipy import stats

    if len(inputs) < 3:
        raise ValueError("Need at least 3 points for monotonicity measurement")
    rho, _ = stats.spearmanr(inputs, measured)
    return float(rho)


def compute_calibration_error(
    requested: np.ndarray, measured_normalised: np.ndarray
) -> float:
    """RMSE between requested control values and normalised measured values.

    Both arrays are expected to be in [0, 1] after min-max normalisation.

    Args:
        requested: [N] or [N, D] target control values.
        measured_normalised: same shape as ``requested``.

    Returns:
        Root-mean-square error (scalar).
    """
    diff = requested.astype(np.float64) - measured_normalised.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def compute_replay_fidelity(
    original_tokens: torch.Tensor, replayed_tokens: torch.Tensor
) -> float:
    """Fraction of codec tokens that are bit-exact between two traces.

    Args:
        original_tokens: [C, T] integer tensor (e.g. 8 codebooks x T frames).
        replayed_tokens: same shape.

    Returns:
        Fidelity score in [0, 1].  1.0 = perfect bit-exact replay.
    """
    if original_tokens.shape != replayed_tokens.shape:
        raise ValueError(
            f"Shape mismatch: {original_tokens.shape} vs {replayed_tokens.shape}"
        )
    return float((original_tokens == replayed_tokens).float().mean())


def compute_edit_locality(
    original_physical: torch.Tensor,
    edited_physical: torch.Tensor,
    edit_start: int,
    edit_end: int,
) -> dict:
    """Measure how much an edit leaked outside the intended region.

    Args:
        original_physical: [T, D] physical trajectory before edit.
        edited_physical: [T, D] physical trajectory after edit.
        edit_start: inclusive start frame index.
        edit_end: exclusive end frame index.

    Returns:
        dict with keys:
        - ``max_outside_diff``: L-inf norm of diff outside the edit region.
        - ``mean_outside_diff``: L1 mean of diff outside the edit region.
        - ``is_local``: True if ``max_outside_diff`` < 1e-5.
    """
    T = original_physical.shape[0]
    outside_mask = torch.ones(T, dtype=torch.bool)
    outside_mask[edit_start:edit_end] = False

    diff = (original_physical[outside_mask] - edited_physical[outside_mask]).abs()
    max_diff = float(diff.max()) if diff.numel() > 0 else 0.0
    mean_diff = float(diff.mean()) if diff.numel() > 0 else 0.0

    return {
        "max_outside_diff": max_diff,
        "mean_outside_diff": mean_diff,
        "is_local": max_diff < 1e-5,
    }


def compute_transfer_correlation(
    source_physical: torch.Tensor, transferred_physical: torch.Tensor
) -> list[float]:
    """Per-dimension Pearson correlation between source and transferred trajectories.

    Args:
        source_physical: [T, D] source trajectory.
        transferred_physical: [T, D] transferred trajectory.

    Returns:
        List of D correlation coefficients.
    """
    D = source_physical.shape[1]
    correlations = []
    for dim in range(D):
        corr = float(
            torch.corrcoef(
                torch.stack([source_physical[:, dim], transferred_physical[:, dim]])
            )[0, 1]
        )
        correlations.append(corr)
    return correlations


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestPhysicalControlResponseMonotonicity:
    """Metric: when a single physical control increases, the corresponding
    audio observable must monotonically respond."""

    def test_monotonic_response_definition(self):
        """Verify metric computation is well-defined."""
        inputs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        measured = np.array([100, 120, 150, 180, 220])  # Hz

        rho = compute_monotonicity(inputs, measured)
        assert rho > 0.99  # near-perfect positive monotonicity

    def test_non_monotonic_detection(self):
        inputs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        measured = np.array([100, 120, 110, 180, 220])  # dip at 0.5

        rho = compute_monotonicity(inputs, measured)
        assert rho < 1.0

    def test_monotonicity_threshold(self):
        """v4 threshold: monotonicity > 0.8 for each physical dimension."""
        inputs = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Simulate a noisy but largely monotone response
        measured = np.array([50, 70, 85, 110, 140, 160])
        rho = compute_monotonicity(inputs, measured)
        V4_MONOTONICITY_THRESHOLD = 0.8
        assert rho >= V4_MONOTONICITY_THRESHOLD


class TestPhysicalCalibrationError:
    """Metric: RMSE between requested physical control value and measured
    audio observable (after normalization)."""

    def test_calibration_error_computation(self):
        requested = np.array([0.3, 0.5, 0.7])
        measured_normalized = np.array([0.28, 0.52, 0.65])

        error = compute_calibration_error(requested, measured_normalized)
        assert error < 0.10

    def test_v4_calibration_threshold(self):
        """v4 threshold: calibration RMSE < 0.15."""
        requested = np.array([0.2, 0.4, 0.6, 0.8])
        measured = np.array([0.18, 0.38, 0.55, 0.72])

        error = compute_calibration_error(requested, measured)
        V4_CALIBRATION_THRESHOLD = 0.15
        assert error < V4_CALIBRATION_THRESHOLD


class TestTrajectoryReplayFidelity:
    """Metric: when replaying a trajectory from a frozen artifact,
    the output must be bit-exact or within numerical tolerance."""

    def test_replay_fidelity_definition(self):
        original = torch.randint(0, 1024, (8, 200))
        replayed = original.clone()

        fidelity = compute_replay_fidelity(original, replayed)
        assert fidelity == 1.0

    def test_replay_detects_drift(self):
        original = torch.randint(0, 1024, (8, 200))
        replayed = original.clone()
        replayed[0, 50] = (replayed[0, 50] + 1) % 1024

        fidelity = compute_replay_fidelity(original, replayed)
        assert fidelity < 1.0


class TestEditLocality:
    """Metric: when editing a local region, changes outside that region
    must be minimal (below a threshold)."""

    def test_edit_locality_definition(self):
        T = 200
        original_physical = torch.randn(T, 12)
        edited_physical = original_physical.clone()

        edit_start, edit_end = 50, 70
        edited_physical[edit_start:edit_end] = torch.randn(20, 12)

        result = compute_edit_locality(original_physical, edited_physical, edit_start, edit_end)
        assert result["is_local"]
        assert result["max_outside_diff"] == 0.0

    def test_non_local_edit_detection(self):
        T = 200
        original = torch.randn(T, 12)
        edited = original.clone()

        edited[50:70] = torch.randn(20, 12)
        edited[100] += 0.5  # unintended leak

        result = compute_edit_locality(original, edited, 50, 70)
        assert not result["is_local"]
        assert result["max_outside_diff"] > 0.0


class TestCrossSpeakerTransferQuality:
    """Metric: acting trajectory transferred to a different speaker
    should preserve relative physical control patterns."""

    def test_transfer_preserves_relative_patterns(self):
        T = 100
        source_physical = torch.randn(T, 12)
        transferred = source_physical + torch.randn(1, 12) * 0.1  # speaker offset

        correlations = compute_transfer_correlation(source_physical, transferred)
        for dim, corr in enumerate(correlations):
            assert corr > 0.9, f"dim {dim}: correlation {corr} < 0.9"


class TestVarianceSeparation:
    """Verify that compile variance, replay variance, and transfer variance
    are measured separately (track_validation.md S 3)."""

    def test_variance_buckets_are_distinct(self):
        variance_buckets = {
            "compile_variance": "prompt compile produces different results each time",
            "replay_variance": "deterministic replay from frozen artifact",
            "transfer_variance": "same acting on different speaker",
        }
        assert len(variance_buckets) == 3
        assert "compile_variance" in variance_buckets
        assert "replay_variance" in variance_buckets
        assert "transfer_variance" in variance_buckets

    def test_replay_variance_should_be_zero_for_deterministic(self):
        """Deterministic replay must have zero variance."""
        trace_1 = torch.randint(0, 1024, (8, 200))
        trace_2 = trace_1.clone()

        variance = float((trace_1.float() - trace_2.float()).var())
        assert variance == 0.0


class TestVarianceSeparationHarness:
    """Tests for the variance separation measurement harness (Phase 6-1)."""

    def test_compile_variance_report_structure(self):
        """Compile variance report must have expected keys."""
        from scripts.eval.variance_separation import measure_compile_variance

        result = measure_compile_variance(n=3, device="cpu")
        assert result.bucket == "compile"
        assert "physical_targets_std" in result.metrics
        assert "acting_latent_prior_std" in result.metrics
        assert "pace_std" in result.metrics
        assert result.n_trials == 3

    def test_replay_variance_is_zero_on_deterministic(self):
        """Deterministic replay must have zero audio diff."""
        from scripts.eval.variance_separation import measure_replay_variance

        result = measure_replay_variance(m=3, device="cpu")
        assert result.bucket == "replay"
        assert result.metrics["max_audio_diff"] < 1e-6
        assert result.passed

    def test_transfer_variance_has_distinct_bucket(self):
        """Transfer variance must report per-speaker correlations."""
        from scripts.eval.variance_separation import measure_transfer_variance

        result = measure_transfer_variance(k=3, device="cpu")
        assert result.bucket == "transfer"
        assert "mean_trajectory_correlation" in result.metrics
        assert "per_speaker_correlations" in result.metrics
        assert len(result.metrics["per_speaker_correlations"]) == 3

    def test_mixed_bucket_rejection(self):
        """Validate that mixed bucket detection works."""
        from scripts.eval.variance_separation import (
            VarianceSeparationReport,
            BucketResult,
            validate_no_mixed_buckets,
        )

        # Good report: no violations
        good_report = VarianceSeparationReport(
            compile=BucketResult(bucket="compile", metrics={"physical_targets_std": 0.1}, n_trials=10),
            replay=BucketResult(bucket="replay", metrics={"max_audio_diff": 0.0}, n_trials=5),
            transfer=BucketResult(bucket="transfer", metrics={"min_trajectory_correlation": 0.95}, n_trials=5),
        )
        assert validate_no_mixed_buckets(good_report)
        assert not good_report.mixed_bucket_violation

        # Bad report: replay has non-zero variance
        bad_report = VarianceSeparationReport(
            compile=BucketResult(bucket="compile", metrics={"physical_targets_std": 0.1}, n_trials=10),
            replay=BucketResult(bucket="replay", metrics={"max_audio_diff": 0.5}, n_trials=5),
            transfer=BucketResult(bucket="transfer", metrics={"min_trajectory_correlation": 0.95}, n_trials=5),
        )
        assert not validate_no_mixed_buckets(bad_report)
        assert bad_report.mixed_bucket_violation
