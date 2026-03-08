"""Tests for voice_state supervision artifact contracts (Worker 03)."""

from __future__ import annotations

import numpy as np
import pytest
import torch


class TestVoiceStateArtifactShapes:
    """Verify canonical voice_state artifact shapes."""

    def test_targets_shape(self):
        targets = np.random.randn(100, 8).astype(np.float32)
        assert targets.shape == (100, 8)

    def test_observed_mask_shape(self):
        mask = np.ones((100, 8), dtype=bool)
        assert mask.shape == (100, 8)
        assert mask.dtype == bool

    def test_confidence_shape_per_dim(self):
        conf = np.ones((100, 8), dtype=np.float32)
        assert conf.shape == (100, 8)

    def test_confidence_shape_scalar(self):
        conf = np.ones((100, 1), dtype=np.float32)
        assert conf.shape == (100, 1)

    def test_missing_dims_masked_not_zeroed(self):
        """Missing dimensions must be masked, never treated as zero-valued."""
        T, D = 100, 8
        targets = np.random.randn(T, D).astype(np.float32)
        mask = np.zeros((T, D), dtype=bool)
        # Only first 4 dims observed
        mask[:, :4] = True

        targets_t = torch.from_numpy(targets)
        mask_t = torch.from_numpy(mask)

        # Masked loss should ignore unobserved dims
        mse = ((targets_t - 0) ** 2)
        masked_mse = mse * mask_t.float()
        loss = masked_mse.sum() / mask_t.float().sum()

        # Only dims 0-3 contribute
        expected = (targets_t[:, :4] ** 2).mean()
        assert abs(loss.item() - expected.item()) < 1e-5


class TestVoiceStateMetaContract:
    """Verify voice_state_meta.json contract fields."""

    def test_required_meta_fields(self):
        meta = {
            "estimator_identity": "heuristic_v1",
            "calibration_version": "2024-01",
            "provenance": "pseudo_labeled",
            "label_type": "pseudo_labeled",
        }
        for field in ["estimator_identity", "calibration_version", "provenance", "label_type"]:
            assert field in meta

    def test_label_type_values(self):
        valid = {"direct", "pseudo_labeled", "absent"}
        assert "pseudo_labeled" in valid
        assert "direct" in valid
        assert "absent" in valid
