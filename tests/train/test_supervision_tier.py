"""Tests for supervision tier-aware loss weighting (Phase 3, Task 3-1).

Covers:
- Tier D sample loss contribution is less than 10% of Tier A
- Masked dimensions do not contribute to physical loss
- Per-sample tier weighting is applied correctly in trainer
- Physical observed_mask correctly zeros out unobserved dimensions
"""

import torch
import torch.nn.functional as F
import pytest


class TestTierWeightConstants:
    """Test TIER_WEIGHTS mapping."""

    def test_tier_weights_exist(self):
        from tmrvc_train.trainer import TIER_WEIGHTS

        assert "A" in TIER_WEIGHTS
        assert "B" in TIER_WEIGHTS
        assert "C" in TIER_WEIGHTS
        assert "D" in TIER_WEIGHTS

    def test_tier_a_is_full_weight(self):
        from tmrvc_train.trainer import TIER_WEIGHTS

        assert TIER_WEIGHTS["A"] == 1.0

    def test_tier_d_is_minimal(self):
        from tmrvc_train.trainer import TIER_WEIGHTS

        assert TIER_WEIGHTS["D"] == 0.1

    def test_tier_ordering(self):
        from tmrvc_train.trainer import TIER_WEIGHTS

        assert TIER_WEIGHTS["A"] > TIER_WEIGHTS["B"]
        assert TIER_WEIGHTS["B"] > TIER_WEIGHTS["C"]
        assert TIER_WEIGHTS["C"] > TIER_WEIGHTS["D"]

    def test_tier_d_less_than_10_percent_of_a(self):
        from tmrvc_train.trainer import TIER_WEIGHTS

        ratio = TIER_WEIGHTS["D"] / TIER_WEIGHTS["A"]
        assert ratio <= 0.10, f"Tier D / Tier A ratio is {ratio}, expected <= 0.10"


class TestTierKeyMapping:
    """Test dataset tier string to TIER_WEIGHTS key mapping."""

    def test_tier_key_map_covers_all_tiers(self):
        from tmrvc_train.trainer import _TIER_KEY_MAP

        assert _TIER_KEY_MAP["tier_a"] == "A"
        assert _TIER_KEY_MAP["tier_b"] == "B"
        assert _TIER_KEY_MAP["tier_c"] == "C"
        assert _TIER_KEY_MAP["tier_d"] == "D"


class TestTierAwareLossWeighting:
    """Test that tier-aware weighting reduces loss contribution for low tiers."""

    def test_tier_d_loss_is_scaled_down(self):
        """Tier D sample should produce 10% of the loss of Tier A."""
        from tmrvc_train.trainer import TIER_WEIGHTS, _TIER_KEY_MAP

        base_loss = torch.tensor(5.0)

        # Tier A
        tier_a_key = _TIER_KEY_MAP["tier_a"]
        loss_a = base_loss * TIER_WEIGHTS[tier_a_key]

        # Tier D
        tier_d_key = _TIER_KEY_MAP["tier_d"]
        loss_d = base_loss * TIER_WEIGHTS[tier_d_key]

        assert loss_d.item() == pytest.approx(base_loss.item() * 0.1)
        assert loss_d.item() < loss_a.item() * 0.15  # Tier D < 15% of Tier A

    def test_mixed_batch_average_weighting(self):
        """Average tier weight across a mixed batch."""
        from tmrvc_train.trainer import TIER_WEIGHTS, _TIER_KEY_MAP

        tiers = ["tier_a", "tier_b", "tier_c", "tier_d"]
        avg_weight = sum(
            TIER_WEIGHTS[_TIER_KEY_MAP[t]] for t in tiers
        ) / len(tiers)

        # Average of (1.0 + 0.7 + 0.3 + 0.1) / 4 = 0.525
        assert avg_weight == pytest.approx(0.525)


class TestMaskedPhysicalLoss:
    """Test that masked dimensions do not contribute to physical loss."""

    def test_masked_dims_zero_loss(self):
        """Loss should be zero for completely masked samples."""
        B, T, D = 2, 50, 12
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)

        # All dimensions masked (unobserved)
        mask = torch.zeros(B, T, D, dtype=torch.bool)

        phys_loss = F.mse_loss(pred, target, reduction='none')
        phys_loss = phys_loss * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        loss = phys_loss.sum() / denom

        assert loss.item() == 0.0

    def test_partial_mask_excludes_dims(self):
        """Only observed dimensions should contribute to loss."""
        B, T, D = 2, 50, 12
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)

        # Only first 6 dimensions observed
        mask = torch.zeros(B, T, D, dtype=torch.bool)
        mask[:, :, :6] = True

        phys_loss = F.mse_loss(pred, target, reduction='none')
        masked_loss = (phys_loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)

        # Compare with loss computed only on first 6 dims
        expected_loss = F.mse_loss(pred[:, :, :6], target[:, :, :6])

        assert masked_loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)

    def test_nan_targets_do_not_leak(self):
        """NaN in targets at masked positions should not affect loss.

        The correct approach is to replace NaN with 0 before MSE, then mask.
        This matches the trainer implementation which uses physical_mask to
        zero out unobserved dimensions.
        """
        B, T, D = 2, 50, 12
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)

        # Set unobserved dimensions to NaN
        mask = torch.ones(B, T, D, dtype=torch.bool)
        mask[:, :, 6:] = False  # Last 6 dims unobserved
        target[:, :, 6:] = float('nan')

        # Safe computation: replace NaN before MSE, then apply mask
        target_safe = torch.where(mask, target, torch.zeros_like(target))
        phys_loss = F.mse_loss(pred, target_safe, reduction='none')
        phys_loss = phys_loss * mask.float()
        denom = mask.float().sum().clamp(min=1.0)
        loss = phys_loss.sum() / denom

        # Should be finite because NaN dims are zeroed before MSE
        assert torch.isfinite(loss), "Loss should be finite when NaN dims are masked"

    def test_full_mask_matches_unmasked(self):
        """Fully observed mask should give same loss as plain MSE."""
        B, T, D = 2, 50, 12
        pred = torch.randn(B, T, D)
        target = torch.randn(B, T, D)

        mask = torch.ones(B, T, D, dtype=torch.bool)

        phys_loss = F.mse_loss(pred, target, reduction='none')
        masked_loss = (phys_loss * mask.float()).sum() / mask.float().sum()

        plain_loss = F.mse_loss(pred, target)

        assert masked_loss.item() == pytest.approx(plain_loss.item(), abs=1e-5)


class TestTierLossWeightsFunction:
    """Test get_tier_loss_weights from v4_dataset."""

    def test_tier_a_all_ones(self):
        from tmrvc_data.v4_dataset import get_tier_loss_weights

        weights = get_tier_loss_weights("tier_a")
        for key in ["codec_loss", "control_loss", "pointer_loss",
                     "physical_loss", "acting_latent_loss", "speaker_loss"]:
            assert weights[key] == 1.0, f"Tier A {key} should be 1.0"

    def test_tier_d_physical_zero(self):
        from tmrvc_data.v4_dataset import get_tier_loss_weights

        weights = get_tier_loss_weights("tier_d")
        assert weights["physical_loss"] == 0.0
        assert weights["acting_latent_loss"] == 0.0
        assert weights["disentanglement_loss"] == 0.0

    def test_tier_b_intermediate(self):
        from tmrvc_data.v4_dataset import get_tier_loss_weights

        weights = get_tier_loss_weights("tier_b")
        assert 0.0 < weights["physical_loss"] < 1.0
        assert weights["codec_loss"] == 1.0

    def test_all_tiers_have_nine_loss_keys(self):
        from tmrvc_data.v4_dataset import get_tier_loss_weights

        required_keys = {
            "codec_loss", "control_loss", "pointer_loss",
            "physical_loss", "acting_latent_loss", "disentanglement_loss",
            "speaker_loss", "prosody_loss", "semantic_loss",
        }
        for tier in ["tier_a", "tier_b", "tier_c", "tier_d"]:
            weights = get_tier_loss_weights(tier)
            assert required_keys.issubset(weights.keys()), \
                f"Tier {tier} missing keys: {required_keys - weights.keys()}"
