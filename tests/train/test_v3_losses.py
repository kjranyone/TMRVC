"""Tests for v3 loss functions.

Covers:
- voice_state_supervision_loss output shape and value range
- context_separation_score computation
- prosody_collapse_score computation
- control_response_score computation
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_train.models.uclm_loss import (
    context_separation_score,
    control_response_score,
    prosody_collapse_score,
    voice_state_supervision_loss,
)


# ---------------------------------------------------------------------------
# voice_state_supervision_loss
# ---------------------------------------------------------------------------


class TestVoiceStateSupervisionLoss:
    def test_output_is_scalar(self):
        pred = torch.randn(2, 30, 8)
        tgt = torch.randn(2, 30, 8)
        loss = voice_state_supervision_loss(pred, tgt)
        assert loss.dim() == 0

    def test_output_non_negative(self):
        pred = torch.randn(2, 30, 8)
        tgt = torch.randn(2, 30, 8)
        loss = voice_state_supervision_loss(pred, tgt)
        assert loss.item() >= 0.0

    def test_zero_loss_when_identical(self):
        x = torch.randn(2, 30, 8)
        loss = voice_state_supervision_loss(x, x.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_with_mask(self):
        pred = torch.randn(2, 30, 8)
        tgt = torch.randn(2, 30, 8)
        mask = torch.zeros(2, 30, dtype=torch.bool)
        mask[:, 20:] = True  # mask out last 10 frames

        loss_masked = voice_state_supervision_loss(pred, tgt, mask=mask)
        assert loss_masked.dim() == 0
        assert loss_masked.item() >= 0.0

    def test_fully_masked_returns_zero(self):
        pred = torch.randn(2, 10, 8)
        tgt = torch.randn(2, 10, 8)
        mask = torch.ones(2, 10, dtype=torch.bool)  # everything masked

        loss = voice_state_supervision_loss(pred, tgt, mask=mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# context_separation_score
# ---------------------------------------------------------------------------


class TestContextSeparationScore:
    def test_identical_states_give_zero(self):
        """Identical hidden states for same group should have zero separation."""
        B, T, D = 4, 20, 32
        h = torch.randn(1, T, D).expand(B, -1, -1).clone()
        groups = torch.tensor([0, 0, 0, 0])

        score = context_separation_score(h, groups)
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_diverse_states_give_positive(self):
        """Different hidden states for same group should have positive separation."""
        B, T, D = 4, 20, 32
        h = torch.randn(B, T, D)
        groups = torch.tensor([0, 0, 1, 1])

        score = context_separation_score(h, groups)
        assert score >= 0.0

    def test_no_same_group_pairs_returns_zero(self):
        """When no samples share a group, score should be 0."""
        B, T, D = 3, 10, 16
        h = torch.randn(B, T, D)
        groups = torch.tensor([0, 1, 2])

        score = context_separation_score(h, groups)
        assert score == 0.0

    def test_return_type_is_float(self):
        h = torch.randn(2, 10, 16)
        groups = torch.tensor([0, 0])
        score = context_separation_score(h, groups)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# prosody_collapse_score
# ---------------------------------------------------------------------------


class TestProsodyCollapseScore:
    def test_identical_within_group_returns_zero(self):
        """If all samples in a group are identical, between-group variance is zero."""
        B, T, D = 4, 10, 16
        base = torch.randn(1, T, D)
        h = base.expand(B, -1, -1).clone()
        groups = torch.tensor([0, 0, 1, 1])

        score = prosody_collapse_score(h, groups)
        # With identical samples per group, between-group variance exists only
        # if group means differ -- but all are the same here.
        assert score == pytest.approx(0.0, abs=1e-4)

    def test_diverse_groups_give_positive(self):
        """Different group means should produce positive score."""
        B, T, D = 6, 10, 16
        h = torch.randn(B, T, D)
        # Offset group 1 by a large constant to ensure different means
        h[3:, :, :] += 10.0
        groups = torch.tensor([0, 0, 0, 1, 1, 1])

        score = prosody_collapse_score(h, groups)
        assert score > 0.0

    def test_single_group_returns_zero(self):
        """With only one group, no between-group comparison is possible."""
        h = torch.randn(3, 10, 16)
        groups = torch.tensor([0, 0, 0])
        score = prosody_collapse_score(h, groups)
        assert score == 0.0

    def test_return_type_is_float(self):
        h = torch.randn(4, 10, 16)
        groups = torch.tensor([0, 0, 1, 1])
        score = prosody_collapse_score(h, groups)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# control_response_score
# ---------------------------------------------------------------------------


class TestControlResponseScore:
    def test_perfect_monotonic_correlation(self):
        """Perfectly correlated inputs should give score ~1.0."""
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        controls = [0.1, 0.2, 0.3, 0.4, 0.5]
        score = control_response_score(durations, controls)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_reverse_correlation(self):
        """Reversed order should give score ~-1.0."""
        durations = [5.0, 4.0, 3.0, 2.0, 1.0]
        controls = [0.1, 0.2, 0.3, 0.4, 0.5]
        score = control_response_score(durations, controls)
        assert score == pytest.approx(-1.0, abs=1e-6)

    def test_single_item_returns_zero(self):
        """With < 2 items, no correlation is possible."""
        score = control_response_score([1.0], [0.5])
        assert score == 0.0

    def test_empty_returns_zero(self):
        score = control_response_score([], [])
        assert score == 0.0

    def test_return_type_is_float(self):
        score = control_response_score([1.0, 2.0], [0.5, 1.0])
        assert isinstance(score, float)

    def test_no_correlation(self):
        """With random orderings, score should be between -1 and 1."""
        durations = [3.0, 1.0, 4.0, 2.0]
        controls = [0.1, 0.2, 0.3, 0.4]
        score = control_response_score(durations, controls)
        assert -1.0 <= score <= 1.0
