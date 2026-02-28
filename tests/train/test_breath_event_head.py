"""Tests for BreathEventHead model and BreathEventLoss."""

from __future__ import annotations

import torch

from tmrvc_train.models.breath_event_head import BreathEventHead, BreathEventLoss


class TestBreathEventHead:
    def test_forward_shapes(self):
        head = BreathEventHead()
        x = torch.randn(2, 256, 50)
        onset, dur, intensity, pause = head(x)

        assert onset.shape == (2, 1, 50)
        assert dur.shape == (2, 1, 50)
        assert intensity.shape == (2, 1, 50)
        assert pause.shape == (2, 1, 50)

    def test_output_ranges(self):
        head = BreathEventHead()
        x = torch.randn(4, 256, 30)
        onset, dur, intensity, pause = head(x)

        # Duration outputs are Softplus → always positive
        assert (dur >= 0).all()
        assert (pause >= 0).all()
        # Intensity is Sigmoid → (0, 1)
        assert (intensity >= 0).all()
        assert (intensity <= 1).all()
        # Onset logits can be any value (raw logits)

    def test_custom_dimensions(self):
        head = BreathEventHead(d_input=128, d_hidden=64, n_blocks=2)
        x = torch.randn(2, 128, 40)
        onset, dur, intensity, pause = head(x)
        assert onset.shape == (2, 1, 40)

    def test_single_frame(self):
        head = BreathEventHead()
        x = torch.randn(1, 256, 1)
        onset, dur, intensity, pause = head(x)
        assert onset.shape == (1, 1, 1)

    def test_gradient_flow(self):
        head = BreathEventHead()
        x = torch.randn(2, 256, 20, requires_grad=True)
        onset, dur, intensity, pause = head(x)
        loss = onset.sum() + dur.sum() + intensity.sum() + pause.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_param_count(self):
        head = BreathEventHead()
        n_params = sum(p.numel() for p in head.parameters())
        # Should be reasonable (~100K range)
        assert 10_000 < n_params < 500_000


class TestBreathEventLoss:
    def _make_inputs(self, B=2, T=50):
        pred_onset = torch.randn(B, 1, T)
        pred_dur = torch.abs(torch.randn(B, 1, T))
        pred_int = torch.sigmoid(torch.randn(B, 1, T))
        pred_pause = torch.abs(torch.randn(B, 1, T))

        gt_onset = torch.zeros(B, T)
        gt_onset[0, 5] = 1.0
        gt_onset[0, 20] = 1.0
        gt_onset[1, 10] = 1.0

        gt_dur = torch.zeros(B, T)
        gt_dur[0, 5] = 300.0
        gt_dur[0, 20] = 200.0
        gt_dur[1, 10] = 250.0

        gt_int = torch.zeros(B, T)
        gt_int[0, 5] = 0.8
        gt_int[0, 20] = 0.6
        gt_int[1, 10] = 0.7

        gt_pause = torch.zeros(B, T)
        gt_pause[0, 30] = 150.0
        gt_pause[1, 25] = 100.0

        mask = torch.ones(B, 1, T)
        return pred_onset, pred_dur, pred_int, pred_pause, gt_onset, gt_dur, gt_int, gt_pause, mask

    def test_forward_returns_all_keys(self):
        loss_fn = BreathEventLoss()
        args = self._make_inputs()
        result = loss_fn(*args)

        assert "event_onset" in result
        assert "event_dur" in result
        assert "event_amp" in result
        assert "event_total" in result

    def test_losses_are_finite(self):
        loss_fn = BreathEventLoss()
        args = self._make_inputs()
        result = loss_fn(*args)

        for k, v in result.items():
            assert not v.isnan(), f"{k} is NaN"
            assert not v.isinf(), f"{k} is Inf"

    def test_losses_are_nonnegative(self):
        loss_fn = BreathEventLoss()
        args = self._make_inputs()
        result = loss_fn(*args)

        for k, v in result.items():
            assert v.item() >= 0, f"{k} is negative: {v.item()}"

    def test_total_is_weighted_sum(self):
        loss_fn = BreathEventLoss(lambda_onset=0.5, lambda_dur=0.3, lambda_amp=0.2)
        args = self._make_inputs()
        result = loss_fn(*args)

        expected_total = (
            0.5 * result["event_onset"]
            + 0.3 * result["event_dur"]
            + 0.2 * result["event_amp"]
        )
        assert torch.allclose(result["event_total"], expected_total, atol=1e-6)

    def test_masking(self):
        loss_fn = BreathEventLoss()
        args = list(self._make_inputs(B=2, T=50))
        # Mask out second half
        args[8] = torch.ones(2, 1, 50)
        args[8][:, :, 25:] = 0.0
        result1 = loss_fn(*args)

        # Full mask
        args[8] = torch.ones(2, 1, 50)
        result2 = loss_fn(*args)

        # Losses should differ when mask changes
        assert result1["event_onset"].item() != result2["event_onset"].item()

    def test_no_events(self):
        """All-zero GT should still produce valid (non-NaN) losses."""
        loss_fn = BreathEventLoss()
        B, T = 2, 30
        pred_onset = torch.randn(B, 1, T)
        pred_dur = torch.abs(torch.randn(B, 1, T))
        pred_int = torch.sigmoid(torch.randn(B, 1, T))
        pred_pause = torch.abs(torch.randn(B, 1, T))

        gt_onset = torch.zeros(B, T)
        gt_dur = torch.zeros(B, T)
        gt_int = torch.zeros(B, T)
        gt_pause = torch.zeros(B, T)
        mask = torch.ones(B, 1, T)

        result = loss_fn(
            pred_onset, pred_dur, pred_int, pred_pause,
            gt_onset, gt_dur, gt_int, gt_pause, mask,
        )
        for k, v in result.items():
            assert not v.isnan(), f"{k} is NaN with no events"

    def test_focal_gamma_effect(self):
        """Higher gamma should reduce loss for well-classified samples."""
        args = self._make_inputs()

        loss_low_gamma = BreathEventLoss(focal_gamma=0.0)
        loss_high_gamma = BreathEventLoss(focal_gamma=4.0)

        result_low = loss_low_gamma(*args)
        result_high = loss_high_gamma(*args)

        # With focal weighting, onset loss should differ
        assert result_low["event_onset"].item() != result_high["event_onset"].item()

    def test_gradient_flows(self):
        loss_fn = BreathEventLoss()
        head = BreathEventHead()
        x = torch.randn(2, 256, 50, requires_grad=True)
        onset, dur, intensity, pause = head(x)

        gt_onset = torch.zeros(2, 50)
        gt_onset[0, 5] = 1.0
        gt_dur = torch.zeros(2, 50)
        gt_dur[0, 5] = 300.0
        gt_int = torch.zeros(2, 50)
        gt_int[0, 5] = 0.8
        gt_pause = torch.zeros(2, 50)
        mask = torch.ones(2, 1, 50)

        result = loss_fn(onset, dur, intensity, pause, gt_onset, gt_dur, gt_int, gt_pause, mask)
        result["event_total"].backward()
        assert x.grad is not None
