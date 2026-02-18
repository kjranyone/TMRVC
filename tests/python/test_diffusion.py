"""Tests for FlowMatchingScheduler."""

import torch
import pytest

from tmrvc_train.diffusion import FlowMatchingScheduler


class TestFlowMatchingScheduler:
    """Tests for FlowMatchingScheduler."""

    @pytest.fixture
    def scheduler(self):
        return FlowMatchingScheduler()

    def test_forward_process_shapes(self, scheduler):
        x_0 = torch.randn(2, 80, 50)
        t = torch.rand(2, 1, 1)
        x_t, v_target = scheduler.forward_process(x_0, t)
        assert x_t.shape == x_0.shape
        assert v_target.shape == x_0.shape

    def test_forward_process_t0_recovers_x0(self, scheduler):
        """At t=0, x_t should equal x_0."""
        x_0 = torch.randn(1, 80, 10)
        t = torch.zeros(1, 1, 1)
        x_t, _ = scheduler.forward_process(x_0, t)
        torch.testing.assert_close(x_t, x_0)

    def test_forward_process_t1_is_noise(self, scheduler):
        """At t=1, x_t should be pure noise (independent of x_0)."""
        x_0 = torch.zeros(1, 80, 10)
        t = torch.ones(1, 1, 1)
        x_t, v_target = scheduler.forward_process(x_0, t)
        # x_t = (1-1)*x_0 + 1*noise = noise
        # v_target = noise - x_0 = noise
        torch.testing.assert_close(x_t, v_target)

    def test_velocity_target(self, scheduler):
        """v_target = noise - x_0."""
        torch.manual_seed(42)
        x_0 = torch.randn(1, 80, 10)
        t = torch.tensor([[[0.5]]])

        torch.manual_seed(123)
        x_t, v_target = scheduler.forward_process(x_0, t)

        # Reconstruct: x_t = (1-t)*x_0 + t*noise → noise = (x_t - (1-t)*x_0) / t
        noise_reconstructed = (x_t - (1 - t) * x_0) / t
        v_expected = noise_reconstructed - x_0
        torch.testing.assert_close(v_target, v_expected, atol=1e-5, rtol=1e-4)

    def test_sample_shape(self, scheduler):
        """Sample should return correct shape."""
        # Simple mock model that returns zeros
        class MockModel(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                return torch.zeros_like(x)

        model = MockModel()
        shape = (1, 80, 10)
        sample = scheduler.sample(model, shape, steps=5)
        assert sample.shape == shape

    def test_sample_with_zero_velocity_returns_noise(self, scheduler):
        """If model always predicts zero velocity, output should be the initial noise."""
        class ZeroModel(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                return torch.zeros_like(x)

        model = ZeroModel()
        torch.manual_seed(42)
        shape = (1, 80, 10)
        sample = scheduler.sample(model, shape, steps=10)
        # With zero velocity, x stays at initial noise
        torch.manual_seed(42)
        expected = torch.randn(shape)
        torch.testing.assert_close(sample, expected)


class TestSwayTimesteps:
    """Tests for sway sampling timestep generation."""

    @pytest.fixture
    def scheduler(self):
        return FlowMatchingScheduler()

    def test_sway_timesteps_monotonic(self, scheduler):
        """Timesteps should be monotonically decreasing from 1.0 to 0.0."""
        ts = scheduler._sway_timesteps(steps=10, sway_coeff=1.0)
        assert ts[0] == pytest.approx(1.0)
        assert ts[-1] == pytest.approx(0.0)
        for i in range(len(ts) - 1):
            assert ts[i] > ts[i + 1], f"Not monotonic at {i}: {ts[i]} <= {ts[i+1]}"

    def test_sway_timesteps_nonuniform(self, scheduler):
        """With sway_coeff > 0, steps should be non-uniformly spaced."""
        ts_sway = scheduler._sway_timesteps(steps=20, sway_coeff=1.0)
        # Compute step sizes
        deltas = ts_sway[:-1] - ts_sway[1:]
        # First half should have larger steps than second half (more concentrated in mid-range)
        first_half_mean = deltas[:10].mean()
        second_half_mean = deltas[10:].mean()
        assert first_half_mean != pytest.approx(second_half_mean, abs=1e-3)

    def test_sample_with_sway_shape(self, scheduler):
        """Sample with sway should produce same shape as uniform."""
        class MockModel(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                return torch.zeros_like(x)

        model = MockModel()
        shape = (1, 80, 10)
        sample = scheduler.sample(model, shape, steps=5, sway_coefficient=1.0)
        assert sample.shape == shape


class TestOTForwardProcess:
    """Tests for Optimal Transport Conditional Flow Matching."""

    @pytest.fixture
    def scheduler(self):
        return FlowMatchingScheduler()

    def test_ot_forward_process_shapes(self, scheduler):
        """Output shapes should match standard forward_process."""
        x_0 = torch.randn(4, 80, 50)
        t = torch.rand(4, 1, 1)
        x_t, v_target = scheduler.ot_forward_process(x_0, t)
        assert x_t.shape == x_0.shape
        assert v_target.shape == x_0.shape

    def test_ot_forward_process_t0(self, scheduler):
        """At t=0, x_t should equal x_0."""
        x_0 = torch.randn(4, 80, 10)
        t = torch.zeros(4, 1, 1)
        x_t, _ = scheduler.ot_forward_process(x_0, t)
        torch.testing.assert_close(x_t, x_0)

    def test_ot_forward_process_lower_cost(self, scheduler):
        """OT pairing should produce lower transport cost than random pairing."""
        torch.manual_seed(42)
        x_0 = torch.randn(8, 80, 10)
        t = torch.ones(8, 1, 1) * 0.5

        # OT forward process
        x_t_ot, v_ot = scheduler.ot_forward_process(x_0, t)
        cost_ot = v_ot.reshape(8, -1).pow(2).sum(dim=-1).mean()

        # Standard (random) forward process — average over multiple seeds
        costs_random = []
        for seed in range(5):
            torch.manual_seed(seed + 100)
            _, v_rand = scheduler.forward_process(x_0, t)
            costs_random.append(v_rand.reshape(8, -1).pow(2).sum(dim=-1).mean().item())
        cost_random_avg = sum(costs_random) / len(costs_random)

        # OT cost should be lower on average
        assert cost_ot.item() < cost_random_avg


class TestSampleCFG:
    """Tests for Classifier-Free Guidance sampling."""

    @pytest.fixture
    def scheduler(self):
        return FlowMatchingScheduler()

    def test_sample_cfg_scale_1_matches_standard(self, scheduler):
        """cfg_scale=1.0 should produce identical results to standard sample."""
        class MockModel(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                return x * 0.1

        model = MockModel()
        shape = (1, 80, 10)

        torch.manual_seed(42)
        sample_std = scheduler.sample(model, shape, steps=5)
        torch.manual_seed(42)
        sample_cfg = scheduler.sample_cfg(model, shape, steps=5, cfg_scale=1.0)

        torch.testing.assert_close(sample_std, sample_cfg)


class TestReflow:
    """Tests for Reflow pair generation and forward process."""

    @pytest.fixture
    def scheduler(self):
        return FlowMatchingScheduler()

    def test_generate_reflow_pairs_shape(self, scheduler):
        """Generated pairs should have same shape as input."""
        class MockModel(torch.nn.Module):
            def forward(self, x, t, **kwargs):
                return torch.randn_like(x) * 0.01

        model = MockModel()
        x_0 = torch.randn(2, 80, 10)
        x_1_noise, x_0_teacher = scheduler.generate_reflow_pairs(model, x_0, steps=5)
        assert x_1_noise.shape == x_0.shape
        assert x_0_teacher.shape == x_0.shape

    def test_reflow_forward_process_shapes(self, scheduler):
        """Reflow forward process outputs should match input shapes."""
        x_0_teacher = torch.randn(2, 80, 10)
        x_1_noise = torch.randn(2, 80, 10)
        t = torch.rand(2, 1, 1)
        x_t, v_target = scheduler.reflow_forward_process(x_0_teacher, x_1_noise, t)
        assert x_t.shape == x_0_teacher.shape
        assert v_target.shape == x_0_teacher.shape

    def test_reflow_forward_process_t0(self, scheduler):
        """At t=0, x_t should equal x_0_teacher."""
        x_0_teacher = torch.randn(2, 80, 10)
        x_1_noise = torch.randn(2, 80, 10)
        t = torch.zeros(2, 1, 1)
        x_t, _ = scheduler.reflow_forward_process(x_0_teacher, x_1_noise, t)
        torch.testing.assert_close(x_t, x_0_teacher)

    def test_reflow_forward_process_t1(self, scheduler):
        """At t=1, x_t should equal x_1_noise."""
        x_0_teacher = torch.randn(2, 80, 10)
        x_1_noise = torch.randn(2, 80, 10)
        t = torch.ones(2, 1, 1)
        x_t, _ = scheduler.reflow_forward_process(x_0_teacher, x_1_noise, t)
        torch.testing.assert_close(x_t, x_1_noise)
