"""Tests for LCD (Latency-Conditioned Distillation) module and losses."""

from __future__ import annotations

import torch

from tmrvc_train.lcd import (
    LatencyConditioner,
    LatencyLoss,
    MonotonicityLoss,
    latency_budget,
)


class TestLatencyBudget:
    def test_q_zero_gives_hq(self):
        q = torch.tensor([0.0])
        assert latency_budget(q).item() == 80.0

    def test_q_one_gives_live(self):
        q = torch.tensor([1.0])
        assert latency_budget(q).item() == 20.0

    def test_q_half(self):
        q = torch.tensor([0.5])
        assert abs(latency_budget(q).item() - 50.0) < 1e-5

    def test_batch(self):
        q = torch.tensor([0.0, 0.5, 1.0])
        budgets = latency_budget(q)
        assert budgets.shape == (3,)
        assert abs(budgets[0].item() - 80.0) < 1e-5
        assert abs(budgets[2].item() - 20.0) < 1e-5

    def test_monotonic(self):
        q = torch.linspace(0, 1, 10)
        budgets = latency_budget(q)
        # Budget decreases as q increases
        for i in range(len(budgets) - 1):
            assert budgets[i] >= budgets[i + 1]


class TestLatencyConditioner:
    def test_output_shape(self):
        cond = LatencyConditioner(d_output=32)
        q = torch.rand(4)
        out = cond(q)
        assert out.shape == (4, 32)

    def test_different_q_gives_different_output(self):
        cond = LatencyConditioner(d_output=32)
        q1 = torch.tensor([0.0])
        q2 = torch.tensor([1.0])
        out1 = cond(q1)
        out2 = cond(q2)
        assert not torch.allclose(out1, out2)

    def test_gradient_flow(self):
        cond = LatencyConditioner(d_output=32)
        q = torch.rand(2, requires_grad=True)
        out = cond(q)
        out.sum().backward()
        assert q.grad is not None

    def test_param_count(self):
        cond = LatencyConditioner(d_output=32)
        n_params = sum(p.numel() for p in cond.parameters())
        assert 50 < n_params < 5000


class TestLatencyLoss:
    def test_no_violation(self):
        loss_fn = LatencyLoss(base_runtime_ms=10.0, overhead_scale=1.0)
        q = torch.tensor([0.0])  # budget = 80ms
        act_norm = torch.tensor([1.0])  # est = 10 + 1 = 11ms << 80ms
        loss = loss_fn(q, act_norm)
        assert loss.item() == 0.0

    def test_violation(self):
        loss_fn = LatencyLoss(base_runtime_ms=10.0, overhead_scale=100.0)
        q = torch.tensor([1.0])  # budget = 20ms
        act_norm = torch.tensor([1.0])  # est = 10 + 100 = 110ms >> 20ms
        loss = loss_fn(q, act_norm)
        assert loss.item() > 0.0

    def test_batch(self):
        loss_fn = LatencyLoss()
        q = torch.rand(4)
        act_norm = torch.rand(4)
        loss = loss_fn(q, act_norm)
        assert not loss.isnan()
        assert not loss.isinf()

    def test_gradient_flow(self):
        loss_fn = LatencyLoss(base_runtime_ms=5.0, overhead_scale=50.0)
        q = torch.tensor([1.0])  # tight budget
        act_norm = torch.tensor([1.0], requires_grad=True)
        loss = loss_fn(q, act_norm)
        loss.backward()
        assert act_norm.grad is not None


class TestMonotonicityLoss:
    def test_no_violation(self):
        loss_fn = MonotonicityLoss(margin=0.05)
        # High quality for high latency, low for low latency
        quality_low_lat = torch.tensor([0.5])  # less latency → lower quality
        quality_high_lat = torch.tensor([0.8])  # more latency → higher quality
        loss = loss_fn(quality_low_lat, quality_high_lat)
        assert loss.item() == 0.0

    def test_violation(self):
        loss_fn = MonotonicityLoss(margin=0.05)
        # Reversed: high quality for low latency (violation)
        quality_low_lat = torch.tensor([0.9])
        quality_high_lat = torch.tensor([0.5])
        loss = loss_fn(quality_low_lat, quality_high_lat)
        assert loss.item() > 0.0

    def test_margin_effect(self):
        quality_low = torch.tensor([0.7])
        quality_high = torch.tensor([0.72])  # only 0.02 margin

        loss_tight = MonotonicityLoss(margin=0.01)
        loss_strict = MonotonicityLoss(margin=0.1)

        # With tight margin (0.01), 0.02 gap is ok
        assert loss_tight(quality_low, quality_high).item() == 0.0
        # With strict margin (0.1), 0.02 gap is not enough
        assert loss_strict(quality_low, quality_high).item() > 0.0

    def test_batch(self):
        loss_fn = MonotonicityLoss(margin=0.05)
        quality_low = torch.rand(4)
        quality_high = quality_low + 0.1  # always higher
        loss = loss_fn(quality_low, quality_high)
        assert loss.item() == 0.0


class TestB4Config:
    def test_b4_yaml_loads(self):
        import yaml
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "research" / "b4.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert cfg["variant"] == "b4"
        assert cfg["model"]["ssl_enabled"] is True
        assert cfg["model"]["bpeh_enabled"] is True
        assert cfg["model"]["lcd_enabled"] is True
        assert "lcd" in cfg
        assert cfg["lcd"]["mono_margin"] == 0.05

    def test_b4_loss_weights(self):
        import yaml
        from pathlib import Path

        cfg_path = Path(__file__).resolve().parents[2] / "configs" / "research" / "b4.yaml"
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        lw = cfg["loss_weights"]
        assert "lambda_latency" in lw
        assert "lambda_mono" in lw
        assert lw["lambda_latency"] == 0.2
        assert lw["lambda_mono"] == 0.2


class TestDistillationLCD:
    def test_distillation_config_lcd_fields(self):
        from tmrvc_train.distillation import DistillationConfig

        cfg = DistillationConfig(enable_lcd=True, lambda_latency=0.3, lambda_mono=0.1)
        assert cfg.enable_lcd is True
        assert cfg.lambda_latency == 0.3
        assert cfg.lambda_mono == 0.1
        assert cfg.mono_margin == 0.05

    def test_lcd_conditioner_in_trainer(self):
        from unittest.mock import MagicMock
        from tmrvc_train.distillation import DistillationConfig, DistillationTrainer

        # Create a real parameter for converter so device resolution works
        real_param = torch.nn.Parameter(torch.zeros(2))

        teacher = MagicMock()
        teacher.parameters.return_value = iter([torch.zeros(1)])
        teacher.eval = MagicMock()

        content_encoder = MagicMock()
        content_encoder.parameters.return_value = iter([torch.zeros(1)])
        converter = MagicMock()
        converter.parameters.return_value = iter([real_param])
        vocoder = MagicMock()
        vocoder.parameters.return_value = iter([torch.zeros(1)])
        ir_estimator = MagicMock()
        ir_estimator.parameters.return_value = iter([torch.zeros(1)])

        config = DistillationConfig(enable_lcd=True)
        scheduler = MagicMock()

        all_params = [torch.zeros(2, requires_grad=True)]
        optimizer = torch.optim.Adam(all_params, lr=1e-3)
        dataloader = []

        trainer = DistillationTrainer(
            teacher=teacher,
            content_encoder=content_encoder,
            converter=converter,
            vocoder=vocoder,
            ir_estimator=ir_estimator,
            scheduler=scheduler,
            optimizer=optimizer,
            dataloader=dataloader,
            config=config,
        )

        assert trainer.lcd_conditioner is not None
        assert trainer.lcd_latency_loss is not None
        assert trainer.lcd_mono_loss is not None
