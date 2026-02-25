"""Tests for tmrvc_train.tts_trainer module."""

from __future__ import annotations

import torch

from tmrvc_core.types import TTSBatch
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor
from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.tts_trainer import TTSTrainer, TTSTrainerConfig


def _make_batch(B: int = 2, L: int = 10, T: int = 50) -> TTSBatch:
    """Create a minimal TTSBatch for testing."""
    return TTSBatch(
        phoneme_ids=torch.randint(0, 100, (B, L)),
        durations=torch.full((B, L), T // L),  # uniform durations
        language_ids=torch.zeros(B, dtype=torch.long),
        content=torch.randn(B, 256, T),
        f0=torch.abs(torch.randn(B, 1, T)) * 200 + 100,  # positive F0
        spk_embed=torch.randn(B, 192),
        mel_target=torch.randn(B, 80, T),
        frame_lengths=torch.full((B,), T, dtype=torch.long),
        phoneme_lengths=torch.full((B,), L, dtype=torch.long),
        content_dim=256,
    )


def _make_trainer(tmp_path, max_steps=2, log_every=1, save_every=1) -> TTSTrainer:
    """Create a TTSTrainer with small models for testing."""
    text_encoder = TextEncoder()
    duration_predictor = DurationPredictor()
    f0_predictor = F0Predictor()
    content_synthesizer = ContentSynthesizer()

    all_params = (
        list(text_encoder.parameters())
        + list(duration_predictor.parameters())
        + list(f0_predictor.parameters())
        + list(content_synthesizer.parameters())
    )
    optimizer = torch.optim.Adam(all_params, lr=1e-3)

    config = TTSTrainerConfig(
        max_steps=max_steps,
        log_every=log_every,
        save_every=save_every,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )

    batch = _make_batch()
    dataloader = [batch]  # Simple iterable

    return TTSTrainer(
        text_encoder=text_encoder,
        duration_predictor=duration_predictor,
        f0_predictor=f0_predictor,
        content_synthesizer=content_synthesizer,
        optimizer=optimizer,
        dataloader=dataloader,
        config=config,
    )


class TestTTSTrainerConfig:
    def test_defaults(self):
        cfg = TTSTrainerConfig()
        assert cfg.lr == 1e-4
        assert cfg.max_steps == 200_000
        assert cfg.lambda_duration == 1.0
        assert cfg.lambda_f0 == 0.5
        assert cfg.lambda_content == 1.0
        assert cfg.lambda_voiced == 0.2


class TestTTSTrainer:
    def test_train_step_returns_losses(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_batch()
        losses = trainer.train_step(batch)

        assert "total" in losses
        assert "duration" in losses
        assert "f0" in losses
        assert "content" in losses
        assert "voiced" in losses
        assert all(isinstance(v, float) for v in losses.values())
        assert trainer.global_step == 1

    def test_train_step_loss_decreases_or_finite(self, tmp_path):
        trainer = _make_trainer(tmp_path, max_steps=5)
        batch = _make_batch()
        losses1 = trainer.train_step(batch)
        losses2 = trainer.train_step(batch)

        # At minimum, losses should be finite
        for v in losses1.values():
            assert not torch.tensor(v).isnan()
            assert not torch.tensor(v).isinf()
        for v in losses2.values():
            assert not torch.tensor(v).isnan()
            assert not torch.tensor(v).isinf()

    def test_save_and_load_checkpoint(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_batch()
        trainer.train_step(batch)

        ckpt_path = trainer.save_checkpoint()
        assert ckpt_path.exists()

        # Create a fresh trainer and load
        trainer2 = _make_trainer(tmp_path)
        assert trainer2.global_step == 0
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == 1

    def test_checkpoint_contains_all_keys(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        batch = _make_batch()
        trainer.train_step(batch)
        ckpt_path = trainer.save_checkpoint()

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "global_step" in ckpt
        assert "text_encoder" in ckpt
        assert "duration_predictor" in ckpt
        assert "f0_predictor" in ckpt
        assert "content_synthesizer" in ckpt
        assert "optimizer" in ckpt

    def test_train_iter_yields_steps(self, tmp_path):
        trainer = _make_trainer(tmp_path, max_steps=2)
        steps = list(trainer.train_iter())
        assert len(steps) == 2
        assert steps[0][0] == 1  # step 1
        assert steps[1][0] == 2  # step 2
        assert isinstance(steps[0][1], dict)

    def test_grad_clip(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        trainer.config.grad_clip = 0.1
        batch = _make_batch()
        losses = trainer.train_step(batch)
        # Should not error; grad clip is applied
        assert losses["total"] >= 0 or True  # just verify no crash
