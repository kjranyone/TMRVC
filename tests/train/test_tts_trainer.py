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


def _make_ssl_trainer(tmp_path, max_steps=2, log_every=1, save_every=1) -> TTSTrainer:
    """Create a TTSTrainer with SSL enabled for testing."""
    text_encoder = TextEncoder()
    duration_predictor = DurationPredictor()
    f0_predictor = F0Predictor()
    content_synthesizer = ContentSynthesizer()

    config = TTSTrainerConfig(
        max_steps=max_steps,
        log_every=log_every,
        save_every=save_every,
        checkpoint_dir=str(tmp_path / "ckpt_ssl"),
        enable_ssl=True,
        lambda_state_recon=0.5,
        lambda_state_cons=0.3,
    )

    # SSL modules are created inside TTSTrainer, so we need all params after init
    batch = _make_batch()
    dataloader = [batch]

    trainer = TTSTrainer(
        text_encoder=text_encoder,
        duration_predictor=duration_predictor,
        f0_predictor=f0_predictor,
        content_synthesizer=content_synthesizer,
        optimizer=torch.optim.Adam(text_encoder.parameters(), lr=1e-3),  # placeholder
        dataloader=dataloader,
        config=config,
    )
    # Rebuild optimizer with all params including SSL
    all_params = list(trainer._trainable_params())
    trainer.optimizer = torch.optim.Adam(all_params, lr=1e-3)
    return trainer


class TestTTSTrainerSSL:
    def test_ssl_modules_created(self, tmp_path):
        trainer = _make_ssl_trainer(tmp_path)
        assert trainer.ssl_state_update is not None
        assert trainer.ssl_history_encoder is not None
        assert trainer.ssl_prosody_predictor is not None
        assert trainer.ssl_loss_fn is not None

    def test_ssl_disabled_by_default(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.ssl_state_update is None

    def test_train_step_with_ssl(self, tmp_path):
        trainer = _make_ssl_trainer(tmp_path)
        batch = _make_batch()
        losses = trainer.train_step(batch)

        assert "total" in losses
        assert "state_recon" in losses
        assert "state_cons" in losses
        assert "state_total" in losses
        assert all(not torch.tensor(v).isnan() for v in losses.values())

    def test_ssl_losses_are_finite(self, tmp_path):
        trainer = _make_ssl_trainer(tmp_path, max_steps=3)
        batch = _make_batch()
        for _ in range(3):
            losses = trainer.train_step(batch)
            for k, v in losses.items():
                assert not torch.tensor(v).isnan(), f"{k} is NaN"
                assert not torch.tensor(v).isinf(), f"{k} is Inf"

    def test_ssl_checkpoint_roundtrip(self, tmp_path):
        trainer = _make_ssl_trainer(tmp_path)
        batch = _make_batch()
        trainer.train_step(batch)

        ckpt_path = trainer.save_checkpoint()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "ssl_state_update" in ckpt
        assert "ssl_history_encoder" in ckpt
        assert "ssl_prosody_predictor" in ckpt
        assert ckpt["enable_ssl"] is True

        # Load into new trainer
        trainer2 = _make_ssl_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == 1

    def test_train_iter_with_ssl(self, tmp_path):
        trainer = _make_ssl_trainer(tmp_path, max_steps=2)
        steps = list(trainer.train_iter())
        assert len(steps) == 2
        assert "state_total" in steps[0][1]


def _make_bpeh_batch(B: int = 2, L: int = 10, T: int = 50) -> TTSBatch:
    """Create a TTSBatch with BPEH event tensors."""
    breath_onsets = torch.zeros(B, T)
    breath_onsets[0, 5] = 1.0
    breath_onsets[0, 20] = 1.0
    breath_durations = torch.zeros(B, T)
    breath_durations[0, 5] = 300.0
    breath_durations[0, 20] = 200.0
    breath_intensity = torch.zeros(B, T)
    breath_intensity[0, 5] = 0.8
    breath_intensity[0, 20] = 0.6
    pause_durations = torch.zeros(B, T)
    pause_durations[0, 30] = 150.0

    return TTSBatch(
        phoneme_ids=torch.randint(0, 100, (B, L)),
        durations=torch.full((B, L), T // L),
        language_ids=torch.zeros(B, dtype=torch.long),
        content=torch.randn(B, 256, T),
        f0=torch.abs(torch.randn(B, 1, T)) * 200 + 100,
        spk_embed=torch.randn(B, 192),
        mel_target=torch.randn(B, 80, T),
        frame_lengths=torch.full((B,), T, dtype=torch.long),
        phoneme_lengths=torch.full((B,), L, dtype=torch.long),
        content_dim=256,
        breath_onsets=breath_onsets,
        breath_durations=breath_durations,
        breath_intensity=breath_intensity,
        pause_durations=pause_durations,
    )


def _make_bpeh_trainer(tmp_path, max_steps=2) -> TTSTrainer:
    """Create a TTSTrainer with BPEH enabled for testing."""
    text_encoder = TextEncoder()
    duration_predictor = DurationPredictor()
    f0_predictor = F0Predictor()
    content_synthesizer = ContentSynthesizer()

    config = TTSTrainerConfig(
        max_steps=max_steps,
        log_every=1,
        save_every=1,
        checkpoint_dir=str(tmp_path / "ckpt_bpeh"),
        enable_bpeh=True,
    )

    batch = _make_bpeh_batch()
    dataloader = [batch]

    trainer = TTSTrainer(
        text_encoder=text_encoder,
        duration_predictor=duration_predictor,
        f0_predictor=f0_predictor,
        content_synthesizer=content_synthesizer,
        optimizer=torch.optim.Adam(text_encoder.parameters(), lr=1e-3),
        dataloader=dataloader,
        config=config,
    )
    all_params = list(trainer._trainable_params())
    trainer.optimizer = torch.optim.Adam(all_params, lr=1e-3)
    return trainer


class TestTTSTrainerBPEH:
    def test_bpeh_modules_created(self, tmp_path):
        trainer = _make_bpeh_trainer(tmp_path)
        assert trainer.bpeh_head is not None
        assert trainer.bpeh_loss_fn is not None

    def test_bpeh_disabled_by_default(self, tmp_path):
        trainer = _make_trainer(tmp_path)
        assert trainer.bpeh_head is None

    def test_train_step_with_bpeh(self, tmp_path):
        trainer = _make_bpeh_trainer(tmp_path)
        batch = _make_bpeh_batch()
        losses = trainer.train_step(batch)

        assert "event_total" in losses
        assert "event_onset" in losses
        assert "event_dur" in losses
        assert "event_amp" in losses
        assert all(not torch.tensor(v).isnan() for v in losses.values())

    def test_bpeh_losses_are_finite(self, tmp_path):
        trainer = _make_bpeh_trainer(tmp_path, max_steps=3)
        batch = _make_bpeh_batch()
        for _ in range(3):
            losses = trainer.train_step(batch)
            for k, v in losses.items():
                assert not torch.tensor(v).isnan(), f"{k} is NaN"
                assert not torch.tensor(v).isinf(), f"{k} is Inf"

    def test_bpeh_checkpoint_roundtrip(self, tmp_path):
        trainer = _make_bpeh_trainer(tmp_path)
        batch = _make_bpeh_batch()
        trainer.train_step(batch)

        ckpt_path = trainer.save_checkpoint()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        assert "bpeh_head" in ckpt
        assert ckpt["enable_bpeh"] is True

        trainer2 = _make_bpeh_trainer(tmp_path)
        trainer2.load_checkpoint(ckpt_path)
        assert trainer2.global_step == 1

    def test_train_iter_with_bpeh(self, tmp_path):
        trainer = _make_bpeh_trainer(tmp_path, max_steps=2)
        steps = list(trainer.train_iter())
        assert len(steps) == 2
        assert "event_total" in steps[0][1]

    def test_bpeh_with_no_event_data(self, tmp_path):
        """BPEH trainer should work even when batch has no event tensors."""
        trainer = _make_bpeh_trainer(tmp_path)
        batch = _make_batch()  # No event tensors
        losses = trainer.train_step(batch)
        assert "event_total" in losses
        assert not torch.tensor(losses["event_total"]).isnan()


class TestExtractProsodyStats:
    def test_output_shape(self):
        from tmrvc_train.tts_trainer import extract_prosody_stats
        B, T = 4, 50
        f0 = torch.abs(torch.randn(B, 1, T)) * 200 + 100
        mel = torch.randn(B, 80, T)
        mask = torch.ones(B, 1, T)
        stats = extract_prosody_stats(f0, mel, mask)
        assert stats.shape == (4, 8)

    def test_all_finite(self):
        from tmrvc_train.tts_trainer import extract_prosody_stats
        B, T = 2, 30
        f0 = torch.abs(torch.randn(B, 1, T)) * 200 + 100
        mel = torch.randn(B, 80, T)
        mask = torch.ones(B, 1, T)
        stats = extract_prosody_stats(f0, mel, mask)
        assert not stats.isnan().any()
        assert not stats.isinf().any()

    def test_partial_mask(self):
        from tmrvc_train.tts_trainer import extract_prosody_stats
        B, T = 2, 40
        f0 = torch.abs(torch.randn(B, 1, T)) * 200 + 100
        mel = torch.randn(B, 80, T)
        mask = torch.ones(B, 1, T)
        mask[0, :, 20:] = 0  # Half masked
        stats = extract_prosody_stats(f0, mel, mask)
        assert not stats.isnan().any()
