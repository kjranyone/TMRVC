"""Tests for UCLMTrainer v3 config flags.

Covers:
- UCLMTrainer accepts pointer_target_source
- UCLMTrainer validates pointer_target_source values
- voice_state_loss_weight propagation
- legacy_duration_loss_weight and delta_voice_state_loss_weight propagation
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.trainer import UCLMTrainer


def _make_small_model():
    """Create a minimal DisentangledUCLM for trainer config tests."""
    return DisentangledUCLM(
        d_model=64,
        n_heads=4,
        n_layers=2,
        rvq_vocab_size=128,
        n_codebooks=8,
        control_vocab_size=32,
        d_explicit=8,
        d_ssl=32,
        d_speaker=32,
        vq_bins=32,
        vocab_size=64,
    )


def _make_trainer(**overrides):
    """Create a UCLMTrainer with a small model and configurable kwargs."""
    model = _make_small_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    defaults = dict(
        model=model,
        optimizer=optimizer,
        device="cpu",
    )
    defaults.update(overrides)
    return UCLMTrainer(**defaults)


# ---------------------------------------------------------------------------
# pointer_target_source
# ---------------------------------------------------------------------------


class TestPointerTargetSource:
    def test_default_is_heuristic_bootstrap(self):
        trainer = _make_trainer()
        assert trainer.pointer_target_source == "heuristic_bootstrap"

    def test_accepts_heuristic_bootstrap(self):
        trainer = _make_trainer(pointer_target_source="heuristic_bootstrap")
        assert trainer.pointer_target_source == "heuristic_bootstrap"

    def test_accepts_legacy_duration(self):
        trainer = _make_trainer(pointer_target_source="legacy_duration")
        assert trainer.pointer_target_source == "legacy_duration"

    def test_accepts_mas(self):
        trainer = _make_trainer(pointer_target_source="mas")
        assert trainer.pointer_target_source == "mas"

    def test_accepts_ctc(self):
        trainer = _make_trainer(pointer_target_source="ctc")
        assert trainer.pointer_target_source == "ctc"

    def test_invalid_source_raises_at_init(self):
        """An invalid pointer_target_source should raise ValueError at init."""
        with pytest.raises(ValueError, match="pointer_target_source"):
            _make_trainer(pointer_target_source="nonexistent_source")


# ---------------------------------------------------------------------------
# voice_state_loss_weight
# ---------------------------------------------------------------------------


class TestVoiceStateLossWeight:
    def test_default_is_zero(self):
        trainer = _make_trainer()
        assert trainer.voice_state_loss_weight == 0.0

    def test_propagation(self):
        trainer = _make_trainer(voice_state_loss_weight=0.5)
        assert trainer.voice_state_loss_weight == 0.5

    def test_delta_default_is_zero(self):
        trainer = _make_trainer()
        assert trainer.delta_voice_state_loss_weight == 0.0

    def test_delta_propagation(self):
        trainer = _make_trainer(delta_voice_state_loss_weight=0.3)
        assert trainer.delta_voice_state_loss_weight == 0.3


# ---------------------------------------------------------------------------
# legacy_duration_loss_weight
# ---------------------------------------------------------------------------


class TestLegacyDurationLossWeight:
    def test_default_is_zero(self):
        trainer = _make_trainer()
        assert trainer.legacy_duration_loss_weight == 0.0

    def test_propagation(self):
        trainer = _make_trainer(legacy_duration_loss_weight=0.1)
        assert trainer.legacy_duration_loss_weight == 0.1


# ---------------------------------------------------------------------------
# tts_mode default
# ---------------------------------------------------------------------------


class TestTrainerTtsMode:
    def test_default_is_pointer(self):
        trainer = _make_trainer()
        assert trainer.tts_mode == "pointer"

    def test_accepts_legacy_duration(self):
        trainer = _make_trainer(tts_mode="legacy_duration")
        assert trainer.tts_mode == "legacy_duration"
