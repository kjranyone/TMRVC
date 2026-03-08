"""Tests for UCLM v3 pointer-based TTS: model shapes, PointerHead, backward
compatibility, pointer losses, and trainer pointer mode."""

from __future__ import annotations

import copy

import pytest
import torch

from tmrvc_core.types import PointerState
from tmrvc_train.models.uclm_model import DisentangledUCLM, PointerHead
from tmrvc_train.models.uclm_loss import (
    pointer_advance_loss,
    progress_regression_loss,
    uclm_loss,
)
from tmrvc_train.trainer import UCLMTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_D_MODEL = 256
_N_HEADS = 4
_N_LAYERS = 2
_B = 2
_T = 20  # acoustic frames
_L = 10  # phoneme length
_N_CODEBOOKS = 8
_N_CONTROL = 4


def _make_small_model() -> DisentangledUCLM:
    return DisentangledUCLM(
        d_model=_D_MODEL,
        n_heads=_N_HEADS,
        n_layers=_N_LAYERS,
        num_speakers=8,
    )


def _make_pointer_inputs(model: DisentangledUCLM) -> dict:
    """Build minimal inputs for forward_tts_pointer on CPU."""
    phoneme_ids = torch.randint(1, 200, (_B, _L))
    language_ids = torch.zeros(_B, dtype=torch.long)
    speaker_embed = torch.randn(_B, 192)
    explicit_state = torch.randn(_B, _T, 8)
    ssl_state = torch.randn(_B, _T, 128)
    target_a = torch.randint(0, 1024, (_B, _N_CODEBOOKS, _T))
    target_b = torch.randint(0, 60, (_B, _N_CONTROL, _T))
    return dict(
        phoneme_ids=phoneme_ids,
        language_ids=language_ids,
        pointer_state=None,
        speaker_embed=speaker_embed,
        explicit_state=explicit_state,
        ssl_state=ssl_state,
        target_a=target_a,
        target_b=target_b,
        target_length=_T,
    )


# ---------------------------------------------------------------------------
# a) Pointer model shape tests
# ---------------------------------------------------------------------------


class TestForwardTtsPointerShapes:
    def test_output_keys(self):
        model = _make_small_model()
        model.eval()
        inputs = _make_pointer_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        expected_keys = {"logits_a", "logits_b", "pointer_logits", "progress_delta"}
        assert expected_keys.issubset(set(out.keys())), (
            f"Missing keys: {expected_keys - set(out.keys())}"
        )

    def test_logits_a_shape(self):
        model = _make_small_model()
        model.eval()
        inputs = _make_pointer_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        # logits_a: [B, n_codebooks, T, rvq_vocab_size]
        assert out["logits_a"].shape[0] == _B
        assert out["logits_a"].shape[1] == model.n_codebooks
        assert out["logits_a"].shape[2] == _T
        assert out["logits_a"].shape[3] == model.rvq_vocab_size

    def test_logits_b_shape(self):
        model = _make_small_model()
        model.eval()
        inputs = _make_pointer_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        # logits_b: [B, n_control, T, control_vocab_size]
        assert out["logits_b"].shape[0] == _B
        assert out["logits_b"].shape[2] == _T
        assert out["logits_b"].shape[3] == model.control_vocab_size

    def test_pointer_logits_shape(self):
        model = _make_small_model()
        model.eval()
        inputs = _make_pointer_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert out["pointer_logits"].shape == (_B, _T, 1)

    def test_progress_delta_shape(self):
        model = _make_small_model()
        model.eval()
        inputs = _make_pointer_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert out["progress_delta"].shape == (_B, _T, 1)

    def test_hidden_states_present(self):
        model = _make_small_model()
        model.eval()
        inputs = _make_pointer_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert "hidden_states" in out
        assert out["hidden_states"].shape[0] == _B
        assert out["hidden_states"].shape[1] == _T


# ---------------------------------------------------------------------------
# b) PointerHead output shape test
# ---------------------------------------------------------------------------


class TestPointerHead:
    def test_output_shapes(self):
        head = PointerHead(d_model=_D_MODEL)
        x = torch.randn(2, 50, _D_MODEL)
        advance_logit, progress_delta, boundary_confidence = head(x)

        assert advance_logit.shape == (2, 50, 1)
        assert progress_delta.shape == (2, 50, 1)
        assert boundary_confidence.shape == (2, 50, 1)

    def test_progress_delta_bounded(self):
        """progress_delta should be in [0, 1] because of Sigmoid."""
        head = PointerHead(d_model=_D_MODEL)
        x = torch.randn(2, 50, _D_MODEL)
        _, progress_delta, _ = head(x)

        assert progress_delta.min() >= 0.0
        assert progress_delta.max() <= 1.0


# ---------------------------------------------------------------------------
# c) v2 checkpoint backward compatibility
# ---------------------------------------------------------------------------


class TestV2CheckpointBackwardCompat:
    def test_load_v2_state_dict_strict_false(self):
        """A v3 model (with pointer_head) should load a v2 state_dict via strict=False."""
        model_v3 = _make_small_model()
        full_state = model_v3.state_dict()

        # Simulate v2 state_dict by removing pointer_head keys
        v2_state = {
            k: v for k, v in full_state.items() if not k.startswith("pointer_head.")
        }
        assert len(v2_state) < len(full_state), "pointer_head keys should have been removed"

        # Create a fresh v3 model and load the v2 state dict
        model_v3_new = _make_small_model()
        missing, unexpected = model_v3_new.load_state_dict(v2_state, strict=False)

        # pointer_head keys should be in missing
        assert any("pointer_head" in k for k in missing)
        assert len(unexpected) == 0

    def test_load_v2_state_dict_strict_raises(self):
        """strict=True should fail when pointer_head keys are absent."""
        model = _make_small_model()
        v2_state = {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("pointer_head.")
        }
        with pytest.raises(RuntimeError):
            _make_small_model().load_state_dict(v2_state, strict=True)


# ---------------------------------------------------------------------------
# d) Pointer loss tests
# ---------------------------------------------------------------------------


class TestPointerLosses:
    def test_pointer_advance_loss_scalar(self):
        logits = torch.randn(2, 30, 1)
        targets = torch.randint(0, 2, (2, 30))
        loss = pointer_advance_loss(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0.0

    def test_pointer_advance_loss_with_mask(self):
        logits = torch.randn(2, 30, 1)
        targets = torch.randint(0, 2, (2, 30))
        mask = torch.zeros(2, 30, dtype=torch.bool)
        mask[:, 25:] = True  # mask out last 5 frames

        loss_masked = pointer_advance_loss(logits, targets, mask=mask)
        assert loss_masked.dim() == 0

    def test_pointer_advance_loss_all_masked(self):
        logits = torch.randn(2, 30, 1)
        targets = torch.randint(0, 2, (2, 30))
        mask = torch.ones(2, 30, dtype=torch.bool)  # all masked
        loss = pointer_advance_loss(logits, targets, mask=mask)
        assert loss.item() == 0.0

    def test_progress_regression_loss_scalar(self):
        pred = torch.rand(2, 30, 1)  # after sigmoid, values in [0,1]
        targets = torch.rand(2, 30)
        loss = progress_regression_loss(pred, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_progress_regression_loss_with_mask(self):
        pred = torch.rand(2, 30, 1)
        targets = torch.rand(2, 30)
        mask = torch.zeros(2, 30, dtype=torch.bool)
        mask[:, 20:] = True
        loss = progress_regression_loss(pred, targets, mask=mask)
        assert loss.dim() == 0

    def test_uclm_loss_with_pointer_components(self):
        """uclm_loss should incorporate pointer losses when provided."""
        B, T = 2, 20
        n_cb, vocab_a = 8, 1024
        n_slots, vocab_b = 4, 64

        logits_a = torch.randn(B, n_cb, T, vocab_a)
        logits_b = torch.randn(B, n_slots, T, vocab_b)
        target_a = torch.randint(0, vocab_a, (B, n_cb, T))
        target_b = torch.randint(0, vocab_b, (B, n_slots, T))
        pointer_logits = torch.randn(B, T, 1)
        advance_targets = torch.randint(0, 2, (B, T))
        progress_delta = torch.rand(B, T, 1)
        progress_targets = torch.rand(B, T)

        losses = uclm_loss(
            logits_a=logits_a,
            logits_b=logits_b,
            target_a=target_a,
            target_b=target_b,
            pointer_logits=pointer_logits,
            advance_targets=advance_targets,
            progress_delta=progress_delta,
            progress_targets=progress_targets,
        )

        assert "loss" in losses
        assert "loss_pointer" in losses
        assert "loss_progress" in losses
        assert losses["loss"].dim() == 0

    def test_uclm_loss_without_pointer_components(self):
        """uclm_loss should work without pointer arguments (v2 compat)."""
        B, T = 2, 20
        n_cb, vocab_a = 8, 1024
        n_slots, vocab_b = 4, 64

        losses = uclm_loss(
            logits_a=torch.randn(B, n_cb, T, vocab_a),
            logits_b=torch.randn(B, n_slots, T, vocab_b),
            target_a=torch.randint(0, vocab_a, (B, n_cb, T)),
            target_b=torch.randint(0, vocab_b, (B, n_slots, T)),
        )

        assert "loss" in losses
        assert "loss_pointer" not in losses
        assert "loss_progress" not in losses


# ---------------------------------------------------------------------------
# e) Trainer pointer mode test
# ---------------------------------------------------------------------------


class TestTrainerPointerMode:
    def test_train_step_pointer_mode_no_durations(self):
        """UCLMTrainer.train_step should work with tts_mode='pointer' and no
        durations in batch."""
        model = _make_small_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = UCLMTrainer(model, optimizer, device="cpu", tts_prob=1.0, tts_mode="pointer")

        batch = {
            "target_a": torch.randint(0, 1024, (_B, _N_CODEBOOKS, _T)),
            "target_b": torch.randint(0, 64, (_B, _N_CONTROL, _T)),
            "source_a_t": torch.randint(0, 1024, (_B, _N_CODEBOOKS, _T)),
            "explicit_state": torch.randn(_B, _T, 8),
            "ssl_state": torch.randn(_B, _T, 128),
            "speaker_embed": torch.randn(_B, 192),
            "speaker_id": torch.zeros(_B, dtype=torch.long),
            "phoneme_ids": torch.randint(1, 200, (_B, _L)),
            "phoneme_lens": torch.full((_B,), _L, dtype=torch.long),
            "language_id": torch.zeros(_B, dtype=torch.long),
            # NOTE: no "durations" key
        }

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        # In pointer mode with tts_prob=1.0, mode should be TTS
        assert metrics["mode"] == 1

    def test_train_step_pointer_mode_falls_back_to_vc_without_phonemes(self):
        """Without phoneme_ids the trainer should fall back to VC mode."""
        model = _make_small_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        trainer = UCLMTrainer(model, optimizer, device="cpu", tts_prob=1.0, tts_mode="pointer")

        batch = {
            "target_a": torch.randint(0, 1024, (_B, _N_CODEBOOKS, _T)),
            "target_b": torch.randint(0, 64, (_B, _N_CONTROL, _T)),
            "source_a_t": torch.randint(0, 1024, (_B, _N_CODEBOOKS, _T)),
            "explicit_state": torch.randn(_B, _T, 8),
            "ssl_state": torch.randn(_B, _T, 128),
            "speaker_embed": torch.randn(_B, 192),
            "speaker_id": torch.zeros(_B, dtype=torch.long),
            # No phoneme_ids -> should fall back to VC
        }

        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["mode"] == 0  # VC
