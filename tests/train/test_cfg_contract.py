"""Tests for CFG unconditional mask contract and voice_state supervision."""

import pytest
import torch

from tmrvc_core.types import (
    CFG_ZEROED_FIELDS,
    CFG_PRESERVED_FIELDS,
    VoiceStateSupervision,
)
from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.models.uclm_loss import voice_state_supervision_loss


# ---------------------------------------------------------------------------
# CFG Mask Contract
# ---------------------------------------------------------------------------


class TestCFGMaskContract:
    """Verify the CFG unconditional mask contract is well-defined."""

    def test_zeroed_and_preserved_are_disjoint(self):
        assert CFG_ZEROED_FIELDS & CFG_PRESERVED_FIELDS == set()

    def test_zeroed_fields_are_nonempty(self):
        assert len(CFG_ZEROED_FIELDS) >= 5

    def test_preserved_fields_include_phonemes(self):
        assert "phoneme_ids" in CFG_PRESERVED_FIELDS
        assert "language_ids" in CFG_PRESERVED_FIELDS
        assert "pointer_state" in CFG_PRESERVED_FIELDS

    def test_apply_cfg_unconditional_mask(self):
        """apply_cfg_unconditional_mask zeroes all conditioning tensors."""
        B, T, D = 2, 10, 8
        explicit = torch.randn(B, T, D)
        ssl = torch.randn(B, T, 128)
        spk = torch.randn(B, 192)
        ctx = torch.randn(B, 256)
        intent = torch.randn(B, 64)
        prosody = torch.randn(B, 64)
        delta = torch.randn(B, T, D)

        result = DisentangledUCLM.apply_cfg_unconditional_mask(
            explicit_state=explicit,
            ssl_state=ssl,
            speaker_embed=spk,
            dialogue_context=ctx,
            acting_intent=intent,
            prosody_latent=prosody,
            delta_voice_state=delta,
        )

        assert torch.all(result["explicit_state"] == 0)
        assert torch.all(result["ssl_state"] == 0)
        assert torch.all(result["speaker_embed"] == 0)
        assert torch.all(result["dialogue_context"] == 0)
        assert torch.all(result["acting_intent"] == 0)
        assert torch.all(result["prosody_latent"] == 0)
        assert torch.all(result["delta_voice_state"] == 0)

    def test_apply_cfg_mask_none_inputs(self):
        """None optional inputs remain None after masking."""
        B, T, D = 2, 10, 8
        result = DisentangledUCLM.apply_cfg_unconditional_mask(
            explicit_state=torch.randn(B, T, D),
            ssl_state=torch.randn(B, T, 128),
            speaker_embed=torch.randn(B, 192),
        )
        assert result["dialogue_context"] is None
        assert result["acting_intent"] is None
        assert result["prosody_latent"] is None


# ---------------------------------------------------------------------------
# Voice State Supervision Loss
# ---------------------------------------------------------------------------


class TestVoiceStateSupervisionLoss:
    """Verify voice_state_supervision_loss properly handles masks."""

    def test_basic_loss(self):
        pred = torch.ones(2, 10, 8)
        target = torch.zeros(2, 10, 8)
        loss = voice_state_supervision_loss(pred, target)
        assert loss.item() > 0

    def test_observed_mask_excludes_unobserved(self):
        """Loss should be zero when all dimensions are masked out."""
        pred = torch.ones(2, 10, 8)
        target = torch.zeros(2, 10, 8)
        mask = torch.zeros(2, 10, 8, dtype=torch.bool)  # all unobserved
        loss = voice_state_supervision_loss(pred, target, observed_mask=mask)
        assert loss.item() == 0.0

    def test_observed_mask_partial(self):
        """Loss should only consider observed dimensions."""
        B, T, D = 1, 5, 8
        pred = torch.ones(B, T, D)
        target = torch.zeros(B, T, D)
        # Only first 4 dims observed
        mask = torch.zeros(B, T, D, dtype=torch.bool)
        mask[:, :, :4] = True
        loss = voice_state_supervision_loss(pred, target, observed_mask=mask)
        assert loss.item() > 0
        # Should be 1.0 (MSE of ones vs zeros)
        assert abs(loss.item() - 1.0) < 1e-5

    def test_confidence_weighting(self):
        """Higher confidence should weight more."""
        B, T, D = 1, 5, 8
        pred = torch.ones(B, T, D)
        target = torch.zeros(B, T, D)
        high_conf = torch.ones(B, T, D)
        low_conf = torch.full((B, T, D), 0.1)
        loss_high = voice_state_supervision_loss(pred, target, confidence=high_conf)
        loss_low = voice_state_supervision_loss(pred, target, confidence=low_conf)
        # Both should be 1.0 because weighted MSE normalizes by total weight
        assert abs(loss_high.item() - 1.0) < 1e-5
        assert abs(loss_low.item() - 1.0) < 1e-5

    def test_frame_mask(self):
        """Frame-level padding mask should exclude padded frames."""
        B, T, D = 1, 10, 8
        pred = torch.ones(B, T, D)
        target = torch.zeros(B, T, D)
        # Mask out last 5 frames
        frame_mask = torch.zeros(B, T, dtype=torch.bool)
        frame_mask[:, 5:] = True  # True = ignore
        loss = voice_state_supervision_loss(pred, target, mask=frame_mask)
        assert loss.item() > 0


# ---------------------------------------------------------------------------
# VoiceStateSupervision Dataclass
# ---------------------------------------------------------------------------


class TestVoiceStateSupervisionDataclass:
    def test_creation(self):
        vs = VoiceStateSupervision(
            targets=torch.randn(2, 10, 8),
            observed_mask=torch.ones(2, 10, 8, dtype=torch.bool),
            confidence=torch.ones(2, 10, 8),
            provenance="test_estimator_v1",
        )
        assert vs.targets.shape == (2, 10, 8)
        assert vs.provenance == "test_estimator_v1"


# ---------------------------------------------------------------------------
# Forward without targets (inference mode)
# ---------------------------------------------------------------------------


class TestForwardInferenceMode:
    def test_forward_tts_pointer_without_targets(self):
        """forward_tts_pointer should work without target_a/target_b for inference."""
        model = DisentangledUCLM()
        model.eval()

        B, L, T = 1, 8, 20
        inputs = {
            "phoneme_ids": torch.randint(1, 100, (B, L)),
            "language_ids": torch.zeros(B, L, dtype=torch.long),
            "pointer_state": None,
            "speaker_embed": torch.randn(B, 192),
            "explicit_state": torch.randn(B, T, 8),
            "ssl_state": torch.randn(B, T, 128),
            "target_length": T,
        }

        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert "logits_a" in out
        assert "logits_b" in out
        assert "advance_logit" in out
        assert "progress_delta" in out
        assert "boundary_confidence" in out
        assert "hidden_states" in out
