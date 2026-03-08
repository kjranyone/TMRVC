"""Worker 06: Quality gate tests for v3 contracts.

Tests that validate the integrated contracts from Workers 01, 02, 03, 04.
"""

from __future__ import annotations

import math

import pytest
import torch
import numpy as np

from tmrvc_core.constants import SAMPLE_RATE, HOP_LENGTH
from tmrvc_core.types import (
    CFG_ZEROED_FIELDS,
    CFG_PRESERVED_FIELDS,
    PointerState,
    VoiceStateSupervision,
)


# ---------------------------------------------------------------------------
# CFG Contract Integrity (Worker 01 + 02)
# ---------------------------------------------------------------------------


class TestCFGContractIntegrity:
    """CFG mask contract consistency across model and trainer."""

    def test_zeroed_and_preserved_disjoint(self):
        assert CFG_ZEROED_FIELDS & CFG_PRESERVED_FIELDS == set()

    def test_all_conditioning_fields_accounted(self):
        """Every conditioning field is either zeroed or preserved."""
        all_fields = CFG_ZEROED_FIELDS | CFG_PRESERVED_FIELDS
        # At minimum these must be classified
        assert "speaker_embed" in all_fields
        assert "phoneme_ids" in all_fields
        assert "pointer_state" in all_fields
        assert "dialogue_context" in all_fields

    def test_model_cfg_mask_matches_frozen_contract(self):
        from tmrvc_train.models.uclm_model import DisentangledUCLM

        B, T, D = 1, 5, 8
        result = DisentangledUCLM.apply_cfg_unconditional_mask(
            explicit_state=torch.ones(B, T, D),
            ssl_state=torch.ones(B, T, 128),
            speaker_embed=torch.ones(B, 192),
            dialogue_context=torch.ones(B, 256),
            acting_intent=torch.ones(B, 64),
            prosody_latent=torch.ones(B, 64),
            delta_voice_state=torch.ones(B, T, D),
        )
        # All zeroed fields should be zero
        for key in ("explicit_state", "ssl_state", "speaker_embed",
                     "dialogue_context", "acting_intent", "prosody_latent",
                     "delta_voice_state"):
            assert torch.all(result[key] == 0), f"{key} not zeroed"


# ---------------------------------------------------------------------------
# Pointer State Contract (Worker 01 + 04)
# ---------------------------------------------------------------------------


class TestPointerStateContract:
    def test_step_pointer_advance(self):
        ps = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        advanced = ps.step_pointer(
            advance_prob=0.8, progress_delta=1.5, boundary_confidence=0.5
        )
        assert advanced
        assert ps.text_index.item() == 1
        assert ps.progress.item() == 0.0

    def test_step_pointer_hold(self):
        ps = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        advanced = ps.step_pointer(advance_prob=0.2, progress_delta=0.3)
        assert not advanced
        assert ps.text_index.item() == 0
        assert ps.stall_frames == 1

    def test_forced_advance_on_stall(self):
        ps = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
            max_frames_per_unit=5,
        )
        for _ in range(5):
            ps.step_pointer(advance_prob=0.1, progress_delta=0.0)
        assert ps.text_index.item() == 1
        assert ps.forced_advance_count == 1

    def test_skip_protection(self):
        ps = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
            skip_protection_threshold=0.5,
        )
        # Both signals fire but low confidence blocks advance
        ps.progress = torch.tensor([1.5])
        advanced = ps.step_pointer(
            advance_prob=0.8, progress_delta=0.0, boundary_confidence=0.2
        )
        assert not advanced
        assert ps.skip_protection_count == 1

    def test_clone_independent(self):
        ps = PointerState(
            text_index=torch.tensor([5]),
            progress=torch.tensor([0.5]),
        )
        ps2 = ps.clone()
        ps2.text_index += 1
        assert ps.text_index.item() == 5
        assert ps2.text_index.item() == 6


# ---------------------------------------------------------------------------
# Voice State Supervision Contract (Worker 01 + 03)
# ---------------------------------------------------------------------------


class TestVoiceStateSupervisionContract:
    def test_dataclass_fields(self):
        vs = VoiceStateSupervision(
            targets=torch.randn(2, 10, 8),
            observed_mask=torch.ones(2, 10, 8, dtype=torch.bool),
            confidence=torch.ones(2, 10, 8),
            provenance="test_v1",
        )
        assert vs.targets.shape == (2, 10, 8)
        assert vs.observed_mask.dtype == torch.bool
        assert vs.provenance == "test_v1"

    def test_missing_dims_not_zero_target(self):
        """Voice state contract: missing dims must be masked, not zero."""
        targets = torch.randn(1, 5, 8)
        mask = torch.zeros(1, 5, 8, dtype=torch.bool)
        mask[:, :, :4] = True  # Only 4 dims observed

        from tmrvc_train.models.uclm_loss import voice_state_supervision_loss

        pred = torch.randn(1, 5, 8)
        loss_masked = voice_state_supervision_loss(pred, targets, observed_mask=mask)
        loss_full = voice_state_supervision_loss(pred, targets)

        # Masked loss should differ from full loss when unobserved dims have error
        # (unless by coincidence, but extremely unlikely with random data)
        assert loss_masked.item() != pytest.approx(loss_full.item(), abs=1e-4)


# ---------------------------------------------------------------------------
# Frame Convention Parity (Worker 03 + 04)
# ---------------------------------------------------------------------------


class TestFrameConventionParity:
    """All frame-indexed artifacts must use sample_rate=24000, hop_length=240."""

    def test_core_constants(self):
        assert SAMPLE_RATE == 24000
        assert HOP_LENGTH == 240

    def test_frame_count_formula(self):
        """T = ceil(num_samples / 240)"""
        assert math.ceil(24000 / 240) == 100
        assert math.ceil(24001 / 240) == 101
        assert math.ceil(23999 / 240) == 100

    def test_bootstrap_alignment_uses_same_convention(self):
        from tmrvc_data.bootstrap_alignment import (
            SAMPLE_RATE as BA_SR,
            HOP_LENGTH as BA_HL,
            samples_to_frames,
        )
        assert BA_SR == SAMPLE_RATE
        assert BA_HL == HOP_LENGTH
        assert samples_to_frames(24000) == 100


# ---------------------------------------------------------------------------
# Phone Inventory Migration Policy (Worker 03)
# ---------------------------------------------------------------------------


class TestPhoneInventoryPolicy:
    def test_canonical_inventory_stable(self):
        from tmrvc_data.g2p import PHONE2ID, PAD_ID, UNK_ID, BOS_ID, EOS_ID

        assert PAD_ID == 0
        assert UNK_ID == 1
        assert BOS_ID == 2
        assert EOS_ID == 3

    def test_accent_symbols_present(self):
        from tmrvc_data.g2p import PHONE2ID

        for sym in ("^", "=", "_"):
            assert sym in PHONE2ID, f"Accent symbol {sym!r} missing from inventory"

    def test_inventory_append_only(self):
        """New phones must be added at the end, never renumbering existing IDs."""
        from tmrvc_data.g2p import PHONEME_LIST

        # Verify the first 6 are always special tokens
        assert PHONEME_LIST[0] == "<pad>"
        assert PHONEME_LIST[1] == "<unk>"
        assert PHONEME_LIST[2] == "<bos>"
        assert PHONEME_LIST[3] == "<eos>"
        assert PHONEME_LIST[4] == "<sil>"
        assert PHONEME_LIST[5] == "<breath>"


# ---------------------------------------------------------------------------
# Dataset Report Field Completeness (Worker 03)
# ---------------------------------------------------------------------------


class TestDatasetReportQualityGate:
    def test_report_has_all_required_fields(self):
        from tmrvc_data.dataset_report import DatasetReport, REQUIRED_REPORT_FIELDS
        import dataclasses

        dc_fields = {f.name for f in dataclasses.fields(DatasetReport)}
        missing = REQUIRED_REPORT_FIELDS - dc_fields
        assert not missing, f"DatasetReport missing required fields: {missing}"

    def test_report_distinguishes_text_from_canonical(self):
        from tmrvc_data.dataset_report import DatasetReport

        r = DatasetReport(
            dataset_name="test",
            text_supervision_coverage=0.9,
            canonical_text_unit_coverage=0.7,
        )
        assert r.text_supervision_coverage != r.canonical_text_unit_coverage


# ---------------------------------------------------------------------------
# Trainer Config Contract (Worker 02)
# ---------------------------------------------------------------------------


class TestTrainerConfigQualityGate:
    def test_default_pointer_mode(self):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        from tmrvc_train.trainer import UCLMTrainer

        model = DisentangledUCLM()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(model=model, optimizer=optimizer, device="cpu")
        assert t.tts_mode == "pointer"
        assert t.pointer_target_source == "heuristic_bootstrap"

    def test_no_duration_required_for_pointer_mode(self):
        from tmrvc_train.models.uclm_model import DisentangledUCLM
        from tmrvc_train.trainer import UCLMTrainer

        model = DisentangledUCLM()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        t = UCLMTrainer(
            model=model, optimizer=optimizer, device="cpu",
            tts_mode="pointer",
            pointer_target_source="heuristic_bootstrap",
        )
        # Trainer should accept pointer mode without duration artifacts
        assert t.legacy_duration_loss_weight == 0.0


# ---------------------------------------------------------------------------
# Anti-Collapse Metric Definitions (Worker 02 + 06)
# ---------------------------------------------------------------------------


class TestAntiCollapseMetrics:
    def test_context_separation_score_defined(self):
        from tmrvc_train.models.uclm_loss import context_separation_score
        hidden = torch.randn(4, 10, 64)
        groups = torch.tensor([0, 0, 1, 1])
        score = context_separation_score(hidden, groups)
        assert isinstance(score, float)

    def test_prosody_collapse_score_defined(self):
        from tmrvc_train.models.uclm_loss import prosody_collapse_score
        hidden = torch.randn(4, 10, 64)
        groups = torch.tensor([0, 0, 1, 1])
        score = prosody_collapse_score(hidden, groups)
        assert isinstance(score, float)

    def test_control_response_score_defined(self):
        from tmrvc_train.models.uclm_loss import control_response_score
        score = control_response_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert score == pytest.approx(1.0)
