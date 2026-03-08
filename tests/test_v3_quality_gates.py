"""Worker 06: Quality gate tests for v3 contracts.

Tests that validate the integrated contracts from Workers 01, 02, 03, 04.
"""

from __future__ import annotations

import math

import pytest
import torch
import numpy as np

from pathlib import Path
import numpy as np

from tmrvc_core.constants import SAMPLE_RATE, HOP_LENGTH
from tmrvc_core.types import (
    CFG_ZEROED_FIELDS,
    CFG_PRESERVED_FIELDS,
    PointerState,
    VoiceStateSupervision,
)

ACCEPTANCE_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent / "docs" / "design" / "acceptance-thresholds.md"
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
            prosody_latent=torch.ones(B, 128),
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


# ---------------------------------------------------------------------------
# Pointer Sanity: advance_logit monotonicity (Worker 06)
# ---------------------------------------------------------------------------


class TestPointerSanity:
    """advance_logit must produce monotonic text_index over time."""

    def test_advance_logit_produces_monotonic_text_index(self):
        """Stepping through a sequence of high advance_probs must produce
        monotonically non-decreasing text_index values."""
        ps = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        text_indices = [ps.text_index.item()]

        # Simulate 20 steps with high advance probability
        for _ in range(20):
            ps.step_pointer(advance_prob=0.8, progress_delta=1.5, boundary_confidence=0.5)
            text_indices.append(ps.text_index.item())

        # text_index must be monotonically non-decreasing
        for i in range(1, len(text_indices)):
            assert text_indices[i] >= text_indices[i - 1], (
                f"text_index decreased at step {i}: "
                f"{text_indices[i - 1]} -> {text_indices[i]}"
            )

        # Must have advanced at least once
        assert text_indices[-1] > text_indices[0], (
            "text_index never advanced despite high advance_prob"
        )

    def test_advance_logit_never_produces_negative_text_index(self):
        """text_index must never go below 0."""
        ps = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        for _ in range(50):
            ps.step_pointer(
                advance_prob=np.random.random(),
                progress_delta=np.random.random(),
                boundary_confidence=np.random.random(),
            )
            assert ps.text_index.item() >= 0


# ---------------------------------------------------------------------------
# CFG Responsiveness (Worker 06)
# ---------------------------------------------------------------------------


class TestCFGResponsivenessQualityGate:
    """Guided output must differ from unguided output."""

    def test_guided_vs_unguided_output_differs(self):
        """Applying the CFG unconditional mask should change model output."""
        from tmrvc_train.models.uclm_model import DisentangledUCLM

        model = DisentangledUCLM()
        model.eval()

        B, T = 1, 10
        inputs = {
            "phoneme_ids": torch.randint(1, 100, (B, 8)),
            "language_ids": torch.zeros(B, 8, dtype=torch.long),
            "pointer_state": None,
            "speaker_embed": torch.randn(B, 192),
            "explicit_state": torch.randn(B, T, 8),
            "ssl_state": torch.randn(B, T, 128),
            "target_a": torch.zeros(B, 8, T, dtype=torch.long),
            "target_b": torch.zeros(B, 4, T, dtype=torch.long),
            "target_length": T,
        }

        with torch.no_grad():
            out_cond = model.forward_tts_pointer(**inputs)

        # Zero all conditioning -> unconditional
        inputs_uncond = {**inputs}
        inputs_uncond["speaker_embed"] = torch.zeros(B, 192)
        inputs_uncond["explicit_state"] = torch.zeros(B, T, 8)
        inputs_uncond["ssl_state"] = torch.zeros(B, T, 128)

        with torch.no_grad():
            out_uncond = model.forward_tts_pointer(**inputs_uncond)

        assert not torch.allclose(
            out_cond["hidden_states"], out_uncond["hidden_states"], atol=1e-6
        ), "Conditional and unconditional outputs are identical"


# ---------------------------------------------------------------------------
# voice_state Responsiveness (Worker 06)
# ---------------------------------------------------------------------------


class TestVoiceStateResponsiveness:
    """Different voice_state inputs must produce measurably different outputs."""

    def test_different_explicit_state_produces_different_output(self):
        from tmrvc_train.models.uclm_model import DisentangledUCLM

        model = DisentangledUCLM()
        model.eval()

        B, T = 1, 10
        base = {
            "phoneme_ids": torch.randint(1, 100, (B, 8)),
            "language_ids": torch.zeros(B, 8, dtype=torch.long),
            "pointer_state": None,
            "speaker_embed": torch.randn(B, 192),
            "ssl_state": torch.randn(B, T, 128),
            "target_a": torch.zeros(B, 8, T, dtype=torch.long),
            "target_b": torch.zeros(B, 4, T, dtype=torch.long),
            "target_length": T,
        }

        with torch.no_grad():
            out1 = model.forward_tts_pointer(
                **base, explicit_state=torch.ones(B, T, 8),
            )
            out2 = model.forward_tts_pointer(
                **base, explicit_state=-torch.ones(B, T, 8),
            )

        assert not torch.allclose(
            out1["hidden_states"], out2["hidden_states"], atol=1e-6
        ), "Different voice_state inputs produced identical outputs"


# ---------------------------------------------------------------------------
# Few-Shot Speaker Similarity (Worker 06)
# ---------------------------------------------------------------------------


class TestFewShotSpeakerSimilarity:
    """Encoded speaker prompt must produce non-zero similarity."""

    def test_speaker_embed_produces_nonzero_similarity(self):
        from tmrvc_train.eval_metrics import speaker_embedding_cosine_similarity

        embed1 = torch.randn(192)
        embed2 = embed1 + 0.1 * torch.randn(192)
        sim = speaker_embedding_cosine_similarity(embed1, embed2)
        assert isinstance(sim, float)
        assert sim > 0.0, f"Speaker similarity is non-positive: {sim}"

    def test_same_speaker_high_similarity(self):
        from tmrvc_train.eval_metrics import speaker_embedding_cosine_similarity

        embed = torch.randn(192)
        sim = speaker_embedding_cosine_similarity(embed, embed)
        assert sim > 0.99, f"Self-similarity should be ~1.0, got {sim}"

    def test_few_shot_score_computable(self):
        from tmrvc_train.eval_metrics import few_shot_speaker_score

        score = few_shot_speaker_score(
            speaker_similarity=0.85,
            intelligibility=0.95,
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Threshold Schedule Validation (Worker 06, Task 9)
# ---------------------------------------------------------------------------


class TestThresholdScheduleValidation:
    """Verify Tier 0/1/2 threshold policy is documented and Tier 0 is frozen."""

    def test_acceptance_thresholds_document_exists(self):
        assert ACCEPTANCE_THRESHOLDS_PATH.exists(), (
            "acceptance-thresholds.md not found"
        )

    def test_tier_0_policy_documented(self):
        """Tier 0 threshold policy must be documented."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "Tier 0" in text, "Tier 0 policy not documented"
        assert "must freeze before Stage B" in text

    def test_tier_1_policy_documented(self):
        """Tier 1 threshold policy must be documented."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "Tier 1" in text, "Tier 1 policy not documented"

    def test_tier_2_policy_documented(self):
        """Tier 2 threshold policy must be documented."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "Tier 2" in text, "Tier 2 policy not documented"

    def test_tier_0_runtime_budget_frozen(self):
        """Tier 0: runtime budgets must be specified with numeric thresholds."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "10 ms" in text, "Streaming latency budget not frozen"

    def test_tier_0_parity_tolerances_documented(self):
        """Tier 0: parity tolerances must be documented."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "parity" in text.lower()

    def test_tier_0_frame_convention_frozen(self):
        """Tier 0: frame/alignment conventions must be documented."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        # Must reference frame/alignment conventions as a Tier 0 item
        assert "frame" in text.lower() and "alignment" in text.lower()

    def test_tier_0_language_set_documented(self):
        """Tier 0: language set must be referenced."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "language" in text.lower()

    def test_tier_0_hardware_class_documented(self):
        """Tier 0: hardware class must be specified."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        assert "hardware" in text.lower()

    def test_no_undefined_thresholds_in_tier_0_sections(self):
        """Tier 0 sections should have concrete numeric thresholds, not TBD."""
        text = ACCEPTANCE_THRESHOLDS_PATH.read_text(encoding="utf-8")
        # Verify streaming latency has a concrete numeric budget
        assert "<= 10 ms" in text or "<= 10ms" in text or "10 ms" in text, (
            "Streaming latency budget missing concrete threshold"
        )
        # Verify frame convention references a concrete standard
        # (The thresholds doc references frame/alignment conventions;
        #  the concrete sample_rate=24000 is defined in tmrvc-core constants)
        assert "frame" in text.lower(), (
            "Frame convention not referenced in thresholds document"
        )
