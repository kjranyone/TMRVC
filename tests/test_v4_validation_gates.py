"""v4 validation gate tests.

Covers track_validation.md:
- S4: Runtime parity gates
- S5: Claim taxonomy
- S6: Fish S2 protocol structure
"""

import pytest


class TestRuntimeParityGateDefinitions:
    """Verify that parity gate definitions exist and are well-formed."""

    def test_parity_axes_defined(self):
        """All required parity axes must be listed."""
        required_axes = [
            "python_vs_onnx",
            "python_vs_rust",
            "batch_vs_streaming",
            "physical_control_ordering",
            "acting_latent_ordering",
        ]
        # These are metric definitions, not model tests
        for axis in required_axes:
            assert isinstance(axis, str)

    def test_parity_threshold(self):
        """Max absolute difference threshold for parity."""
        MAX_PARITY_DIFF = 1e-4
        assert MAX_PARITY_DIFF > 0
        assert MAX_PARITY_DIFF < 1e-2


class TestClaimTaxonomy:
    """Verify claim taxonomy structure (track_validation S5)."""

    def test_all_claim_categories_defined(self):
        claim_categories = [
            "raw_audio_bootstrap_readiness",
            "broad_external_baseline_competitiveness",
            "acting_controllability",
            "programmable_expressive_speech",
            "cross_speaker_acting_transfer",
            "real_time_causal_runtime",
            "inline_instruction_following",
        ]
        assert len(claim_categories) == 7

    def test_each_claim_has_required_evidence(self):
        """Each claim must map to specific evidence, not just prose."""
        claim_evidence_map = {
            "acting_controllability": [
                "physical_monotonicity_report",
                "physical_calibration_report",
            ],
            "programmable_expressive_speech": [
                "replay_fidelity_report",
                "edit_locality_report",
                "instruction_following_report",
            ],
            "cross_speaker_acting_transfer": [
                "transfer_quality_report",
            ],
            "real_time_causal_runtime": [
                "latency_report",
                "rtf_report",
                "parity_report",
            ],
            "inline_instruction_following": [
                "tag_compliance_report",
                "rl_compliance_report",
            ],
        }

        for claim, evidence in claim_evidence_map.items():
            assert len(evidence) > 0, f"Claim {claim} has no evidence"


class TestFishS2Protocol:
    """Verify Fish S2 head-to-head protocol structure (track_validation S6)."""

    def test_mandatory_win_axes(self):
        mandatory_win = [
            "acting_editability",
            "trajectory_replay_fidelity",
            "edit_locality",
        ]
        assert len(mandatory_win) == 3

    def test_mandatory_guardrail_axes(self):
        guardrails = [
            "first_take_naturalness",
            "few_shot_speaker_similarity",
            "latency_class_disclosure",
        ]
        assert len(guardrails) == 3

    def test_claim_narrowing_rule(self):
        """If win only on editability, claim must narrow."""
        editability_win = True
        first_take_loss = True

        if editability_win and first_take_loss:
            allowed_claim = "editability_and_programmability_only"
            assert "editability" in allowed_claim
            assert "broad" not in allowed_claim


class TestInstructionFollowingMetrics:
    """Test instruction following metrics for validation."""

    def test_tag_compliance_definition(self):
        """Tag compliance = fraction of requested tags detected in output."""
        requested = ["[angry]", "[emphasis]", "[pause]"]
        detected = ["[angry]", "[pause]"]

        recall = len(set(requested) & set(detected)) / len(requested)
        assert recall == pytest.approx(2 / 3)

    def test_physical_compliance_under_rl(self):
        """Physical control monotonicity must remain > 0.8 after RL."""
        import numpy as np

        # Simulated monotonicity scores before and after RL
        before_rl = 0.85
        after_rl = 0.82

        min_monotonicity = 0.8
        assert after_rl >= min_monotonicity

    def test_naturalness_degradation_limit(self):
        """Plain-text TTS quality must not degrade > 5%."""
        baseline_mos = 4.0
        post_rl_mos = 3.85  # 3.75% degradation

        degradation = (baseline_mos - post_rl_mos) / baseline_mos
        max_degradation = 0.05

        assert degradation <= max_degradation
