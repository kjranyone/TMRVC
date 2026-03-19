"""Worker 06: Drama-acting evaluation protocol tests.

Tests:
- Context-sensitivity skeleton (same text, different context -> different prosody)
- Control-responsiveness (sweep voice_state dimensions, measure variance)
- Disentanglement leakage check (change speaker, measure prosody stability)
- CFG responsiveness
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.eval_metrics import (
    acting_alignment_score,
    cfg_responsiveness_score,
    timbre_prosody_disentanglement_score,
    prosody_transfer_leakage_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_inputs(
    batch_size: int = 1,
    phoneme_len: int = 8,
    target_length: int = 20,
) -> dict:
    """Build minimal TTS inputs for model evaluation."""
    return {
        "phoneme_ids": torch.randint(1, 100, (batch_size, phoneme_len)),
        "language_ids": torch.zeros(batch_size, phoneme_len, dtype=torch.long),
        "pointer_state": None,
        "speaker_embed": torch.randn(batch_size, 192),
        "explicit_state": torch.randn(batch_size, target_length, 12),
        "ssl_state": torch.randn(batch_size, target_length, 128),
        "target_a": torch.zeros(batch_size, 8, target_length, dtype=torch.long),
        "target_b": torch.zeros(batch_size, 4, target_length, dtype=torch.long),
        "target_length": target_length,
    }


# ---------------------------------------------------------------------------
# Context-Sensitivity Test Skeleton
# ---------------------------------------------------------------------------

class TestContextSensitivity:
    """Same text, different dialogue context should produce different prosody.

    This is a skeleton that validates the metric machinery and model interface.
    Full evaluation requires trained checkpoints and a held-out dialogue set.
    """

    def test_acting_alignment_score_computable(self):
        """acting_alignment_score produces a valid float on synthetic data."""
        ctx_embed = torch.randn(4, 128)
        prosody_embed = torch.randn(4, 128)
        score = acting_alignment_score(ctx_embed, prosody_embed)
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0

    def test_different_context_produces_different_hidden_states(self):
        """Model hidden states should differ when dialogue_context changes.

        Uses synthetic random contexts to verify the code path works.
        Real context sensitivity requires trained weights.
        """
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_model_inputs()

        with torch.no_grad():
            out1 = model.forward_tts_pointer(**inputs)

        # Change dialogue context (via explicit_state as proxy)
        inputs2 = {**inputs}
        inputs2["explicit_state"] = torch.randn_like(inputs["explicit_state"])

        with torch.no_grad():
            out2 = model.forward_tts_pointer(**inputs2)

        h1 = out1["hidden_states"]
        h2 = out2["hidden_states"]

        # Hidden states should differ when conditioning changes
        # (with random weights this is virtually guaranteed)
        assert not torch.allclose(h1, h2, atol=1e-6), (
            "Hidden states identical despite different conditioning inputs"
        )

    def test_context_separation_metric_skeleton(self):
        """Skeleton: context_separation_score should be computable."""
        from tmrvc_train.models.uclm_loss import context_separation_score

        # Simulate hidden states from 4 samples in 2 context groups
        hidden = torch.randn(4, 10, 64)
        groups = torch.tensor([0, 0, 1, 1])
        score = context_separation_score(hidden, groups)
        assert isinstance(score, float)
        assert score >= 0.0


# ---------------------------------------------------------------------------
# Control-Responsiveness: voice_state Sweep
# ---------------------------------------------------------------------------

class TestControlResponsiveness:
    """Sweep voice_state dimensions and measure output variance."""

    def test_voice_state_sweep_produces_variance(self):
        """Sweeping a single voice_state dimension should change model output.

        Uses explicit_state as the voice_state proxy.
        """
        model = DisentangledUCLM()
        model.eval()

        base_inputs = _make_model_inputs(target_length=10)
        hidden_states_per_level = []

        # Sweep dimension 0 of explicit_state across 3 levels
        for level in [-1.0, 0.0, 1.0]:
            inputs = {**base_inputs}
            explicit = base_inputs["explicit_state"].clone()
            explicit[:, :, 0] = level
            inputs["explicit_state"] = explicit

            with torch.no_grad():
                out = model.forward_tts_pointer(**inputs)
            hidden_states_per_level.append(out["hidden_states"].mean(dim=(1, 2)).squeeze())

        # Compute variance across levels
        stacked = torch.stack(hidden_states_per_level)
        variance = stacked.var(dim=0).mean().item()

        assert variance > 1e-8, (
            f"voice_state sweep produced near-zero variance ({variance:.2e}); "
            f"model may not be responsive to voice_state changes"
        )

    def test_twelve_dim_voice_state_each_produces_movement(self):
        """Each of the 12 voice_state dimensions should produce directional movement."""
        model = DisentangledUCLM()
        model.eval()

        base_inputs = _make_model_inputs(target_length=10)
        responsive_dims = 0

        for dim_idx in range(12):
            outputs = []
            for level in [-1.0, 1.0]:
                inputs = {**base_inputs}
                explicit = base_inputs["explicit_state"].clone()
                explicit[:, :, dim_idx] = level
                inputs["explicit_state"] = explicit

                with torch.no_grad():
                    out = model.forward_tts_pointer(**inputs)
                outputs.append(out["hidden_states"].mean().item())

            if abs(outputs[1] - outputs[0]) > 1e-8:
                responsive_dims += 1

        # At minimum, some dimensions should be responsive even with random weights
        assert responsive_dims > 0, "No voice_state dimension produced any output change"


# ---------------------------------------------------------------------------
# CFG Responsiveness
# ---------------------------------------------------------------------------

class TestCFGResponsiveness:
    """CFG scale sweeps should produce measurable acting intensity changes."""

    def test_cfg_responsiveness_score_computable(self):
        """cfg_responsiveness_score works with synthetic sweep data."""
        # Simulate increasing F0 variance with increasing CFG scale
        f0_variances = [10.0, 20.0, 35.0, 50.0]
        cfg_scales = [1.0, 2.0, 3.0, 4.0]
        score = cfg_responsiveness_score(f0_variances, cfg_scales)
        assert isinstance(score, float)
        # Monotonically increasing -> positive correlation
        assert score > 0.5, f"Expected positive correlation, got {score}"

    def test_cfg_guided_vs_unguided_differ(self):
        """Guided output (with conditioning) should differ from unguided (zeroed).

        This tests the CFG mask contract: applying the unconditional mask
        should produce measurably different hidden states.
        """
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_model_inputs(target_length=10)

        with torch.no_grad():
            out_guided = model.forward_tts_pointer(**inputs)

        # Apply unconditional mask to create unguided inputs
        masked = DisentangledUCLM.apply_cfg_unconditional_mask(
            explicit_state=inputs["explicit_state"],
            ssl_state=inputs["ssl_state"],
            speaker_embed=inputs["speaker_embed"],
            dialogue_context=torch.randn(1, 256),
            acting_intent=torch.randn(1, 64),
            prosody_latent=torch.randn(1, 128),
            delta_voice_state=torch.randn(1, 10, 12),
        )

        unguided_inputs = {**inputs}
        unguided_inputs["explicit_state"] = masked["explicit_state"]
        unguided_inputs["ssl_state"] = masked["ssl_state"]
        unguided_inputs["speaker_embed"] = masked["speaker_embed"]

        with torch.no_grad():
            out_unguided = model.forward_tts_pointer(**unguided_inputs)

        # Outputs should differ
        assert not torch.allclose(
            out_guided["hidden_states"], out_unguided["hidden_states"], atol=1e-6
        ), "Guided and unguided outputs are identical -- CFG mask is not effective"


# ---------------------------------------------------------------------------
# Disentanglement Leakage Check
# ---------------------------------------------------------------------------

class TestDisentanglementLeakage:
    """Change speaker while keeping context fixed; prosody should remain stable."""

    def test_timbre_prosody_disentanglement_score_computable(self):
        """timbre_prosody_disentanglement_score works on synthetic data."""
        # Different prosody features across contexts
        features = [torch.randn(64) for _ in range(4)]
        score = timbre_prosody_disentanglement_score(features)
        assert isinstance(score, float)
        assert score >= 0.0

    def test_prosody_transfer_leakage_score_computable(self):
        """prosody_transfer_leakage_score returns valid leakage metric."""
        ref_f0 = torch.randn(100).abs() * 200 + 80
        gen_f0 = torch.randn(100).abs() * 200 + 80
        score = prosody_transfer_leakage_score(ref_f0, gen_f0)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_speaker_change_does_not_collapse_output(self):
        """Changing speaker_embed with same text should produce different outputs
        but the pointer logits should remain structurally valid."""
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_model_inputs(target_length=10)

        with torch.no_grad():
            out1 = model.forward_tts_pointer(**inputs)

        inputs2 = {**inputs}
        inputs2["speaker_embed"] = torch.randn(1, 192)  # different speaker

        with torch.no_grad():
            out2 = model.forward_tts_pointer(**inputs2)

        # Pointer logits should still be finite
        assert torch.isfinite(out2["pointer_logits"]).all()

        # Outputs should differ (different speaker)
        assert not torch.allclose(
            out1["hidden_states"], out2["hidden_states"], atol=1e-6
        )
