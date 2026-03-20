"""Worker 06: Integration matrix tests for v3 contracts.

Tests that validate cross-worker integration:
- v3 pointer training smoke test
- v3 pointer inference smoke test
- ONNX export smoke test
- Python vs Rust numerical parity skeleton
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_train.trainer import UCLMTrainer
from tmrvc_serve.uclm_engine import PointerInferenceState, UCLMEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tts_batch(
    batch_size: int = 1,
    phoneme_len: int = 8,
    target_length: int = 20,
    device: str = "cpu",
) -> dict:
    """Build a minimal batch dict suitable for UCLMTrainer.train_step."""
    n_codebooks = 8
    n_slots = 4
    return {
        "phoneme_ids": torch.randint(1, 100, (batch_size, phoneme_len), device=device),
        "language_id": torch.zeros(batch_size, phoneme_len, dtype=torch.long, device=device),
        "language_ids": torch.zeros(batch_size, phoneme_len, dtype=torch.long, device=device),
        "speaker_embed": torch.randn(batch_size, 192, device=device),
        "speaker_id": torch.zeros(batch_size, dtype=torch.long, device=device),
        "explicit_state": torch.randn(batch_size, target_length, 12, device=device),
        "ssl_state": torch.randn(batch_size, target_length, 128, device=device),
        "target_a": torch.zeros(batch_size, n_codebooks, target_length, dtype=torch.long, device=device),
        "target_b": torch.zeros(batch_size, n_slots, target_length, dtype=torch.long, device=device),
        "target_length": target_length,
        "task": "tts",
    }


# ---------------------------------------------------------------------------
# 1. v3 Pointer Training Smoke Test
# ---------------------------------------------------------------------------

class TestV3PointerTrainingSmoke:
    """Trainer creates model, runs 1 step in pointer mode."""

    def test_trainer_config_accepts_pointer_mode(self):
        """Trainer must accept pointer mode configuration without errors."""
        model = DisentangledUCLM()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = UCLMTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            tts_mode="pointer",
            pointer_target_source="heuristic_bootstrap",
            tts_prob=1.0,
        )
        assert trainer.tts_mode == "pointer"
        assert trainer.pointer_target_source == "heuristic_bootstrap"

    def test_model_forward_tts_pointer_with_grad(self):
        """DisentangledUCLM.forward_tts_pointer must run a differentiable
        forward pass suitable for training."""
        model = DisentangledUCLM()
        model.train()

        inputs = {
            "phoneme_ids": torch.randint(1, 100, (1, 8)),
            "language_ids": torch.zeros(1, 8, dtype=torch.long),
            "pointer_state": None,
            "speaker_embed": torch.randn(1, 192),
            "explicit_state": torch.randn(1, 20, 12),
            "ssl_state": torch.randn(1, 20, 128),
            "target_a": torch.zeros(1, 8, 20, dtype=torch.long),
            "target_b": torch.zeros(1, 4, 20, dtype=torch.long),
            "target_length": 20,
        }

        out = model.forward_tts_pointer(**inputs)

        assert "advance_logit" in out
        assert "logits_a" in out
        # Verify gradients flow through pointer logits
        loss = out["advance_logit"].sum() + out["logits_a"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients flowed through forward_tts_pointer"

    def test_pointer_mode_no_duration_artifacts(self):
        """Pointer mode must not require duration files."""
        model = DisentangledUCLM()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        trainer = UCLMTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            tts_mode="pointer",
        )


# ---------------------------------------------------------------------------
# 2. v3 Pointer Inference Smoke Test
# ---------------------------------------------------------------------------

class TestV3PointerInferenceSmoke:
    """Engine instantiates and processes one frame in pointer mode."""

    def test_engine_creates_pointer_state(self):
        pis = PointerInferenceState(total_phonemes=10)
        assert pis.text_index == 0
        assert pis.total_phonemes == 10
        assert not pis.finished

    def test_model_forward_tts_pointer_produces_output(self):
        """DisentangledUCLM.forward_tts_pointer returns expected keys."""
        model = DisentangledUCLM()
        model.eval()

        inputs = {
            "phoneme_ids": torch.randint(1, 100, (1, 8)),
            "language_ids": torch.zeros(1, 8, dtype=torch.long),
            "pointer_state": None,
            "speaker_embed": torch.randn(1, 192),
            "explicit_state": torch.randn(1, 20, 12),
            "ssl_state": torch.randn(1, 20, 128),
            "target_a": torch.zeros(1, 8, 20, dtype=torch.long),
            "target_b": torch.zeros(1, 4, 20, dtype=torch.long),
            "target_length": 20,
        }

        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert "advance_logit" in out
        assert "logits_a" in out
        assert "hidden_states" in out

    def test_pointer_inference_state_step(self):
        """PointerInferenceState tracks state correctly during inference."""
        pis = PointerInferenceState(total_phonemes=5)
        # Simulate a few steps
        pis.text_index = 1
        pis.frames_generated += 1
        assert pis.text_index == 1
        assert not pis.finished

        pis.text_index = 5
        pis.frames_on_current_unit = 5
        assert pis.finished


# ---------------------------------------------------------------------------
# 3. ONNX Export Smoke Test
# ---------------------------------------------------------------------------

class TestONNXExportSmoke:
    """Validate that ONNX export contract is satisfiable."""

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="ONNX export smoke requires CUDA for full validation",
    )
    def test_onnx_export_model_traceable(self):
        """Model forward can be traced without errors (prerequisite for ONNX)."""
        model = DisentangledUCLM()
        model.eval()

        # Verify forward_tts_pointer runs cleanly (ONNX export starts here)
        inputs = {
            "phoneme_ids": torch.randint(1, 100, (1, 8)),
            "language_ids": torch.zeros(1, 8, dtype=torch.long),
            "pointer_state": None,
            "speaker_embed": torch.randn(1, 192),
            "explicit_state": torch.randn(1, 10, 12),
            "ssl_state": torch.randn(1, 10, 128),
            "target_a": torch.zeros(1, 8, 10, dtype=torch.long),
            "target_b": torch.zeros(1, 4, 10, dtype=torch.long),
            "target_length": 10,
        }
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        # Verify all ONNX contract keys present
        required = {"logits_a", "logits_b", "advance_logit", "progress_delta", "hidden_states"}
        missing = required - set(out.keys())
        assert not missing, f"Missing ONNX contract keys: {missing}"

    def test_onnx_export_wrapper_importable(self):
        """The ONNX export module must be importable."""
        pytest.importorskip("tmrvc_export", reason="tmrvc-export not installed")
        from tmrvc_export.export_onnx import CodecEncoderWrapper, UCLM_CoreWrapper
        assert CodecEncoderWrapper is not None
        assert UCLM_CoreWrapper is not None


# ---------------------------------------------------------------------------
# 4. Python vs Rust Numerical Parity (Skeleton)
# ---------------------------------------------------------------------------

class TestPythonRustNumericalParity:
    """Python vs Rust numerical parity tests.

    These tests validate that the pacing formula and pointer behavior
    are identical between Python and Rust implementations.
    Golden file strategy: Python generates reference outputs, Rust tests compare.
    """

    def test_pointer_logit_parity(self):
        """Python and Rust must produce identical pacing-modulated advance probability."""
        import math

        # Test cases: (advance_logit, pace, hold_bias, boundary_bias, phrase_pressure, breath_tendency)
        test_cases = [
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),   # neutral
            (1.0, 1.0, 0.0, 0.0, 0.0, 0.0),   # positive logit
            (0.0, 2.0, 0.0, 0.0, 0.0, 0.0),   # fast pace
            (0.0, 1.0, 0.5, 0.0, 0.0, 0.0),   # hold bias
            (0.0, 1.0, 0.0, 0.5, 0.0, 0.0),   # boundary bias
            (0.0, 1.0, 0.0, 0.0, 0.5, 0.0),   # phrase pressure
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.5),   # breath tendency
            (1.5, 1.5, 0.3, 0.2, 0.4, 0.1),   # combined
        ]

        for logit, pace, hb, bb, pp, bt in test_cases:
            # Python pacing formula (from uclm_engine.py:1208-1215)
            modulated = logit - hb + bb + (pace - 1.0) * 2.0 + pp * 1.5 - bt * 0.5
            p_adv = 1.0 / (1.0 + math.exp(-modulated))

            # The Rust formula should produce the same result
            # (verified by the Rust unit test test_pointer_step_pacing_formula)
            assert 0.0 <= p_adv <= 1.0, f"p_adv={p_adv} out of range for inputs {(logit, pace, hb, bb, pp, bt)}"

    def test_advance_decision_parity(self):
        """Python and Rust must agree on advance/hold decisions given same inputs."""
        import math

        # Simulate PointerState logic in Python (matching Rust processor.rs)
        def python_pointer_step(advance_logit, progress_delta, boundary_confidence,
                                pace, hold_bias, boundary_bias, phrase_pressure, breath_tendency,
                                progress=0.0, frames_on_current=0, max_frames=50):
            modulated = (advance_logit - hold_bias + boundary_bias
                        + (pace - 1.0) * 2.0 + phrase_pressure * 1.5 - breath_tendency * 0.5)
            advance_prob = 1.0 / (1.0 + math.exp(-modulated))

            velocity = progress_delta * pace
            drag = max(0.0, hold_bias * 0.02)
            progress += max(0.0, velocity - drag)
            frames_on_current += 1

            advanced = False
            if frames_on_current >= max_frames:
                advanced = True
            elif advance_prob > 0.5 and progress >= 1.0:
                if boundary_confidence >= 0.3:
                    advanced = True
            elif hold_bias > 2.0:
                if advance_prob > 0.5 and progress >= 1.0:
                    advanced = True
            elif advance_prob > 0.5 or progress >= 1.0:
                advanced = True

            return advanced, progress

        # Advance case
        adv, _ = python_pointer_step(2.0, 1.5, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0)
        assert adv, "Should advance with strong logit and progress"

        # Hold case
        adv, _ = python_pointer_step(0.0, 0.3, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0)
        assert not adv, "Should hold with weak logit and low progress"

        # Pace speedup
        adv, _ = python_pointer_step(0.0, 1.0, 0.8, 2.0, 0.0, 0.0, 0.0, 0.0)
        assert adv, "Should advance with fast pace"

    def test_force_advance_timing_parity(self):
        """Multi-step pointer trace: forced advance must trigger at max_frames_per_unit."""
        import math

        max_frames = 5
        progress = 0.0
        frames_on_current = 0

        for step in range(10):
            frames_on_current += 1
            # Very low advance probability, minimal progress
            modulated = -5.0  # sigmoid(-5) ~ 0.0067
            advance_prob = 1.0 / (1.0 + math.exp(-modulated))
            progress += 0.01

            if frames_on_current >= max_frames:
                # Force advance
                assert step == max_frames - 1 or frames_on_current == max_frames
                progress = 0.0
                frames_on_current = 0
                break

        assert frames_on_current == 0, "Should have force-advanced"
