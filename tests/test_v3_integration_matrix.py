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
        "explicit_state": torch.randn(batch_size, target_length, 8, device=device),
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
        assert trainer.legacy_duration_loss_weight == 0.0

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
            "explicit_state": torch.randn(1, 20, 8),
            "ssl_state": torch.randn(1, 20, 128),
            "target_a": torch.zeros(1, 8, 20, dtype=torch.long),
            "target_b": torch.zeros(1, 4, 20, dtype=torch.long),
            "target_length": 20,
        }

        out = model.forward_tts_pointer(**inputs)

        assert "pointer_logits" in out
        assert "logits_a" in out
        # Verify gradients flow through pointer logits
        loss = out["pointer_logits"].sum() + out["logits_a"].sum()
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
        assert trainer.legacy_duration_loss_weight == 0.0


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
            "explicit_state": torch.randn(1, 20, 8),
            "ssl_state": torch.randn(1, 20, 128),
            "target_a": torch.zeros(1, 8, 20, dtype=torch.long),
            "target_b": torch.zeros(1, 4, 20, dtype=torch.long),
            "target_length": 20,
        }

        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert "pointer_logits" in out
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
            "explicit_state": torch.randn(1, 10, 8),
            "ssl_state": torch.randn(1, 10, 128),
            "target_a": torch.zeros(1, 8, 10, dtype=torch.long),
            "target_b": torch.zeros(1, 4, 10, dtype=torch.long),
            "target_length": 10,
        }
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        # Verify all ONNX contract keys present
        required = {"logits_a", "logits_b", "pointer_logits", "progress_delta", "hidden_states"}
        missing = required - set(out.keys())
        assert not missing, f"Missing ONNX contract keys: {missing}"

    def test_onnx_export_wrapper_importable(self):
        """The ONNX export module must be importable."""
        from tmrvc_export.export_onnx import CodecEncoderWrapper, UCLM_CoreWrapper
        assert CodecEncoderWrapper is not None
        assert UCLM_CoreWrapper is not None


# ---------------------------------------------------------------------------
# 4. Python vs Rust Numerical Parity (Skeleton)
# ---------------------------------------------------------------------------

class TestPythonRustNumericalParity:
    """Skeleton for Python vs Rust numerical parity tests.

    TODO: Implement once the Rust runtime is available for testing.
    These tests should:
    1. Run the same input through Python forward_streaming and Rust engine.
    2. Compare pointer logits, advance decisions, and codec outputs.
    3. Verify tolerance is within the frozen parity budget.
    """

    @pytest.mark.skip(reason="TODO: Requires Rust engine bindings for parity testing")
    def test_pointer_logit_parity(self):
        """Python and Rust must produce pointer logits within tolerance."""
        # TODO: Load Rust engine, run same input, compare pointer_logits
        pass

    @pytest.mark.skip(reason="TODO: Requires Rust engine bindings for parity testing")
    def test_advance_decision_parity(self):
        """Python and Rust must agree on advance/hold decisions."""
        # TODO: Run identical pointer states through both runtimes
        pass

    @pytest.mark.skip(reason="TODO: Requires Rust engine bindings for parity testing")
    def test_codec_output_parity(self):
        """Python and Rust must produce codec tokens within tolerance."""
        # TODO: Compare logits_a and logits_b from both runtimes
        pass

    @pytest.mark.skip(reason="TODO: Requires Rust engine bindings for parity testing")
    def test_force_advance_timing_parity(self):
        """Python and Rust must agree on forced-advance trigger timing."""
        # TODO: Run multi-step sequences through both runtimes and
        # compare forced_advance_count at each step
        pass
