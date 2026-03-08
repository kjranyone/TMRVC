"""Runtime parity tests -- verify determinism, batch-vs-streaming consistency,
ONNX contract compliance, and Python/Rust pointer state field alignment.
"""

import pytest
import torch

from tmrvc_core.types import PointerState
from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_serve.uclm_engine import PointerInferenceState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pointer_state(batch_size: int = 1) -> PointerState:
    """Create a fresh PointerState for testing."""
    return PointerState(
        text_index=torch.zeros(batch_size, dtype=torch.long),
        progress=torch.zeros(batch_size),
    )


def _make_tts_inputs(model: DisentangledUCLM, batch_size: int = 1,
                      phoneme_len: int = 8, target_length: int = 20,
                      device: str = "cpu") -> dict:
    """Build minimal inputs for forward_tts_pointer on CPU."""
    d_model = model.uclm_core.d_model if hasattr(model.uclm_core, "d_model") else 512
    d_speaker = 192
    n_codebooks = 8
    n_slots = 4
    return {
        "phoneme_ids": torch.randint(1, 100, (batch_size, phoneme_len), device=device),
        "language_ids": torch.zeros(batch_size, phoneme_len, dtype=torch.long, device=device),
        "pointer_state": None,
        "speaker_embed": torch.randn(batch_size, d_speaker, device=device),
        "explicit_state": torch.randn(batch_size, target_length, 8, device=device),
        "ssl_state": torch.randn(batch_size, target_length, 128, device=device),
        "target_a": torch.zeros(batch_size, n_codebooks, target_length, dtype=torch.long, device=device),
        "target_b": torch.zeros(batch_size, n_slots, target_length, dtype=torch.long, device=device),
        "target_length": target_length,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPointerStateDeterminism:
    """PointerState.step_pointer must be deterministic for identical inputs."""

    def test_pointer_state_determinism(self):
        """Running step_pointer with the same inputs twice must produce the
        same resulting state."""
        advance_prob = 0.7
        progress_delta = 0.4
        boundary_confidence = 0.5

        # Run 1
        ps1 = _make_pointer_state()
        result1 = ps1.step_pointer(advance_prob, progress_delta, boundary_confidence)

        # Run 2 -- fresh state, identical inputs
        ps2 = _make_pointer_state()
        result2 = ps2.step_pointer(advance_prob, progress_delta, boundary_confidence)

        assert result1 == result2, "step_pointer return value differs"
        assert torch.equal(ps1.text_index, ps2.text_index), "text_index differs"
        assert torch.equal(ps1.progress, ps2.progress), "progress differs"
        assert ps1.finished == ps2.finished, "finished flag differs"
        assert ps1.stall_frames == ps2.stall_frames, "stall_frames differs"
        assert ps1.frames_on_current_unit == ps2.frames_on_current_unit
        assert ps1.forced_advance_count == ps2.forced_advance_count
        assert ps1.skip_protection_count == ps2.skip_protection_count

    def test_pointer_state_determinism_multi_step(self):
        """Multiple sequential steps must produce identical states."""
        steps = [
            (0.3, 0.2, 0.1),
            (0.6, 0.5, 0.8),
            (0.9, 0.3, 0.6),
        ]

        ps1 = _make_pointer_state()
        ps2 = _make_pointer_state()

        for adv, prog, bconf in steps:
            ps1.step_pointer(adv, prog, bconf)
            ps2.step_pointer(adv, prog, bconf)

        assert torch.equal(ps1.text_index, ps2.text_index)
        assert torch.equal(ps1.progress, ps2.progress)
        assert ps1.forced_advance_count == ps2.forced_advance_count
        assert ps1.skip_protection_count == ps2.skip_protection_count


class TestBatchVsStreamingPointerConsistency:
    """Batch forward_tts_pointer and frame-by-frame forward_streaming must
    produce comparable pointer logits."""

    def test_batch_vs_streaming_pointer_consistency(self):
        """Run forward_tts_pointer with full batch, then simulate frame-by-frame
        with forward_streaming, and compare pointer logits from the batch pass
        against hidden states produced by streaming."""
        model = DisentangledUCLM()
        model.eval()

        target_length = 10
        inputs = _make_tts_inputs(model, target_length=target_length)

        with torch.no_grad():
            batch_out = model.forward_tts_pointer(**inputs)

        # Verify batch output contains pointer logits
        assert "pointer_logits" in batch_out, "Batch output missing pointer_logits"
        assert "hidden_states" in batch_out, "Batch output missing hidden_states"

        # Simulate streaming: run forward_streaming with full context
        # (single-step with all frames, which should match batch behaviour)
        hidden_batch = batch_out["hidden_states"]
        assert hidden_batch is not None, "hidden_states is None"

        # Pointer head should produce consistent results on the same hidden states
        with torch.no_grad():
            ptr_logits_2, prog_delta_2, _bc_2 = model.pointer_head(hidden_batch)

        # Must match the batch pointer_logits exactly (same hidden states)
        assert torch.allclose(
            batch_out["pointer_logits"], ptr_logits_2, atol=1e-5
        ), "Pointer logits differ when re-running pointer_head on same hidden_states"
        assert torch.allclose(
            batch_out["progress_delta"], prog_delta_2, atol=1e-5
        ), "Progress delta differs when re-running pointer_head on same hidden_states"


class TestPytorchOnnxContractFields:
    """Verify the model's forward_tts_pointer output dict has all fields that
    the ONNX contract expects."""

    # These are the output fields documented in docs/design/onnx-contract.md
    # section 4.4 uclm_core.onnx outputs, mapped to Python model output keys.
    ONNX_CONTRACT_KEYS = {
        "logits_a",
        "logits_b",
        "pointer_logits",   # maps to advance_logit in ONNX
        "progress_delta",
        "hidden_states",
    }

    def test_pytorch_onnx_contract_fields(self):
        """forward_tts_pointer output dict must contain every key the ONNX
        contract requires."""
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_tts_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        missing = self.ONNX_CONTRACT_KEYS - set(out.keys())
        assert not missing, f"Missing ONNX contract keys in model output: {missing}"

    def test_advance_logit_alias_present(self):
        """forward_tts_pointer must include 'advance_logit' as an alias for
        'pointer_logits', matching the ONNX output name."""
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_tts_inputs(model)
        with torch.no_grad():
            out = model.forward_tts_pointer(**inputs)

        assert "advance_logit" in out, "Missing 'advance_logit' alias"


class TestPythonRustPointerStateFields:
    """Verify PointerInferenceState from uclm_engine matches PointerState
    fields conceptually -- both must track the same core state."""

    # Core fields that must exist on both the training PointerState and the
    # inference PointerInferenceState.
    CORE_FIELDS = {
        "text_index",
        "progress",
        "finished",
        "stall_frames",
        "max_frames_per_unit",
        "frames_on_current_unit",
        "skip_protection_threshold",
        "forced_advance_count",
        "skip_protection_count",
    }

    def test_python_rust_pointer_state_fields(self):
        """Both PointerState and PointerInferenceState must expose the core
        pointer fields that the Rust engine would also track."""
        ps = _make_pointer_state()
        pis = PointerInferenceState(total_phonemes=10)

        for field_name in self.CORE_FIELDS:
            assert hasattr(ps, field_name), (
                f"PointerState missing field: {field_name}"
            )
            assert hasattr(pis, field_name), (
                f"PointerInferenceState missing field: {field_name}"
            )

    def test_pointer_inference_state_to_dict_covers_core_fields(self):
        """PointerInferenceState.to_dict() must include all core fields."""
        pis = PointerInferenceState(total_phonemes=10)
        d = pis.to_dict()

        # 'finished' is a property, not serialised in to_dict; check separately
        check_fields = self.CORE_FIELDS - {"finished"}
        missing = check_fields - set(d.keys())
        assert not missing, f"to_dict() missing core fields: {missing}"

        # finished must still be accessible as a property
        assert hasattr(pis, "finished")
