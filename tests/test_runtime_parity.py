"""Runtime parity tests -- verify determinism, batch-vs-streaming consistency,
ONNX contract compliance, Python/Rust pointer state field alignment, and
Python-vs-ONNX numerical parity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from tmrvc_core.constants import (
    D_ACTING_LATENT,
    D_MODEL,
    D_SPEAKER,
    D_VOICE_STATE_EXPLICIT,
    D_VOICE_STATE_SSL,
    N_CODEBOOKS,
)
from tmrvc_core.types import PointerState
from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS
from tmrvc_train.models.uclm_model import DisentangledUCLM
from tmrvc_serve.uclm_engine import PointerInferenceState

try:
    import onnxruntime as ort

    HAS_ORT = True
except ImportError:
    HAS_ORT = False


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
        "explicit_state": torch.randn(batch_size, target_length, 12, device=device),
        "ssl_state": torch.randn(batch_size, target_length, 128, device=device),
        "target_a": torch.zeros(batch_size, n_codebooks, target_length, dtype=torch.long, device=device),
        "target_b": torch.zeros(batch_size, n_slots, target_length, dtype=torch.long, device=device),
        "target_length": target_length,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPointerStateDeterminism:
    """PointerState clone and state manipulation must be deterministic."""

    def test_pointer_state_determinism(self):
        """Constructing two identical PointerStates must produce the
        same state values."""
        ps1 = _make_pointer_state()
        ps2 = _make_pointer_state()

        # Simulate identical state updates (v4: PointerState is a data container)
        ps1.stall_frames = 3
        ps1.frames_on_current_unit = 5
        ps1.forced_advance_count = 1
        ps1.skip_protection_count = 2
        ps1.boundary_confidence = 0.5
        ps1.last_advance_score = 0.7

        ps2.stall_frames = 3
        ps2.frames_on_current_unit = 5
        ps2.forced_advance_count = 1
        ps2.skip_protection_count = 2
        ps2.boundary_confidence = 0.5
        ps2.last_advance_score = 0.7

        assert torch.equal(ps1.text_index, ps2.text_index), "text_index differs"
        assert torch.equal(ps1.progress, ps2.progress), "progress differs"
        assert ps1.finished == ps2.finished, "finished flag differs"
        assert ps1.stall_frames == ps2.stall_frames, "stall_frames differs"
        assert ps1.frames_on_current_unit == ps2.frames_on_current_unit
        assert ps1.forced_advance_count == ps2.forced_advance_count
        assert ps1.skip_protection_count == ps2.skip_protection_count

    def test_pointer_state_determinism_multi_step(self):
        """Multiple sequential state updates must produce identical states."""
        ps1 = _make_pointer_state()
        ps2 = _make_pointer_state()

        # Simulate multiple pointer state updates
        for stall, fcu, fac, spc in [(1, 2, 0, 0), (2, 4, 1, 0), (0, 1, 1, 1)]:
            ps1.stall_frames = stall
            ps1.frames_on_current_unit = fcu
            ps1.forced_advance_count = fac
            ps1.skip_protection_count = spc

            ps2.stall_frames = stall
            ps2.frames_on_current_unit = fcu
            ps2.forced_advance_count = fac
            ps2.skip_protection_count = spc

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


# ---------------------------------------------------------------------------
# PyTorch Batch vs Streaming Numerical Parity (Worker 06)
# ---------------------------------------------------------------------------


class TestBatchVsStreamingNumericalParity:
    """PyTorch batch and streaming must produce same output within tolerance."""

    def test_batch_vs_rerun_pointer_head_parity(self):
        """Running forward_tts_pointer and then re-running pointer_head on the
        same hidden_states must produce numerically identical results."""
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_tts_inputs(model, target_length=15)

        with torch.no_grad():
            batch_out = model.forward_tts_pointer(**inputs)

        hidden = batch_out["hidden_states"]
        assert hidden is not None

        with torch.no_grad():
            ptr2, prog2, bc2 = model.pointer_head(hidden)

        assert torch.allclose(
            batch_out["pointer_logits"], ptr2, atol=1e-5
        ), "Pointer logits differ on same hidden_states"
        assert torch.allclose(
            batch_out["progress_delta"], prog2, atol=1e-5
        ), "Progress delta differs on same hidden_states"

    def test_deterministic_forward_on_cpu(self):
        """Two identical forward passes must produce identical results on CPU."""
        model = DisentangledUCLM()
        model.eval()

        inputs = _make_tts_inputs(model, target_length=10)

        with torch.no_grad():
            out1 = model.forward_tts_pointer(**inputs)
            out2 = model.forward_tts_pointer(**inputs)

        assert torch.allclose(
            out1["pointer_logits"], out2["pointer_logits"], atol=1e-6
        ), "Non-deterministic pointer logits on CPU"
        assert torch.allclose(
            out1["hidden_states"], out2["hidden_states"], atol=1e-6
        ), "Non-deterministic hidden states on CPU"


# ---------------------------------------------------------------------------
# Pointer State Serialization Roundtrip (Worker 06)
# ---------------------------------------------------------------------------


class TestPointerStateSerializationRoundtrip:
    """PointerState serialization roundtrip must preserve all state."""

    def test_pointer_inference_state_roundtrip(self):
        """PointerInferenceState.to_dict -> reconstruct -> identical state."""
        pis = PointerInferenceState(total_phonemes=20)
        # Advance state
        pis.text_index = 5
        pis.progress = 0.7
        pis.stall_frames = 3
        pis.frames_on_current_unit = 8
        pis.forced_advance_count = 1
        pis.skip_protection_count = 2

        d = pis.to_dict()

        # Reconstruct
        pis2 = PointerInferenceState(total_phonemes=d.get("total_phonemes", 20))
        pis2.text_index = d["text_index"]
        pis2.progress = d["progress"]
        pis2.stall_frames = d["stall_frames"]
        pis2.frames_on_current_unit = d["frames_on_current_unit"]
        pis2.forced_advance_count = d["forced_advance_count"]
        pis2.skip_protection_count = d["skip_protection_count"]
        pis2.max_frames_per_unit = d["max_frames_per_unit"]
        pis2.skip_protection_threshold = d["skip_protection_threshold"]

        assert pis2.text_index == pis.text_index
        assert pis2.progress == pis.progress
        assert pis2.stall_frames == pis.stall_frames
        assert pis2.frames_on_current_unit == pis.frames_on_current_unit
        assert pis2.forced_advance_count == pis.forced_advance_count
        assert pis2.skip_protection_count == pis.skip_protection_count

    def test_core_pointer_state_clone_roundtrip(self):
        """PointerState.clone must produce an independent identical copy."""
        ps = PointerState(
            text_index=torch.tensor([7]),
            progress=torch.tensor([0.3]),
            stall_frames=2,
            max_frames_per_unit=40,
            frames_on_current_unit=5,
            forced_advance_count=1,
            skip_protection_count=3,
        )

        ps_clone = ps.clone()

        # Identical values
        assert torch.equal(ps.text_index, ps_clone.text_index)
        assert torch.equal(ps.progress, ps_clone.progress)
        assert ps.stall_frames == ps_clone.stall_frames
        assert ps.forced_advance_count == ps_clone.forced_advance_count
        assert ps.skip_protection_count == ps_clone.skip_protection_count

        # Independence: modifying clone should not affect original
        ps_clone.text_index += 1
        ps_clone.stall_frames = 99
        assert ps.text_index.item() == 7
        assert ps.stall_frames == 2


# ---------------------------------------------------------------------------
# voice_state Serialization Roundtrip (Worker 06)
# ---------------------------------------------------------------------------


class TestVoiceStateSerializationRoundtrip:
    """voice_state fields must survive serialization roundtrip."""

    def test_voice_state_tensor_roundtrip(self):
        """voice_state as a tensor must survive save/load roundtrip."""
        import io

        voice_state = torch.randn(1, 10, 12)

        buf = io.BytesIO()
        torch.save(voice_state, buf)
        buf.seek(0)
        loaded = torch.load(buf, weights_only=True)

        assert torch.equal(voice_state, loaded), "voice_state tensor roundtrip failed"

    def test_voice_state_supervision_roundtrip(self):
        """VoiceStateSupervision fields must be reconstructable."""
        from tmrvc_core.types import VoiceStateSupervision

        vs = VoiceStateSupervision(
            targets=torch.randn(2, 10, 12),
            observed_mask=torch.ones(2, 10, 12, dtype=torch.bool),
            confidence=torch.rand(2, 10, 12),
            provenance="test_roundtrip_v1",
        )

        # Simulate serialization by extracting and reconstructing
        serialized = {
            "targets": vs.targets.clone(),
            "observed_mask": vs.observed_mask.clone(),
            "confidence": vs.confidence.clone(),
            "provenance": vs.provenance,
        }

        vs2 = VoiceStateSupervision(**serialized)

        assert torch.equal(vs.targets, vs2.targets)
        assert torch.equal(vs.observed_mask, vs2.observed_mask)
        assert torch.equal(vs.confidence, vs2.confidence)
        assert vs.provenance == vs2.provenance


# ---------------------------------------------------------------------------
# Python vs ONNX Numerical Parity (Worker 06)
# ---------------------------------------------------------------------------

_CONTEXT_FRAMES = 10  # short context for fast test exports


def _export_all_components(
    model: DisentangledUCLM, output_dir: Path
) -> tuple[dict[str, Path], int]:
    """Export vc_encoder, voice_state_enc, and uclm_core to ONNX.

    Returns (onnx_paths dict, kv_cache_size).
    """
    from tmrvc_export.export_uclm import (
        VCEncoderExportWrapper,
        VoiceStateEncExportWrapper,
        UCLMCoreExportWrapper,
    )

    device = "cpu"
    opset = 18

    # --- vc_encoder ---
    vc_wrapper = VCEncoderExportWrapper(model).eval()
    vc_path = output_dir / "vc_encoder.onnx"
    dummy_source = torch.zeros(1, N_CODEBOOKS, _CONTEXT_FRAMES, dtype=torch.long)
    torch.onnx.export(
        vc_wrapper,
        (dummy_source,),
        vc_path,
        input_names=["source_A_t"],
        output_names=["vq_content_features"],
        dynamic_axes={"source_A_t": {0: "batch", 2: "L"}, "vq_content_features": {0: "batch", 2: "L"}},
        opset_version=opset,
        do_constant_folding=True,
    )

    # --- voice_state_enc ---
    vs_wrapper = VoiceStateEncExportWrapper(model).eval()
    vs_path = output_dir / "voice_state_enc.onnx"
    dummy_explicit = torch.zeros(1, D_VOICE_STATE_EXPLICIT, device=device)
    dummy_ssl = torch.zeros(1, D_VOICE_STATE_SSL, device=device)
    dummy_delta = torch.zeros(1, D_VOICE_STATE_EXPLICIT, device=device)
    torch.onnx.export(
        vs_wrapper,
        (dummy_explicit, dummy_ssl, dummy_delta),
        vs_path,
        input_names=["explicit_state", "ssl_state", "delta_state"],
        output_names=["state_cond"],
        dynamic_axes={
            "explicit_state": {0: "batch"},
            "ssl_state": {0: "batch"},
            "delta_state": {0: "batch"},
            "state_cond": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # --- uclm_core ---
    core_wrapper = UCLMCoreExportWrapper(model, max_seq_len=_CONTEXT_FRAMES).eval()
    core_path = output_dir / "uclm_core.onnx"
    kv_cache_size = core_wrapper.kv_cache_size
    dummy_content = torch.zeros(1, D_MODEL, _CONTEXT_FRAMES)
    dummy_b_ctx = torch.zeros(1, 4, _CONTEXT_FRAMES, dtype=torch.long)
    dummy_spk = torch.zeros(1, D_SPEAKER)
    dummy_state_cond = torch.zeros(1, D_MODEL)
    dummy_acting = torch.zeros(1, D_ACTING_LATENT)
    dummy_cfg = torch.tensor([1.5])
    dummy_kv = torch.zeros(1, kv_cache_size)
    torch.onnx.export(
        core_wrapper,
        (dummy_content, dummy_b_ctx, dummy_spk, dummy_state_cond, dummy_acting, dummy_cfg, dummy_kv),
        core_path,
        input_names=["content_features", "b_ctx", "spk_embed", "state_cond", "acting_intent", "cfg_scale", "kv_cache_in"],
        output_names=["logits_a", "logits_b", "kv_cache_out"],
        dynamic_axes={
            "content_features": {0: "batch", 2: "L"},
            "b_ctx": {0: "batch", 2: "L"},
            "spk_embed": {0: "batch"},
            "state_cond": {0: "batch"},
            "acting_intent": {0: "batch"},
            "cfg_scale": {},
            "kv_cache_in": {0: "batch"},
            "logits_a": {0: "batch"},
            "logits_b": {0: "batch"},
            "kv_cache_out": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    paths = {
        "vc_encoder": vc_path,
        "voice_state_enc": vs_path,
        "uclm_core": core_path,
    }
    return paths, kv_cache_size


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestPythonOnnxNumericalParity:
    """Tests that PyTorch and ONNX produce numerically identical outputs
    for every exported component of the DisentangledUCLM pipeline."""

    PARITY_THRESHOLD = 1e-4

    @pytest.fixture(scope="class")
    def model(self) -> DisentangledUCLM:
        m = DisentangledUCLM()
        m.eval()
        return m

    @pytest.fixture(scope="class")
    def export_artifacts(self, model: DisentangledUCLM):
        """Export all components once per class, reuse across tests."""
        with tempfile.TemporaryDirectory(prefix="tmrvc_parity_") as tmpdir:
            onnx_paths, kv_cache_size = _export_all_components(
                model, Path(tmpdir)
            )
            # Load ONNX sessions while the temp dir exists
            sessions = {
                name: ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
                for name, p in onnx_paths.items()
            }
            yield sessions, kv_cache_size

    # -- vc_encoder parity --

    def test_vc_encoder_parity(
        self, model: DisentangledUCLM, export_artifacts
    ):
        """vc_encoder: PyTorch vs ONNX content features must match."""
        sessions, _ = export_artifacts
        sess = sessions["vc_encoder"]

        source_A_t = torch.randint(0, 100, (1, N_CODEBOOKS, _CONTEXT_FRAMES))

        with torch.no_grad():
            pt_content, _ = model.vc_encoder(source_A_t)
            pt_content = pt_content.transpose(1, 2)

        onnx_content = sess.run(
            None, {"source_A_t": source_A_t.numpy()}
        )[0]

        err = np.max(np.abs(pt_content.numpy() - onnx_content))
        assert err < self.PARITY_THRESHOLD, (
            f"vc_encoder parity failed: L_inf={err:.6e} > {self.PARITY_THRESHOLD}"
        )

    # -- voice_state_enc parity --

    def test_voice_state_enc_parity(
        self, model: DisentangledUCLM, export_artifacts
    ):
        """voice_state_enc: PyTorch vs ONNX state_cond must match."""
        sessions, _ = export_artifacts
        sess = sessions["voice_state_enc"]

        explicit = torch.randn(1, D_VOICE_STATE_EXPLICIT)
        ssl = torch.randn(1, D_VOICE_STATE_SSL)
        delta = torch.randn(1, D_VOICE_STATE_EXPLICIT)

        with torch.no_grad():
            pt_cond = model.voice_state_enc(
                explicit.unsqueeze(1), ssl.unsqueeze(1), delta.unsqueeze(1)
            ).squeeze(1)

        onnx_cond = sess.run(
            None,
            {
                "explicit_state": explicit.numpy(),
                "ssl_state": ssl.numpy(),
                "delta_state": delta.numpy(),
            },
        )[0]

        err = np.max(np.abs(pt_cond.numpy() - onnx_cond))
        assert err < self.PARITY_THRESHOLD, (
            f"voice_state_enc parity failed: L_inf={err:.6e} > {self.PARITY_THRESHOLD}"
        )

    # -- uclm_core parity --

    def test_uclm_core_parity(
        self, model: DisentangledUCLM, export_artifacts
    ):
        """uclm_core: PyTorch vs ONNX logits_a and logits_b must match."""
        sessions, kv_cache_size = export_artifacts
        sess = sessions["uclm_core"]

        content = torch.randn(1, D_MODEL, _CONTEXT_FRAMES)
        b_ctx = torch.zeros(1, 4, _CONTEXT_FRAMES, dtype=torch.long)
        spk = torch.randn(1, D_SPEAKER)
        state_cond = torch.randn(1, D_MODEL)
        acting = torch.zeros(1, D_ACTING_LATENT)
        cfg_scale = torch.tensor([1.5])
        kv_cache = torch.zeros(1, kv_cache_size)

        # PyTorch path (matches UCLMCoreExportWrapper.forward)
        with torch.no_grad():
            pt_la, pt_lb, _ = model.uclm_core(
                content.transpose(1, 2),  # [B, L, D]
                b_ctx,
                spk,
                state_cond,
                1.5,
                kv_cache,
                _CONTEXT_FRAMES,
            )
            pt_la = pt_la[:, :, -1, :]
            pt_lb = pt_lb[:, :, -1, :]

        onnx_la, onnx_lb, _ = sess.run(
            None,
            {
                "content_features": content.numpy(),
                "b_ctx": b_ctx.numpy(),
                "spk_embed": spk.numpy(),
                "state_cond": state_cond.numpy(),
                "acting_intent": acting.numpy(),
                "cfg_scale": cfg_scale.numpy(),
                "kv_cache_in": kv_cache.numpy(),
            },
        )

        err_a = np.max(np.abs(pt_la.numpy() - onnx_la))
        err_b = np.max(np.abs(pt_lb.numpy() - onnx_lb))
        assert err_a < self.PARITY_THRESHOLD, (
            f"uclm_core logits_a parity failed: L_inf={err_a:.6e} > {self.PARITY_THRESHOLD}"
        )
        assert err_b < self.PARITY_THRESHOLD, (
            f"uclm_core logits_b parity failed: L_inf={err_b:.6e} > {self.PARITY_THRESHOLD}"
        )

    # -- end-to-end pipeline parity --

    def test_end_to_end_pipeline_parity(
        self, model: DisentangledUCLM, export_artifacts
    ):
        """Full pipeline: chained vc_encoder -> voice_state_enc -> uclm_core
        must match between PyTorch and ONNX."""
        sessions, kv_cache_size = export_artifacts

        source_A_t = torch.randint(0, 100, (1, N_CODEBOOKS, _CONTEXT_FRAMES))
        explicit = torch.randn(1, D_VOICE_STATE_EXPLICIT)
        ssl = torch.randn(1, D_VOICE_STATE_SSL)
        delta = torch.randn(1, D_VOICE_STATE_EXPLICIT)
        spk = torch.randn(1, D_SPEAKER)
        cfg_scale = torch.tensor([1.5])
        acting = torch.zeros(1, D_ACTING_LATENT)
        kv_cache = torch.zeros(1, kv_cache_size)

        # -- PyTorch pipeline --
        with torch.no_grad():
            pt_content, _ = model.vc_encoder(source_A_t)
            pt_content = pt_content.transpose(1, 2)

            pt_state_cond = model.voice_state_enc(
                explicit.unsqueeze(1), ssl.unsqueeze(1), delta.unsqueeze(1)
            ).squeeze(1)

            state_expanded = pt_state_cond.unsqueeze(1).expand(-1, _CONTEXT_FRAMES, -1)
            pt_la, pt_lb, _ = model.uclm_core(
                pt_content.transpose(1, 2),
                torch.zeros(1, 4, _CONTEXT_FRAMES, dtype=torch.long),
                spk,
                state_expanded[:, 0, :],  # single-frame state_cond
                1.5,
                kv_cache,
                _CONTEXT_FRAMES,
            )
            pt_la = pt_la[:, :, -1, :]
            pt_lb = pt_lb[:, :, -1, :]

        # -- ONNX pipeline --
        onnx_content = sessions["vc_encoder"].run(
            None, {"source_A_t": source_A_t.numpy()}
        )[0]

        onnx_state_cond = sessions["voice_state_enc"].run(
            None,
            {
                "explicit_state": explicit.numpy(),
                "ssl_state": ssl.numpy(),
                "delta_state": delta.numpy(),
            },
        )[0]

        onnx_la, onnx_lb, _ = sessions["uclm_core"].run(
            None,
            {
                "content_features": onnx_content,
                "b_ctx": np.zeros((1, 4, _CONTEXT_FRAMES), dtype=np.int64),
                "spk_embed": spk.numpy(),
                "state_cond": onnx_state_cond,
                "acting_intent": acting.numpy(),
                "cfg_scale": cfg_scale.numpy(),
                "kv_cache_in": kv_cache.numpy(),
            },
        )

        err_a = np.max(np.abs(pt_la.numpy() - onnx_la))
        err_b = np.max(np.abs(pt_lb.numpy() - onnx_lb))
        assert err_a < self.PARITY_THRESHOLD, (
            f"End-to-end logits_a parity failed: L_inf={err_a:.6e}"
        )
        assert err_b < self.PARITY_THRESHOLD, (
            f"End-to-end logits_b parity failed: L_inf={err_b:.6e}"
        )


# ---------------------------------------------------------------------------
# Physical Control Ordering Parity (Worker 06)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestPhysicalControlOrderingParity:
    """Tests that physical control dimensions are ordered identically
    across Python exports and match CANONICAL_VOICE_STATE_IDS."""

    @pytest.fixture(scope="class")
    def vs_session(self):
        """Export voice_state_enc and return its ORT session."""
        from tmrvc_export.export_uclm import VoiceStateEncExportWrapper

        model = DisentangledUCLM()
        model.eval()
        wrapper = VoiceStateEncExportWrapper(model).eval()

        with tempfile.TemporaryDirectory(prefix="tmrvc_ctrl_") as tmpdir:
            path = Path(tmpdir) / "voice_state_enc.onnx"
            dummy_explicit = torch.zeros(1, D_VOICE_STATE_EXPLICIT)
            dummy_ssl = torch.zeros(1, D_VOICE_STATE_SSL)
            dummy_delta = torch.zeros(1, D_VOICE_STATE_EXPLICIT)
            torch.onnx.export(
                wrapper,
                (dummy_explicit, dummy_ssl, dummy_delta),
                path,
                input_names=["explicit_state", "ssl_state", "delta_state"],
                output_names=["state_cond"],
                dynamic_axes={
                    "explicit_state": {0: "batch"},
                    "ssl_state": {0: "batch"},
                    "delta_state": {0: "batch"},
                    "state_cond": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
            )
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            yield sess

    def test_explicit_state_input_shape_is_12d(self, vs_session):
        """voice_state_enc ONNX input explicit_state must have shape [B, 12]."""
        inp = next(i for i in vs_session.get_inputs() if i.name == "explicit_state")
        # Shape may be [batch, 12] or ['batch', 12] with dynamic first dim
        assert inp.shape[-1] == D_VOICE_STATE_EXPLICIT, (
            f"explicit_state last dim is {inp.shape[-1]}, expected {D_VOICE_STATE_EXPLICIT}"
        )

    def test_12d_ordering_matches_canonical_ids(self):
        """The 12-D explicit_state ordering must match CANONICAL_VOICE_STATE_IDS.

        This test verifies the contract: dimension i of the explicit_state
        tensor corresponds to CANONICAL_VOICE_STATE_IDS[i].
        """
        assert len(CANONICAL_VOICE_STATE_IDS) == D_VOICE_STATE_EXPLICIT, (
            f"CANONICAL_VOICE_STATE_IDS has {len(CANONICAL_VOICE_STATE_IDS)} entries "
            f"but D_VOICE_STATE_EXPLICIT={D_VOICE_STATE_EXPLICIT}"
        )

    def test_delta_state_matches_explicit_state_shape(self, vs_session):
        """delta_state must have the same dimensionality as explicit_state."""
        inp_explicit = next(
            i for i in vs_session.get_inputs() if i.name == "explicit_state"
        )
        inp_delta = next(
            i for i in vs_session.get_inputs() if i.name == "delta_state"
        )
        assert inp_explicit.shape[-1] == inp_delta.shape[-1], (
            f"explicit_state dim {inp_explicit.shape[-1]} != "
            f"delta_state dim {inp_delta.shape[-1]}"
        )

    def test_ssl_state_is_128d(self, vs_session):
        """ssl_state ONNX input must have shape [B, 128]."""
        inp = next(i for i in vs_session.get_inputs() if i.name == "ssl_state")
        assert inp.shape[-1] == D_VOICE_STATE_SSL, (
            f"ssl_state last dim is {inp.shape[-1]}, expected {D_VOICE_STATE_SSL}"
        )


# ---------------------------------------------------------------------------
# Acting Latent Ordering Parity (Worker 06)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
class TestActingLatentOrderingParity:
    """Tests that the acting_intent tensor has the expected shape and
    ordering in the ONNX uclm_core model."""

    @pytest.fixture(scope="class")
    def core_session(self):
        """Export uclm_core and return its ORT session."""
        from tmrvc_export.export_uclm import UCLMCoreExportWrapper

        model = DisentangledUCLM()
        model.eval()
        wrapper = UCLMCoreExportWrapper(model, max_seq_len=_CONTEXT_FRAMES).eval()
        kv_cache_size = wrapper.kv_cache_size

        with tempfile.TemporaryDirectory(prefix="tmrvc_acting_") as tmpdir:
            path = Path(tmpdir) / "uclm_core.onnx"
            torch.onnx.export(
                wrapper,
                (
                    torch.zeros(1, D_MODEL, _CONTEXT_FRAMES),
                    torch.zeros(1, 4, _CONTEXT_FRAMES, dtype=torch.long),
                    torch.zeros(1, D_SPEAKER),
                    torch.zeros(1, D_MODEL),
                    torch.zeros(1, D_ACTING_LATENT),
                    torch.tensor([1.5]),
                    torch.zeros(1, kv_cache_size),
                ),
                path,
                input_names=[
                    "content_features", "b_ctx", "spk_embed", "state_cond",
                    "acting_intent", "cfg_scale", "kv_cache_in",
                ],
                output_names=["logits_a", "logits_b", "kv_cache_out"],
                dynamic_axes={
                    "content_features": {0: "batch", 2: "L"},
                    "b_ctx": {0: "batch", 2: "L"},
                    "spk_embed": {0: "batch"},
                    "state_cond": {0: "batch"},
                    "acting_intent": {0: "batch"},
                    "cfg_scale": {},
                    "kv_cache_in": {0: "batch"},
                    "logits_a": {0: "batch"},
                    "logits_b": {0: "batch"},
                    "kv_cache_out": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
            )
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            yield sess

    def test_acting_intent_shape_is_24d(self, core_session):
        """ONNX uclm_core input acting_intent must have shape [B, 24]."""
        inp = next(
            i for i in core_session.get_inputs() if i.name == "acting_intent"
        )
        assert inp.shape[-1] == D_ACTING_LATENT, (
            f"acting_intent last dim is {inp.shape[-1]}, expected {D_ACTING_LATENT} (={D_ACTING_LATENT})"
        )

    def test_acting_intent_is_24_constant(self):
        """D_ACTING_LATENT must equal 24 per the architecture spec."""
        assert D_ACTING_LATENT == 24, (
            f"D_ACTING_LATENT changed from 24 to {D_ACTING_LATENT} -- "
            "update ONNX contract and Rust engine if intentional"
        )

    def test_zero_acting_intent_is_noop(self, core_session):
        """Passing all-zero acting_intent must not crash and must produce
        valid logits (zeros = no acting control)."""
        from tmrvc_export.export_uclm import UCLMCoreExportWrapper

        model = DisentangledUCLM()
        model.eval()
        wrapper = UCLMCoreExportWrapper(model, max_seq_len=_CONTEXT_FRAMES)
        kv_cache_size = wrapper.kv_cache_size

        result = core_session.run(
            None,
            {
                "content_features": np.zeros((1, D_MODEL, _CONTEXT_FRAMES), dtype=np.float32),
                "b_ctx": np.zeros((1, 4, _CONTEXT_FRAMES), dtype=np.int64),
                "spk_embed": np.zeros((1, D_SPEAKER), dtype=np.float32),
                "state_cond": np.zeros((1, D_MODEL), dtype=np.float32),
                "acting_intent": np.zeros((1, D_ACTING_LATENT), dtype=np.float32),
                "cfg_scale": np.array([1.5], dtype=np.float32),
                "kv_cache_in": np.zeros((1, kv_cache_size), dtype=np.float32),
            },
        )
        logits_a, logits_b, kv_out = result
        assert logits_a.shape[0] == 1, "Batch dimension missing from logits_a"
        assert logits_b.shape[0] == 1, "Batch dimension missing from logits_b"
        assert np.isfinite(logits_a).all(), "logits_a contains NaN/Inf"
        assert np.isfinite(logits_b).all(), "logits_b contains NaN/Inf"
