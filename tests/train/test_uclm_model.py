"""Tests for DisentangledUCLM model and v3 sub-components.

Covers:
- VC forward pass shape validation
- TTS pointer forward pass shape validation
- SpeakerPromptEncoder forward shape and timbre bottleneck
- ProsodyPredictor training (reparameterization) and eval (deterministic mu)
- DialogueContextProjector with all input combinations
- DisentangledUCLM.encode_speaker_prompt() smoke test
- DisentangledUCLM.predict_prosody() smoke test
- forward_tts_pointer output includes v3 pointer fields
"""

from __future__ import annotations

import pytest
import torch

from tmrvc_core.constants import D_VOICE_STATE_EXPLICIT
from tmrvc_core.types import PointerState
from tmrvc_train.models.uclm_model import (
    DialogueContextProjector,
    DisentangledUCLM,
    ProsodyPredictor,
    SpeakerPromptEncoder,
)

# ---------------------------------------------------------------------------
# Shared fixture: small model dims for fast tests
# ---------------------------------------------------------------------------

_D_MODEL = 64
_D_SPEAKER = 32
_D_PROSODY = 16
_D_DIALOGUE = 32
_D_ACTING = 16
_D_EXPLICIT = D_VOICE_STATE_EXPLICIT  # canonical: 12


def _make_model(**overrides):
    """Create a small DisentangledUCLM for testing."""
    defaults = dict(
        d_model=_D_MODEL,
        n_heads=4,
        n_layers=2,
        rvq_vocab_size=128,
        n_codebooks=8,
        control_vocab_size=32,
        d_explicit=_D_EXPLICIT,
        d_ssl=32,
        d_speaker=_D_SPEAKER,
        vq_bins=32,
        vocab_size=64,
        d_dialogue=_D_DIALOGUE,
        d_acting=_D_ACTING,
        d_prosody=_D_PROSODY,
    )
    defaults.update(overrides)
    return DisentangledUCLM(**defaults)


# ---------------------------------------------------------------------------
# Original VC / TTS pointer tests (preserved)
# ---------------------------------------------------------------------------


def test_uclm_vc_forward():
    B, T = 2, 50
    model = DisentangledUCLM(
        d_model=256,
        n_heads=4,
        n_layers=2,
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_explicit=_D_EXPLICIT,
        d_ssl=128,
        d_speaker=192,
        vq_bins=64,
    )

    source_a_t = torch.randint(0, 1024, (B, 8, T))
    explicit_state = torch.randn(B, T, _D_EXPLICIT)
    ssl_state = torch.randn(B, T, 128)
    speaker_embed = torch.randn(B, 192)

    target_b = torch.randint(0, 64, (B, 4, T))
    out = model.forward_vc(source_a_t, target_b, explicit_state, ssl_state, speaker_embed)

    assert "logits_a" in out
    assert "logits_b" in out
    assert "vq_loss" in out

    assert out["logits_a"].shape == (B, 8, T, 1024)
    assert out["logits_b"].shape == (B, 4, T, 64)
    assert out["vq_loss"].dim() == 0


def test_uclm_tts_pointer_forward():
    B, L, T = 2, 20, 50
    model = DisentangledUCLM(
        d_model=256,
        n_heads=4,
        n_layers=2,
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_explicit=_D_EXPLICIT,
        d_ssl=128,
        d_speaker=192,
        vq_bins=64,
        vocab_size=256,
    )

    phoneme_ids = torch.randint(0, 256, (B, L))
    language_ids = torch.zeros((B,), dtype=torch.long)

    explicit_state = torch.randn(B, T, _D_EXPLICIT)
    ssl_state = torch.randn(B, T, 128)
    speaker_embed = torch.randn(B, 192)

    target_a = torch.randint(0, 1024, (B, 8, T))
    target_b = torch.randint(0, 64, (B, 4, T))
    out = model.forward_tts_pointer(
        phoneme_ids=phoneme_ids,
        language_ids=language_ids,
        pointer_state=None,
        speaker_embed=speaker_embed,
        explicit_state=explicit_state,
        ssl_state=ssl_state,
        target_a=target_a,
        target_b=target_b,
        target_length=T,
    )

    assert "logits_a" in out
    assert "logits_b" in out
    assert "pointer_logits" in out

    assert out["logits_a"].shape == (B, 8, T, 1024)
    assert out["logits_b"].shape == (B, 4, T, 64)


# ---------------------------------------------------------------------------
# SpeakerPromptEncoder tests
# ---------------------------------------------------------------------------


class TestSpeakerPromptEncoder:
    def test_forward_shape(self):
        """Output shapes should be [B, d_model] and [B, T_prompt, d_model]."""
        B, T_prompt, n_codebooks = 2, 30, 8
        enc = SpeakerPromptEncoder(d_model=_D_MODEL, d_speaker=_D_SPEAKER)
        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))

        timbre, prompt_feats, vq_loss, indices = enc(tokens)

        assert timbre.shape == (B, _D_MODEL)
        assert prompt_feats.shape == (B, T_prompt, _D_MODEL)

    def test_forward_without_speaker_embed(self):
        """Should work when speaker_embed is None (prompt-only adaptation)."""
        B, T_prompt, n_codebooks = 1, 20, 4
        enc = SpeakerPromptEncoder(d_model=_D_MODEL, d_speaker=_D_SPEAKER)
        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))

        timbre, prompt_feats, vq_loss, indices = enc(tokens, speaker_embed=None)

        assert timbre.shape == (B, _D_MODEL)
        assert prompt_feats.shape == (B, T_prompt, _D_MODEL)

    def test_forward_with_speaker_embed(self):
        """Should fuse external speaker embedding when provided."""
        B, T_prompt, n_codebooks = 2, 15, 8
        enc = SpeakerPromptEncoder(d_model=_D_MODEL, d_speaker=_D_SPEAKER)
        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))
        spk = torch.randn(B, _D_SPEAKER)

        timbre, prompt_feats, vq_loss, indices = enc(tokens, speaker_embed=spk)

        assert timbre.shape == (B, _D_MODEL)
        assert prompt_feats.shape == (B, T_prompt, _D_MODEL)

    def test_timbre_bottleneck_reduces_dim(self):
        """Timbre bottleneck inner dim should be d_speaker (< d_model),
        preventing full information pass-through."""
        enc = SpeakerPromptEncoder(d_model=_D_MODEL, d_speaker=_D_SPEAKER)

        # The bottleneck is: Linear(d_model, d_speaker) -> Tanh -> Linear(d_speaker, d_model)
        # Inner dimension (d_speaker) should be smaller than d_model.
        bottleneck = enc.timbre_bottleneck
        inner_dim = bottleneck[0].out_features  # first Linear out_features = d_speaker
        assert inner_dim == _D_SPEAKER
        assert inner_dim < _D_MODEL, (
            f"Bottleneck inner dim ({inner_dim}) should be smaller than d_model ({_D_MODEL})"
        )


# ---------------------------------------------------------------------------
# ProsodyPredictor tests
# ---------------------------------------------------------------------------


class TestProsodyPredictor:
    def test_forward_shape_training(self):
        """Training mode should return [B, d_prosody] with reparameterization."""
        B, L = 2, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.train()
        phoneme_feats = torch.randn(B, L, _D_MODEL)

        out = pred(phoneme_feats)

        assert out.shape == (B, _D_PROSODY)

    def test_eval_mode_deterministic(self):
        """Eval mode should return deterministic output given the same seed."""
        B, L = 2, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.eval()
        phoneme_feats = torch.randn(B, L, _D_MODEL)

        torch.manual_seed(0)
        out1 = pred(phoneme_feats)
        torch.manual_seed(0)
        out2 = pred(phoneme_feats)

        # With same seed, flow-matching ODE steps produce identical output
        assert torch.allclose(out1, out2), "Eval mode should be deterministic given same seed"

    def test_training_mode_returns_zeros(self):
        """Training mode forward() returns zeros; actual loss via flow_matching_loss()."""
        B, L = 4, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.train()
        phoneme_feats = torch.randn(B, L, _D_MODEL)

        out = pred(phoneme_feats)
        assert torch.allclose(out, torch.zeros_like(out)), \
            "Training forward() should return zeros (use flow_matching_loss for training)"

    def test_flow_matching_loss_produces_gradient(self):
        """flow_matching_loss() should produce non-zero loss with gradients."""
        B, L = 4, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.train()
        phoneme_feats = torch.randn(B, L, _D_MODEL, requires_grad=True)
        target_prosody = torch.randn(B, pred.d_prosody)

        loss = pred.flow_matching_loss(phoneme_feats, target_prosody)
        assert loss.item() > 0, "Flow-matching loss should be non-zero"
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in pred.parameters()), \
            "Gradients should flow through flow_matching_loss"

    def test_with_dialogue_context(self):
        """Should accept optional dialogue_context."""
        B, L = 2, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.eval()
        phoneme_feats = torch.randn(B, L, _D_MODEL)
        ctx = torch.randn(B, _D_DIALOGUE)

        out = pred(phoneme_feats, dialogue_context=ctx)
        assert out.shape == (B, _D_PROSODY)

    def test_with_speaker_embed(self):
        """Should accept optional speaker_embed."""
        B, L = 2, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.eval()
        phoneme_feats = torch.randn(B, L, _D_MODEL)
        spk = torch.randn(B, _D_MODEL)

        out = pred(phoneme_feats, speaker_embed=spk)
        assert out.shape == (B, _D_PROSODY)

    def test_with_all_optional_inputs(self):
        """Should accept both dialogue_context and speaker_embed."""
        B, L = 2, 20
        pred = ProsodyPredictor(d_model=_D_MODEL, d_prosody=_D_PROSODY)
        pred.eval()
        phoneme_feats = torch.randn(B, L, _D_MODEL)
        ctx = torch.randn(B, _D_DIALOGUE)
        spk = torch.randn(B, _D_MODEL)

        out = pred(phoneme_feats, dialogue_context=ctx, speaker_embed=spk)
        assert out.shape == (B, _D_PROSODY)


# ---------------------------------------------------------------------------
# DialogueContextProjector tests
# ---------------------------------------------------------------------------


class TestDialogueContextProjector:
    def _make_projector(self):
        return DialogueContextProjector(
            d_model=_D_MODEL,
            d_dialogue=_D_DIALOGUE,
            d_acting=_D_ACTING,
            d_prosody=_D_PROSODY,
        )

    def test_no_optional_inputs(self):
        """With no optional inputs, content_features should pass through unchanged."""
        B, T = 2, 30
        proj = self._make_projector()
        content = torch.randn(B, T, _D_MODEL)

        out = proj(content)
        assert torch.allclose(out, content)

    def test_dialogue_context_only(self):
        """Adding dialogue_context should change the output."""
        B, T = 2, 30
        proj = self._make_projector()
        content = torch.randn(B, T, _D_MODEL)
        ctx = torch.randn(B, _D_DIALOGUE)

        out = proj(content, dialogue_context=ctx)
        assert out.shape == (B, T, _D_MODEL)
        assert not torch.allclose(out, content)

    def test_acting_intent_only(self):
        """Adding acting_intent should change the output."""
        B, T = 2, 30
        proj = self._make_projector()
        content = torch.randn(B, T, _D_MODEL)
        act = torch.randn(B, _D_ACTING)

        out = proj(content, acting_intent=act)
        assert out.shape == (B, T, _D_MODEL)
        assert not torch.allclose(out, content)

    def test_prosody_latent_only(self):
        """Adding prosody_latent should change the output."""
        B, T = 2, 30
        proj = self._make_projector()
        content = torch.randn(B, T, _D_MODEL)
        prosody = torch.randn(B, T, _D_PROSODY)

        out = proj(content, prosody_latent=prosody)
        assert out.shape == (B, T, _D_MODEL)
        assert not torch.allclose(out, content)

    def test_prosody_latent_length_mismatch(self):
        """Prosody latent with different T should be interpolated."""
        B, T_content, T_prosody = 2, 30, 15
        proj = self._make_projector()
        content = torch.randn(B, T_content, _D_MODEL)
        prosody = torch.randn(B, T_prosody, _D_PROSODY)

        out = proj(content, prosody_latent=prosody)
        assert out.shape == (B, T_content, _D_MODEL)

    def test_all_optional_inputs(self):
        """With all inputs, output shape should remain [B, T, d_model]."""
        B, T = 2, 30
        proj = self._make_projector()
        content = torch.randn(B, T, _D_MODEL)
        ctx = torch.randn(B, _D_DIALOGUE)
        act = torch.randn(B, _D_ACTING)
        prosody = torch.randn(B, T, _D_PROSODY)

        out = proj(content, dialogue_context=ctx, acting_intent=act, prosody_latent=prosody)
        assert out.shape == (B, T, _D_MODEL)


# ---------------------------------------------------------------------------
# DisentangledUCLM.encode_speaker_prompt() smoke test
# ---------------------------------------------------------------------------


class TestEncodeSpkPrompt:
    def test_encode_speaker_prompt_smoke(self):
        """encode_speaker_prompt should return timbre and summary tokens."""
        B, T_prompt, n_codebooks = 2, 20, 8
        model = _make_model()
        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))

        timbre, summary_tokens, vq_loss, indices = model.encode_speaker_prompt(tokens)
        assert timbre.shape == (B, _D_MODEL)
        # encode_speaker_prompt runs PromptResampler, output is [B, n_summary, D]
        assert summary_tokens.ndim == 3
        assert summary_tokens.shape[0] == B
        assert summary_tokens.shape[2] == _D_MODEL

    def test_encode_speaker_prompt_with_external_embed(self):
        B, T_prompt, n_codebooks = 1, 10, 4
        model = _make_model()
        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))
        spk = torch.randn(B, _D_SPEAKER)

        timbre, prompt_feats, vq_loss, indices = model.encode_speaker_prompt(tokens, speaker_embed=spk)
        assert timbre.shape == (B, _D_MODEL)


# ---------------------------------------------------------------------------
# DisentangledUCLM.predict_prosody() smoke test
# ---------------------------------------------------------------------------


class TestPredictProsody:
    def test_predict_prosody_smoke(self):
        """predict_prosody should return [B, d_prosody]."""
        B, L = 2, 15
        model = _make_model()
        model.eval()
        phoneme_ids = torch.randint(1, 64, (B, L))
        language_ids = torch.zeros((B,), dtype=torch.long)

        out = model.predict_prosody(phoneme_ids, language_ids)
        assert out.shape == (B, _D_PROSODY)

    def test_predict_prosody_with_context(self):
        B, L = 2, 15
        model = _make_model()
        model.eval()
        phoneme_ids = torch.randint(1, 64, (B, L))
        language_ids = torch.zeros((B,), dtype=torch.long)
        ctx = torch.randn(B, _D_DIALOGUE)
        spk = torch.randn(B, _D_MODEL)

        out = model.predict_prosody(phoneme_ids, language_ids, dialogue_context=ctx, speaker_embed=spk)
        assert out.shape == (B, _D_PROSODY)


# ---------------------------------------------------------------------------
# forward_tts_pointer v3 output fields
# ---------------------------------------------------------------------------


class TestForwardTtsPointerV3Fields:
    def test_output_includes_advance_logit(self):
        """forward_tts_pointer output should include advance_logit alias."""
        B, L, T = 2, 10, 30
        model = _make_model()

        out = model.forward_tts_pointer(
            phoneme_ids=torch.randint(1, 64, (B, L)),
            language_ids=torch.zeros((B,), dtype=torch.long),
            pointer_state=None,
            speaker_embed=torch.randn(B, _D_SPEAKER),
            explicit_state=torch.randn(B, T, _D_EXPLICIT),
            ssl_state=torch.randn(B, T, 32),
            target_a=torch.randint(0, 128, (B, 8, T)),
            target_b=torch.randint(0, 32, (B, 4, T)),
            target_length=T,
        )

        assert "advance_logit" in out
        assert out["advance_logit"].shape[0] == B
        assert out["advance_logit"].shape[1] == T

    def test_output_includes_boundary_confidence(self):
        """forward_tts_pointer output should include boundary_confidence."""
        B, L, T = 2, 10, 30
        model = _make_model()

        out = model.forward_tts_pointer(
            phoneme_ids=torch.randint(1, 64, (B, L)),
            language_ids=torch.zeros((B,), dtype=torch.long),
            pointer_state=None,
            speaker_embed=torch.randn(B, _D_SPEAKER),
            explicit_state=torch.randn(B, T, _D_EXPLICIT),
            ssl_state=torch.randn(B, T, 32),
            target_a=torch.randint(0, 128, (B, 8, T)),
            target_b=torch.randint(0, 32, (B, 4, T)),
            target_length=T,
        )

        assert "boundary_confidence" in out
        assert out["boundary_confidence"].shape == (B, T, 1)

    def test_output_includes_next_pointer_state(self):
        """forward_tts_pointer output should include next_pointer_state key."""
        B, L, T = 2, 10, 30
        model = _make_model()

        out = model.forward_tts_pointer(
            phoneme_ids=torch.randint(1, 64, (B, L)),
            language_ids=torch.zeros((B,), dtype=torch.long),
            pointer_state=None,
            speaker_embed=torch.randn(B, _D_SPEAKER),
            explicit_state=torch.randn(B, T, _D_EXPLICIT),
            ssl_state=torch.randn(B, T, 32),
            target_a=torch.randint(0, 128, (B, 8, T)),
            target_b=torch.randint(0, 32, (B, 4, T)),
            target_length=T,
        )

        assert "next_pointer_state" in out


# ---------------------------------------------------------------------------
# PointerState new fields and step_pointer method tests
# ---------------------------------------------------------------------------


class TestPointerStateNewFields:
    def test_pointer_state_max_frames_per_unit_default(self):
        """max_frames_per_unit should default to 50."""
        state = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        assert state.max_frames_per_unit == 50

    def test_pointer_state_frames_on_current_unit_default(self):
        """frames_on_current_unit should default to 0."""
        state = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        assert state.frames_on_current_unit == 0

    def test_pointer_state_skip_protection_threshold_default(self):
        """skip_protection_threshold should default to 0.3."""
        state = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.0]),
        )
        assert state.skip_protection_threshold == pytest.approx(0.3)

    def test_pointer_state_clone_includes_new_fields(self):
        """clone() should copy all new failure-handling fields."""
        state = PointerState(
            text_index=torch.tensor([5]),
            progress=torch.tensor([0.7]),
            finished=False,
            boundary_confidence=0.8,
            stall_frames=3,
            max_frames_per_unit=40,
            frames_on_current_unit=12,
            skip_protection_threshold=0.25,
            forced_advance_count=2,
            skip_protection_count=1,
        )
        cloned = state.clone()

        assert cloned.max_frames_per_unit == 40
        assert cloned.frames_on_current_unit == 12
        assert cloned.skip_protection_threshold == pytest.approx(0.25)
        assert cloned.forced_advance_count == 2
        assert cloned.skip_protection_count == 1
        assert cloned.stall_frames == 3
        assert cloned.boundary_confidence == pytest.approx(0.8)
        assert cloned.finished is False
        # Ensure tensors are independent copies
        assert torch.equal(cloned.text_index, state.text_index)
        assert cloned.text_index is not state.text_index


# ---------------------------------------------------------------------------
# PromptResampler tests (Worker 01 § Architecture)
# ---------------------------------------------------------------------------


class TestPromptResampler:
    def test_resampler_compresses_to_n_summary(self):
        """PromptResampler: [B, T, D] -> [B, 32, D]."""
        from tmrvc_train.models.uclm_transformer import PromptResampler

        B, T_prompt, D = 2, 100, _D_MODEL
        resampler = PromptResampler(d_model=D, n_summary=32, n_heads=4)
        x = torch.randn(B, T_prompt, D)
        out = resampler(x)
        assert out.shape == (B, 32, D)

    def test_resampler_variable_length_input(self):
        """PromptResampler should handle different T_prompt values."""
        from tmrvc_train.models.uclm_transformer import PromptResampler

        D = _D_MODEL
        resampler = PromptResampler(d_model=D, n_summary=32, n_heads=4)

        for T_prompt in [10, 50, 200]:
            x = torch.randn(1, T_prompt, D)
            out = resampler(x)
            assert out.shape == (1, 32, D), f"Failed for T_prompt={T_prompt}"


# ---------------------------------------------------------------------------
# ReferenceEncoder output shape test (Worker 01 § Architecture)
# ---------------------------------------------------------------------------


class TestReferenceEncoderOutput:
    def test_reference_encoder_output_shape(self):
        """ReferenceEncoder should output [B, d_prosody] with d_prosody=128."""
        from tmrvc_train.models.reference_encoder import ReferenceEncoder

        B, n_mels, T_mel = 2, 80, 200
        d_prosody = 128
        enc = ReferenceEncoder(d_prosody=d_prosody, n_mels=n_mels)
        mel = torch.randn(B, n_mels, T_mel)
        out = enc(mel)
        assert out.shape == (B, d_prosody), f"Expected [B, {d_prosody}], got {out.shape}"


# ---------------------------------------------------------------------------
# SpeakerPromptEncoder -> PromptResampler pipeline test
# ---------------------------------------------------------------------------


class TestSpeakerPromptEncoderResamplerPipeline:
    def test_pipeline_output_shapes(self):
        """SpeakerPromptEncoder -> PromptResampler should produce [B, 32, D]."""
        from tmrvc_train.models.uclm_transformer import PromptResampler

        B, T_prompt, n_codebooks = 2, 60, 8
        enc = SpeakerPromptEncoder(d_model=_D_MODEL, d_speaker=_D_SPEAKER)
        resampler = PromptResampler(d_model=_D_MODEL, n_summary=32, n_heads=4)

        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))
        timbre, prompt_feats, vq_loss, indices = enc(tokens)

        assert prompt_feats.shape == (B, T_prompt, _D_MODEL)

        summary = resampler(prompt_feats)
        assert summary.shape == (B, 32, _D_MODEL)

    def test_encode_speaker_prompt_produces_summary(self):
        """DisentangledUCLM.encode_speaker_prompt should produce summary tokens."""
        B, T_prompt, n_codebooks = 2, 40, 8
        model = _make_model()
        tokens = torch.randint(0, 1024, (B, T_prompt, n_codebooks))

        timbre, summary, vq_loss, indices = model.encode_speaker_prompt(tokens)
        # Summary should be [B, n_prompt_summary_tokens, d_model]
        assert summary.ndim == 3
        assert summary.shape[0] == B
        assert summary.shape[2] == _D_MODEL


# ---------------------------------------------------------------------------
# ModernTransformerBlock tests (RoPE, GQA, SwiGLU)
# ---------------------------------------------------------------------------


class TestModernTransformerBlock:
    def test_modern_block_forward_shape(self):
        """ModernTransformerBlock output shape should match input."""
        from tmrvc_train.models.uclm_transformer import ModernTransformerBlock

        B, T, D = 2, 30, _D_MODEL
        block = ModernTransformerBlock(
            d_model=D, n_heads=4, d_ff=D * 4, n_kv_heads=2,
        )
        x = torch.randn(B, T, D)
        memory = torch.randn(B, 20, D)

        out, nk, nv = block(x, memory=memory)
        assert out.shape == (B, T, D)

    def test_modern_block_without_memory(self):
        """ModernTransformerBlock should work without cross-attention memory."""
        from tmrvc_train.models.uclm_transformer import ModernTransformerBlock

        B, T, D = 2, 30, _D_MODEL
        block = ModernTransformerBlock(
            d_model=D, n_heads=4, d_ff=D * 4, n_kv_heads=2,
        )
        x = torch.randn(B, T, D)

        out, nk, nv = block(x, memory=None)
        assert out.shape == (B, T, D)

    def test_modern_block_gqa_kv_heads(self):
        """ModernTransformerBlock should use n_kv_heads=2 for GQA."""
        from tmrvc_train.models.uclm_transformer import ModernTransformerBlock

        block = ModernTransformerBlock(
            d_model=_D_MODEL, n_heads=4, d_ff=_D_MODEL * 4, n_kv_heads=2,
        )
        assert block.attn.n_kv_heads == 2
        assert block.attn.n_rep == 2  # 4 heads / 2 kv_heads = 2

    def test_codec_transformer_use_modern_backbone(self):
        """CodecTransformer with use_modern_backbone=True should use ModernTransformerBlock."""
        from tmrvc_train.models.uclm_transformer import (
            CodecTransformer,
            ModernTransformerBlock,
        )

        ct = CodecTransformer(
            d_model=_D_MODEL, n_heads=4, n_layers=2,
            rvq_vocab_size=128, n_codebooks=8,
            control_vocab_size=32, d_speaker=_D_SPEAKER,
            use_modern_backbone=True,
        )
        assert isinstance(ct.layers[0], ModernTransformerBlock)


# ---------------------------------------------------------------------------
# d_prosody=128 default verification tests
# ---------------------------------------------------------------------------


class TestDProsodyDefaults:
    def test_prosody_predictor_default_d_prosody_128(self):
        """ProsodyPredictor default d_prosody should be 128 (matching constants.yaml)."""
        pred = ProsodyPredictor(d_model=_D_MODEL)
        assert pred.d_prosody == 128

    def test_dialogue_context_projector_default_d_prosody_128(self):
        """DialogueContextProjector default d_prosody should be 128."""
        proj = DialogueContextProjector(d_model=_D_MODEL)
        assert proj.prosody_proj.in_features == 128


# ---------------------------------------------------------------------------
# PointerState property aliases test
# ---------------------------------------------------------------------------


class TestPointerStateProperties:
    def test_progress_value_alias(self):
        """PointerState.progress_value should be an alias for progress."""
        state = PointerState(
            text_index=torch.tensor([3]),
            progress=torch.tensor([0.42]),
        )
        assert torch.equal(state.progress_value, state.progress)

    def test_progress_delta_alias(self):
        """PointerState.progress_delta should return current progress as float."""
        state = PointerState(
            text_index=torch.tensor([0]),
            progress=torch.tensor([0.75]),
        )
        assert abs(state.progress_delta - 0.75) < 1e-5


if __name__ == "__main__":
    test_uclm_vc_forward()
    test_uclm_tts_pointer_forward()
    print("DisentangledUCLM integration tests passed!")
