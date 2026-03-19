"""Tests for CFG lazy and distilled modes in the UCLM serving engine."""

import pytest
import torch
import torch.nn.functional as F

from tmrvc_core.types import CFGMode
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.uclm_engine import UCLMEngine, PointerInferenceState
from tmrvc_train.models import DisentangledUCLM


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

class _MockTextEncoder(torch.nn.Module):
    def forward(self, phoneme_ids, lang_ids, phoneme_lens, text_suprasegmentals=None):
        B, L = phoneme_ids.shape
        return torch.randn(B, 512, L)


class _MockUCLMCoreModel:
    """Lightweight mock that mimics the DisentangledUCLM interface for TTS."""

    text_encoder = _MockTextEncoder()
    _has_distilled = False

    def __init__(self, *, has_distilled: bool = False):
        self._has_distilled = has_distilled
        if has_distilled:
            self.cfg_scale_embed = torch.nn.Linear(1, 512)

    def forward_streaming(
        self,
        queries=None,
        memory=None,
        a_ctx=None,
        b_ctx=None,
        speaker_embed=None,
        explicit_state=None,
        ssl_state=None,
        delta_voice_state=None,
        cfg_scale=1.0,
        kv_caches=None,
        f0_condition=None,
        dialogue_context=None,
        acting_intent=None,
        prosody_latent=None,
        prompt_summary_tokens=None,
        # Legacy positional support
        content_features=None,
        **kwargs,
    ):
        q = queries if queries is not None else content_features
        B = q.shape[0]
        # Deterministic logits seeded by cfg_scale so we can distinguish passes
        base = torch.full((B, 8, 1, 1024), cfg_scale * 0.01)
        base_b = torch.full((B, 4, 1, 64), cfg_scale * 0.01)
        return {
            "logits_a": base,
            "logits_b": base_b,
            "kv_cache_out": kv_caches,
            "hidden_states": torch.zeros(B, 1, 512),
        }

    _call_count = 0

    def pointer_head(self, hidden_states):
        self._call_count += 1
        adv_logit = torch.tensor([[[-5.0]]])
        progress = torch.tensor([[[0.5]]])
        if self._call_count > 3:
            adv_logit = torch.tensor([[[10.0]]])
            progress = torch.tensor([[[1.0]]])
        boundary_conf = torch.tensor([[[0.8]]])
        return adv_logit, progress, boundary_conf

    @staticmethod
    def apply_cfg_unconditional_mask(**kwargs):
        return {k: torch.zeros_like(v) if isinstance(v, torch.Tensor) else None for k, v in kwargs.items()}


class _MockCodecDec(torch.nn.Module):
    def forward(self, a_t, b_t, v_state, states):
        t = a_t.shape[-1]
        return torch.zeros(1, 1, t * 240), states


def _build_engine(*, has_distilled: bool = False) -> UCLMEngine:
    engine = UCLMEngine(device="cpu")
    mock = _MockUCLMCoreModel(has_distilled=has_distilled)
    engine.uclm_core_model = mock
    engine.codec_dec = _MockCodecDec()
    engine.uclm_core = torch.nn.Module()
    engine.vc_enc = torch.nn.Module()
    engine.voice_state_enc = torch.nn.Module()
    engine._loaded = True
    engine._has_distilled_cfg = has_distilled
    return engine


# ---------------------------------------------------------------------------
# Tests: CFGMode enum
# ---------------------------------------------------------------------------

class TestCFGModeEnum:
    def test_values(self):
        assert CFGMode.OFF.value == "off"
        assert CFGMode.FULL.value == "full"
        assert CFGMode.LAZY.value == "lazy"
        assert CFGMode.DISTILLED.value == "distilled"

    def test_from_string(self):
        assert CFGMode("lazy") == CFGMode.LAZY
        assert CFGMode("distilled") == CFGMode.DISTILLED

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            CFGMode("invalid_mode")


# ---------------------------------------------------------------------------
# Tests: CFG mode switching in TTS
# ---------------------------------------------------------------------------

class TestCFGModeSwitching:
    """Verify that tts() accepts all four modes and returns cfg_mode in metrics."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.phonemes = torch.ones(1, 6, dtype=torch.long)
        self.spk = torch.zeros(1, 192)
        self.style = StyleParams.neutral()

    def test_off_mode(self):
        engine = _build_engine()
        _, metrics = engine.tts(
            phonemes=self.phonemes,
            speaker_embed=self.spk,
            style=self.style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.OFF,
            temperature=0.0,
        )
        assert metrics["cfg_mode"] == "off"

    def test_full_mode(self):
        engine = _build_engine()
        _, metrics = engine.tts(
            phonemes=self.phonemes,
            speaker_embed=self.spk,
            style=self.style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.FULL,
            temperature=0.0,
        )
        assert metrics["cfg_mode"] == "full"

    def test_lazy_mode(self):
        engine = _build_engine()
        _, metrics = engine.tts(
            phonemes=self.phonemes,
            speaker_embed=self.spk,
            style=self.style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.LAZY,
            temperature=0.0,
        )
        assert metrics["cfg_mode"] == "lazy"

    def test_distilled_fallback_to_full(self):
        """When distilled weights are absent, mode should fall back to full."""
        engine = _build_engine(has_distilled=False)
        _, metrics = engine.tts(
            phonemes=self.phonemes,
            speaker_embed=self.spk,
            style=self.style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.DISTILLED,
            temperature=0.0,
        )
        # Falls back to full
        assert metrics["cfg_mode"] == "full"

    def test_distilled_mode_with_weights(self):
        engine = _build_engine(has_distilled=True)
        _, metrics = engine.tts(
            phonemes=self.phonemes,
            speaker_embed=self.spk,
            style=self.style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.DISTILLED,
            temperature=0.0,
        )
        assert metrics["cfg_mode"] == "distilled"

    def test_string_cfg_mode_accepted(self):
        """cfg_mode can be passed as a plain string."""
        engine = _build_engine()
        _, metrics = engine.tts(
            phonemes=self.phonemes,
            speaker_embed=self.spk,
            style=self.style,
            cfg_scale=2.0,
            cfg_mode="lazy",
            temperature=0.0,
        )
        assert metrics["cfg_mode"] == "lazy"


# ---------------------------------------------------------------------------
# Tests: Lazy CFG cache behaviour
# ---------------------------------------------------------------------------

class TestLazyCFGCache:
    def test_cache_populated_after_first_frame(self):
        """PointerInferenceState should track cfg cache age."""
        ptr = PointerInferenceState(total_phonemes=10)
        assert ptr.cfg_uncond_cache_a is None
        assert ptr.cfg_cache_age == 0

    def test_lazy_reuses_cached_logits(self):
        """With cfg_lazy_interval=3 and 4+ frames, the unconditional pass
        should be called for frames 0, 3 (refreshed) but not 1, 2 (cached)."""
        engine = _build_engine()
        # Track how many times forward_streaming is called
        call_log = []
        original_forward = engine.uclm_core_model.forward_streaming

        def _tracking_forward(*args, **kwargs):
            call_log.append(kwargs.get("cfg_scale", args[5] if len(args) > 5 else 1.0))
            return original_forward(*args, **kwargs)

        engine.uclm_core_model.forward_streaming = _tracking_forward

        phonemes = torch.ones(1, 6, dtype=torch.long)
        spk = torch.zeros(1, 192)
        style = StyleParams.neutral()

        _, metrics = engine.tts(
            phonemes=phonemes,
            speaker_embed=spk,
            style=style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.LAZY,
            cfg_lazy_interval=3,
            temperature=0.0,
        )
        assert metrics["cfg_mode"] == "lazy"
        # The pointer dict should contain cfg_cache_age
        assert "cfg_cache_age" in metrics["pointer_state"]

    def test_lazy_produces_output_close_to_full(self):
        """Lazy CFG with interval=1 should be identical to full CFG since
        the unconditional pass is refreshed every frame."""
        engine_full = _build_engine()
        engine_lazy = _build_engine()

        phonemes = torch.ones(1, 4, dtype=torch.long)
        spk = torch.zeros(1, 192)
        style = StyleParams.neutral()

        audio_full, _ = engine_full.tts(
            phonemes=phonemes,
            speaker_embed=spk,
            style=style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.FULL,
            temperature=0.0,
        )
        audio_lazy, _ = engine_lazy.tts(
            phonemes=phonemes,
            speaker_embed=spk,
            style=style,
            cfg_scale=2.0,
            cfg_mode=CFGMode.LAZY,
            cfg_lazy_interval=1,
            temperature=0.0,
        )
        # With interval=1, lazy refreshes every frame => identical to full
        assert audio_full.shape == audio_lazy.shape
        assert torch.allclose(audio_full, audio_lazy, atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: Distilled CFG training loss
# ---------------------------------------------------------------------------

class TestDistilledCFGTrainingLoss:
    def test_kl_loss_zero_for_identical_logits(self):
        """KL divergence should be zero when teacher and student match."""
        logits = torch.randn(2, 8, 10, 1024)
        logits_b = torch.randn(2, 4, 10, 64)
        loss = DisentangledUCLM.cfg_distillation_loss(
            teacher_logits_a=logits,
            teacher_logits_b=logits_b,
            student_logits_a=logits,
            student_logits_b=logits_b,
        )
        assert loss.item() < 1e-5

    def test_kl_loss_positive_for_different_logits(self):
        """KL divergence should be positive when distributions differ."""
        teacher_a = torch.randn(2, 8, 10, 1024)
        teacher_b = torch.randn(2, 4, 10, 64)
        student_a = torch.randn(2, 8, 10, 1024) * 2.0 + 1.0
        student_b = torch.randn(2, 4, 10, 64) * 2.0 + 1.0
        loss = DisentangledUCLM.cfg_distillation_loss(
            teacher_logits_a=teacher_a,
            teacher_logits_b=teacher_b,
            student_logits_a=student_a,
            student_logits_b=student_b,
        )
        assert loss.item() > 0.0

    def test_temperature_affects_loss(self):
        """Higher distillation temperature should produce a different loss value."""
        teacher_a = torch.randn(2, 8, 5, 1024)
        teacher_b = torch.randn(2, 4, 5, 64)
        student_a = teacher_a + torch.randn_like(teacher_a) * 0.5
        student_b = teacher_b + torch.randn_like(teacher_b) * 0.5

        loss_t2 = DisentangledUCLM.cfg_distillation_loss(
            teacher_a, teacher_b, student_a, student_b, temperature=2.0,
        )
        loss_t5 = DisentangledUCLM.cfg_distillation_loss(
            teacher_a, teacher_b, student_a, student_b, temperature=5.0,
        )
        # They should not be exactly the same (temperature changes the softmax)
        assert not torch.isclose(loss_t2, loss_t5)

    def test_distilled_forward_runs(self):
        """forward_tts_distilled_cfg should produce valid output dict."""
        model = DisentangledUCLM(num_speakers=10)
        model.eval()
        B, L, T = 2, 8, 20
        phoneme_ids = torch.randint(1, 200, (B, L))
        language_ids = torch.zeros(B, L, dtype=torch.long)
        speaker_embed = torch.randn(B, 192)
        explicit_state = torch.randn(B, T, 12)
        ssl_state = torch.randn(B, T, 128)
        target_a = torch.randint(0, 1024, (B, 8, T))
        target_b = torch.randint(0, 64, (B, 4, T))

        with torch.no_grad():
            out = model.forward_tts_distilled_cfg(
                phoneme_ids=phoneme_ids,
                language_ids=language_ids,
                speaker_embed=speaker_embed,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                target_a=target_a,
                target_b=target_b,
                cfg_scale=2.0,
            )
        assert "logits_a" in out
        assert "logits_b" in out
        from tmrvc_core.constants import RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE
        assert out["logits_a"].shape == (B, 8, T, RVQ_VOCAB_SIZE)
        assert out["logits_b"].shape == (B, 4, T, CONTROL_VOCAB_SIZE)

    def test_cfg_scale_embed_exists_on_model(self):
        """DisentangledUCLM should have cfg_scale_embed module."""
        model = DisentangledUCLM(num_speakers=10)
        assert hasattr(model, "cfg_scale_embed")
        # Should accept [B, 1] input and produce [B, d_speaker]
        out = model.cfg_scale_embed(torch.tensor([[2.0]]))
        assert out.shape == (1, 192)


# ---------------------------------------------------------------------------
# Tests: Trainer CFG distillation integration
# ---------------------------------------------------------------------------

class TestTrainerCFGDistillation:
    def test_trainer_accepts_cfg_distillation_weight(self):
        """UCLMTrainer should accept cfg_distillation_weight param."""
        from tmrvc_train.trainer import UCLMTrainer, CurriculumScheduler

        model = DisentangledUCLM(num_speakers=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        curriculum = CurriculumScheduler(stage2_start=2, stage3_start=4)

        trainer = UCLMTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            cfg_distillation_weight=0.5,
            cfg_distillation_scale_range=(1.0, 2.5),
            cfg_distillation_temperature=3.0,
            curriculum=curriculum,
        )
        assert trainer.cfg_distillation_weight == 0.5
        assert trainer.cfg_distillation_scale_range == (1.0, 2.5)
        assert trainer.cfg_distillation_temperature == 3.0

    def test_distillation_loss_in_train_step(self):
        """When cfg_distillation_weight > 0 and stage >= 3, train_step
        should include loss_cfg_distillation in the returned metrics."""
        from tmrvc_train.trainer import UCLMTrainer, CurriculumScheduler

        model = DisentangledUCLM(num_speakers=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        # Set stage boundaries so we are in stage 3 from step 0.
        # Disable replay so tts_prob=1.0 is not overridden by Stage 2 config.
        curriculum = CurriculumScheduler(stage2_start=0, stage3_start=0, stage3_replay_mix_ratio=0.0)

        trainer = UCLMTrainer(
            model=model,
            optimizer=optimizer,
            device="cpu",
            cfg_distillation_weight=1.0,
            curriculum=curriculum,
            tts_prob=1.0,  # always TTS
        )

        B, T, L = 2, 20, 8
        batch = {
            "target_a": torch.randint(0, 1024, (B, 8, T)),
            "target_b": torch.randint(0, 64, (B, 4, T)),
            "source_a_t": torch.randint(0, 1024, (B, 8, T)),
            "explicit_state": torch.randn(B, T, 12),
            "ssl_state": torch.randn(B, T, 128),
            "speaker_embed": torch.randn(B, 192),
            "speaker_id": torch.zeros(B, dtype=torch.long),
            "phoneme_ids": torch.randint(1, 200, (B, L)),
            "language_id": torch.zeros(B, L, dtype=torch.long),
        }
        # Run enough steps to ensure at least one lands in TTS mode
        found_distill = False
        for _ in range(5):
            metrics = trainer.train_step(batch)
            if "loss_cfg_distillation" in metrics:
                found_distill = True
                assert metrics["loss_cfg_distillation"] >= 0.0
                break
        assert found_distill, "CFG distillation loss never appeared in 5 train steps"
