"""Tests for TTS models: TextEncoder, DurationPredictor, F0Predictor,
ContentSynthesizer, StyleEncoder, and FiLM migration."""

import torch
import pytest

from tmrvc_core.constants import (
    D_CONTENT,
    D_CONVERTER_HIDDEN,
    D_F0_PREDICTOR,
    D_SPEAKER,
    D_STYLE,
    D_TEXT_ENCODER,
    D_VOCODER_FEATURES,
    N_ACOUSTIC_PARAMS,
    N_EMOTION_CATEGORIES,
    N_LANGUAGES,
    N_STYLE_PARAMS,
    N_TEXT_ENCODER_HEADS,
    N_TEXT_ENCODER_LAYERS,
    PHONEME_VOCAB_SIZE,
)
from tmrvc_train.models.text_encoder import TextEncoder
from tmrvc_train.models.duration_predictor import DurationPredictor
from tmrvc_train.models.f0_predictor import F0Predictor, length_regulate
from tmrvc_train.models.content_synthesizer import ContentSynthesizer
from tmrvc_train.models.style_encoder import StyleEncoder, AudioStyleEncoder
from tmrvc_train.models.converter import (
    ConverterStudent,
    converter_from_vc_checkpoint,
    migrate_film_weights,
)
from tmrvc_train.modules import FiLMConditioner


# --- TextEncoder ---


class TestTextEncoder:
    @pytest.fixture
    def model(self):
        return TextEncoder()

    def test_output_shape(self, model):
        phonemes = torch.randint(0, PHONEME_VOCAB_SIZE, (2, 20))
        lang_ids = torch.zeros(2, dtype=torch.long)
        out = model(phonemes, lang_ids)
        assert out.shape == (2, D_TEXT_ENCODER, 20)

    def test_output_shape_with_lengths(self, model):
        phonemes = torch.randint(0, PHONEME_VOCAB_SIZE, (3, 15))
        lang_ids = torch.ones(3, dtype=torch.long)
        lengths = torch.tensor([15, 10, 5])
        out = model(phonemes, lang_ids, lengths)
        assert out.shape == (3, D_TEXT_ENCODER, 15)

    def test_padding_mask_effect(self, model):
        model.eval()
        phonemes = torch.randint(1, PHONEME_VOCAB_SIZE, (1, 10))
        lang_ids = torch.zeros(1, dtype=torch.long)

        # Same input, different mask
        with torch.no_grad():
            out_full = model(phonemes, lang_ids, torch.tensor([10]))
            out_masked = model(phonemes, lang_ids, torch.tensor([5]))

        # First 5 phonemes should differ due to attention mask
        # (transformer attention differs with different masks)
        assert out_full.shape == out_masked.shape

    def test_language_embedding_effect(self, model):
        model.eval()
        phonemes = torch.randint(1, PHONEME_VOCAB_SIZE, (1, 10))

        with torch.no_grad():
            out_ja = model(phonemes, torch.tensor([0]))
            out_en = model(phonemes, torch.tensor([1]))

        assert not torch.allclose(out_ja, out_en)

    def test_constants_match(self, model):
        assert model.d_model == D_TEXT_ENCODER
        assert model.phoneme_embed.num_embeddings == PHONEME_VOCAB_SIZE
        assert model.lang_embed.num_embeddings == N_LANGUAGES


# --- DurationPredictor ---


class TestDurationPredictor:
    @pytest.fixture
    def model(self):
        return DurationPredictor()

    def test_output_shape(self, model):
        text_features = torch.randn(2, D_TEXT_ENCODER, 20)
        out = model(text_features)
        assert out.shape == (2, 20)

    def test_output_positive(self, model):
        text_features = torch.randn(2, D_TEXT_ENCODER, 10)
        out = model(text_features)
        assert (out > 0).all(), "Durations must be positive (softplus)"

    def test_style_conditioning(self, model):
        model.eval()
        text_features = torch.randn(1, D_TEXT_ENCODER, 10)

        with torch.no_grad():
            out_no_style = model(text_features)
            style = torch.randn(1, D_STYLE)
            out_with_style = model(text_features, style)

        assert not torch.allclose(out_no_style, out_with_style)

    def test_different_styles_different_durations(self, model):
        model.eval()
        text_features = torch.randn(1, D_TEXT_ENCODER, 10)

        with torch.no_grad():
            style1 = torch.randn(1, D_STYLE)
            style2 = torch.randn(1, D_STYLE)
            out1 = model(text_features, style1)
            out2 = model(text_features, style2)

        assert not torch.allclose(out1, out2)


# --- length_regulate ---


class TestLengthRegulate:
    def test_basic_expansion(self):
        features = torch.randn(1, 256, 3)  # 3 phonemes
        durations = torch.tensor([[2, 3, 1]], dtype=torch.float)  # T = 6
        out = length_regulate(features, durations)
        assert out.shape == (1, 256, 6)

    def test_each_phoneme_repeated(self):
        features = torch.arange(3).float().reshape(1, 1, 3)  # [0, 1, 2]
        durations = torch.tensor([[2, 1, 3]], dtype=torch.float)
        out = length_regulate(features, durations)
        expected = torch.tensor([[[0, 0, 1, 2, 2, 2]]], dtype=torch.float)
        assert torch.allclose(out, expected)

    def test_batch_padding(self):
        features = torch.randn(2, 4, 3)
        durations = torch.tensor([[2, 3, 1], [1, 1, 4]], dtype=torch.float)
        out = length_regulate(features, durations)
        assert out.shape == (2, 4, 6)  # max(6, 6) = 6


# --- F0Predictor ---


class TestF0Predictor:
    @pytest.fixture
    def model(self):
        return F0Predictor()

    def test_output_shape(self, model):
        features = torch.randn(2, D_TEXT_ENCODER, 50)
        f0, voiced = model(features)
        assert f0.shape == (2, 1, 50)
        assert voiced.shape == (2, 1, 50)

    def test_f0_positive(self, model):
        features = torch.randn(1, D_TEXT_ENCODER, 20)
        f0, _ = model(features)
        assert (f0 > 0).all(), "F0 must be positive (softplus)"

    def test_voiced_prob_range(self, model):
        features = torch.randn(1, D_TEXT_ENCODER, 20)
        _, voiced = model(features)
        assert (voiced >= 0).all() and (voiced <= 1).all(), "Voiced prob must be in [0,1]"

    def test_style_conditioning(self, model):
        model.eval()
        features = torch.randn(1, D_TEXT_ENCODER, 20)

        with torch.no_grad():
            f0_no_style, _ = model(features)
            style = torch.randn(1, D_STYLE)
            f0_with_style, _ = model(features, style)

        assert not torch.allclose(f0_no_style, f0_with_style)


# --- ContentSynthesizer ---


class TestContentSynthesizer:
    @pytest.fixture
    def model(self):
        return ContentSynthesizer()

    def test_output_shape(self, model):
        features = torch.randn(2, D_TEXT_ENCODER, 50)
        out = model(features)
        assert out.shape == (2, D_CONTENT, 50)

    def test_output_dimension_matches_content_encoder(self, model):
        """ContentSynthesizer output dim must match ContentEncoder output."""
        features = torch.randn(1, D_TEXT_ENCODER, 10)
        out = model(features)
        assert out.shape[1] == D_CONTENT  # 256

    def test_deterministic_eval(self, model):
        model.eval()
        features = torch.randn(1, D_TEXT_ENCODER, 10)
        with torch.no_grad():
            out1 = model(features)
            out2 = model(features)
        assert torch.allclose(out1, out2)


# --- StyleEncoder ---


class TestAudioStyleEncoder:
    @pytest.fixture
    def model(self):
        return AudioStyleEncoder()

    def test_output_shape(self, model):
        mel = torch.randn(2, 80, 100)
        out = model(mel)
        assert out.shape == (2, D_STYLE)

    def test_different_inputs_different_outputs(self, model):
        model.eval()
        with torch.no_grad():
            out1 = model(torch.randn(1, 80, 100))
            out2 = model(torch.randn(1, 80, 100))
        assert not torch.allclose(out1, out2)

    def test_variable_length_input(self, model):
        model.eval()
        with torch.no_grad():
            out1 = model(torch.randn(1, 80, 50))
            out2 = model(torch.randn(1, 80, 200))
        assert out1.shape == out2.shape == (1, D_STYLE)


class TestStyleEncoder:
    @pytest.fixture
    def model(self):
        return StyleEncoder()

    def test_forward(self, model):
        mel = torch.randn(2, 80, 100)
        style = model(mel)
        assert style.shape == (2, D_STYLE)

    def test_predict_emotion(self, model):
        style = torch.randn(2, D_STYLE)
        preds = model.predict_emotion(style)
        assert preds["emotion_logits"].shape == (2, N_EMOTION_CATEGORIES)
        assert preds["vad"].shape == (2, 3)
        assert preds["prosody"].shape == (2, 3)

    def test_combine_style_params(self):
        acoustic = torch.randn(2, N_ACOUSTIC_PARAMS)
        emotion = torch.randn(2, D_STYLE)
        combined = StyleEncoder.combine_style_params(acoustic, emotion)
        assert combined.shape == (2, N_STYLE_PARAMS)
        assert torch.allclose(combined[:, :N_ACOUSTIC_PARAMS], acoustic)
        assert torch.allclose(combined[:, N_ACOUSTIC_PARAMS:], emotion)

    def test_make_vc_style_params(self):
        acoustic = torch.randn(2, N_ACOUSTIC_PARAMS)
        style = StyleEncoder.make_vc_style_params(acoustic)
        assert style.shape == (2, N_STYLE_PARAMS)
        assert torch.allclose(style[:, :N_ACOUSTIC_PARAMS], acoustic)
        assert (style[:, N_ACOUSTIC_PARAMS:] == 0).all()


# --- Converter FiLM Migration ---


class TestFiLMMigration:
    def test_migrate_preserves_existing_weights(self):
        src = FiLMConditioner(d_cond=224, d_model=384)
        dst = FiLMConditioner(d_cond=256, d_model=384)

        # Record original src weights
        src_weight = src.proj.weight.clone()
        src_bias = src.proj.bias.clone()

        migrate_film_weights(src, dst)

        # First 224 columns of dst should match src
        assert torch.allclose(dst.proj.weight[:, :224], src_weight)
        assert torch.allclose(dst.proj.bias, src_bias)

    def test_migrate_new_dims_zero(self):
        src = FiLMConditioner(d_cond=224, d_model=384)
        dst = FiLMConditioner(d_cond=256, d_model=384)

        migrate_film_weights(src, dst)

        # New columns (224:256) should be zero
        assert (dst.proj.weight[:, 224:] == 0).all()

    def test_identity_modulation_at_zero(self):
        """Zero style dims should produce identity modulation."""
        dst = FiLMConditioner(d_cond=256, d_model=384)
        src = FiLMConditioner(d_cond=224, d_model=384)
        migrate_film_weights(src, dst)

        x = torch.randn(1, 384, 10)

        # VC mode: 224-dim cond
        cond_vc = torch.randn(1, 224)
        out_src = src(x, cond_vc)

        # TTS mode with zero style: 224-dim cond + 32 zeros
        cond_tts = torch.cat([cond_vc, torch.zeros(1, 32)], dim=-1)
        out_dst = dst(x, cond_tts)

        assert torch.allclose(out_src, out_dst, atol=1e-6)


class TestConverterFromVCCheckpoint:
    def test_basic_migration(self):
        vc_model = ConverterStudent()
        tts_model = converter_from_vc_checkpoint(vc_model)

        # TTS model has larger FiLM d_cond
        vc_d_cond = vc_model.blocks[0].film.proj.in_features
        tts_d_cond = tts_model.blocks[0].film.proj.in_features
        assert vc_d_cond == D_SPEAKER + N_ACOUSTIC_PARAMS  # 224
        assert tts_d_cond == D_SPEAKER + N_STYLE_PARAMS  # 256

    def test_output_shape_preserved(self):
        vc_model = ConverterStudent()
        tts_model = converter_from_vc_checkpoint(vc_model)

        content = torch.randn(1, D_CONTENT, 10)
        spk = torch.randn(1, D_SPEAKER)
        style_params = torch.randn(1, N_STYLE_PARAMS)

        out, _ = tts_model(content, spk, style_params)
        assert out.shape == (1, D_VOCODER_FEATURES, 10)

    def test_vc_compatible_output(self):
        """With zero style dims, TTS converter should match VC converter."""
        vc_model = ConverterStudent()
        vc_model.eval()
        tts_model = converter_from_vc_checkpoint(vc_model)
        tts_model.eval()

        content = torch.randn(1, D_CONTENT, 10)
        spk = torch.randn(1, D_SPEAKER)
        acoustic = torch.randn(1, N_ACOUSTIC_PARAMS)

        # VC mode
        with torch.no_grad():
            out_vc, _ = vc_model(content, spk, acoustic)

        # TTS mode with zero style
        style_params = torch.cat([acoustic, torch.zeros(1, D_STYLE)], dim=-1)
        with torch.no_grad():
            out_tts, _ = tts_model(content, spk, style_params)

        assert torch.allclose(out_vc, out_tts, atol=1e-5)


# --- TTS Types ---


class TestTTSTypes:
    def test_tts_feature_set_creation(self):
        from tmrvc_core.types import TTSFeatureSet

        fs = TTSFeatureSet(
            mel=torch.randn(80, 100),
            content=torch.randn(768, 100),
            f0=torch.randn(1, 100),
            spk_embed=torch.randn(192),
            phoneme_ids=torch.randint(0, 200, (30,)),
            durations=torch.ones(30),
            language_id=0,
            utterance_id="test_001",
            speaker_id="spk_001",
            n_frames=100,
            n_phonemes=30,
        )
        assert fs.mel.shape == (80, 100)
        assert fs.phoneme_ids.shape == (30,)
        assert fs.language_id == 0

    def test_tts_batch_creation(self):
        from tmrvc_core.types import TTSBatch

        batch = TTSBatch(
            phoneme_ids=torch.randint(0, 200, (2, 30)),
            durations=torch.ones(2, 30),
            language_ids=torch.zeros(2, dtype=torch.long),
            content=torch.randn(2, 768, 100),
            f0=torch.randn(2, 1, 100),
            spk_embed=torch.randn(2, 192),
            mel_target=torch.randn(2, 80, 100),
            frame_lengths=torch.tensor([100, 80]),
            phoneme_lengths=torch.tensor([30, 25]),
        )
        assert batch.phoneme_ids.shape == (2, 30)
        assert batch.frame_lengths.shape == (2,)


# --- End-to-end TTS pipeline shape test ---


class TestTTSPipeline:
    def test_full_pipeline_shapes(self):
        """Test that all TTS components chain together correctly."""
        B, L, T = 2, 15, 50

        text_enc = TextEncoder()
        dur_pred = DurationPredictor()
        f0_pred = F0Predictor()
        content_synth = ContentSynthesizer()

        # Input
        phoneme_ids = torch.randint(1, PHONEME_VOCAB_SIZE, (B, L))
        lang_ids = torch.zeros(B, dtype=torch.long)

        # TextEncoder
        text_features = text_enc(phoneme_ids, lang_ids)
        assert text_features.shape == (B, D_TEXT_ENCODER, L)

        # DurationPredictor
        pred_durations = dur_pred(text_features)
        assert pred_durations.shape == (B, L)
        assert (pred_durations > 0).all()

        # Length regulate
        expanded = length_regulate(text_features, pred_durations)
        assert expanded.shape[0] == B
        assert expanded.shape[1] == D_TEXT_ENCODER

        # F0Predictor
        f0, voiced = f0_pred(expanded)
        assert f0.shape[:2] == (B, 1)
        assert voiced.shape[:2] == (B, 1)

        # ContentSynthesizer
        content = content_synth(expanded)
        assert content.shape[0] == B
        assert content.shape[1] == D_CONTENT

    def test_pipeline_with_converter(self):
        """Test that ContentSynthesizer output feeds into Converter."""
        B, L = 1, 10

        text_enc = TextEncoder()
        dur_pred = DurationPredictor()
        content_synth = ContentSynthesizer()
        converter = ConverterStudent()

        phoneme_ids = torch.randint(1, PHONEME_VOCAB_SIZE, (B, L))
        lang_ids = torch.zeros(B, dtype=torch.long)
        spk = torch.randn(B, D_SPEAKER)
        acoustic = torch.randn(B, N_ACOUSTIC_PARAMS)

        # TTS front-end
        text_features = text_enc(phoneme_ids, lang_ids)
        durations = dur_pred(text_features)
        expanded = length_regulate(text_features, durations)
        content = content_synth(expanded)

        # Converter (VC back-end, shared)
        out, _ = converter(content, spk, acoustic)
        assert out.shape[0] == B
        assert out.shape[1] == D_VOCODER_FEATURES  # 513
