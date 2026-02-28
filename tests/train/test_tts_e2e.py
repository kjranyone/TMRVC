"""End-to-end TTS pipeline integration tests.

Tests the full text→audio pipeline with real model forward passes
(random weights, no checkpoints needed).
"""

import torch
import pytest

from tmrvc_core.constants import D_CONTENT, D_STYLE, D_TEXT_ENCODER


class TestTTSPipelineE2E:
    """Full pipeline: TextEncoder → DurationPredictor → length_regulate
    → F0Predictor → ContentSynthesizer → content[B,256,T]."""

    def test_full_pipeline_shapes(self):
        from tmrvc_train.models.text_encoder import TextEncoder
        from tmrvc_train.models.duration_predictor import DurationPredictor
        from tmrvc_train.models.f0_predictor import F0Predictor, length_regulate
        from tmrvc_train.models.content_synthesizer import ContentSynthesizer

        B, L = 2, 20

        text_enc = TextEncoder().eval()
        dur_pred = DurationPredictor().eval()
        f0_pred = F0Predictor().eval()
        content_synth = ContentSynthesizer().eval()

        phoneme_ids = torch.randint(1, 200, (B, L))
        language_ids = torch.zeros(B, dtype=torch.long)
        style_vec = torch.zeros(B, D_STYLE)

        with torch.no_grad():
            # Step 1: Text encoding
            text_features = text_enc(phoneme_ids, language_ids)
            assert text_features.shape == (B, D_TEXT_ENCODER, L)

            # Step 2: Duration prediction
            durations = dur_pred(text_features, style_vec)
            assert durations.shape == (B, L)
            assert (durations >= 0).all(), "Durations must be non-negative"

            # Step 3: Length regulation
            dur_int = torch.round(durations).long().clamp(min=1)
            expanded = length_regulate(text_features, dur_int.float())
            T = dur_int.sum(dim=-1)
            # T can vary per batch; expanded is padded to max T
            assert expanded.shape[0] == B
            assert expanded.shape[1] == D_TEXT_ENCODER
            assert expanded.shape[2] >= T.min().item()

            # Step 4: F0 prediction
            f0, voiced = f0_pred(expanded, style_vec)
            assert f0.shape == (B, 1, expanded.shape[2])
            assert voiced.shape == (B, 1, expanded.shape[2])

            # Step 5: Content synthesis
            content = content_synth(expanded)
            assert content.shape == (B, D_CONTENT, expanded.shape[2])

    def test_pipeline_with_style_conditioning(self):
        from tmrvc_train.models.text_encoder import TextEncoder
        from tmrvc_train.models.duration_predictor import DurationPredictor
        from tmrvc_train.models.f0_predictor import F0Predictor, length_regulate
        from tmrvc_train.models.content_synthesizer import ContentSynthesizer
        from tmrvc_core.dialogue_types import StyleParams

        text_enc = TextEncoder().eval()
        dur_pred = DurationPredictor().eval()
        f0_pred = F0Predictor().eval()
        content_synth = ContentSynthesizer().eval()

        # Build style from dialogue types
        style = StyleParams(emotion="happy", valence=0.8, arousal=0.5, energy=0.3)
        style_vec = torch.tensor([style.to_vector()], dtype=torch.float32)
        assert style_vec.shape == (1, 32)

        phoneme_ids = torch.randint(1, 200, (1, 15))
        language_ids = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            text_features = text_enc(phoneme_ids, language_ids)
            durations = dur_pred(text_features, style_vec)
            dur_int = torch.round(durations).long().clamp(min=1)
            expanded = length_regulate(text_features, dur_int.float())
            f0, voiced = f0_pred(expanded, style_vec)
            content = content_synth(expanded)

        assert content.shape[1] == D_CONTENT
        assert f0.shape[2] == content.shape[2]

    def test_pipeline_with_converter(self):
        """Full pipeline including VC backend (Converter + Vocoder)."""
        from tmrvc_train.models.text_encoder import TextEncoder
        from tmrvc_train.models.duration_predictor import DurationPredictor
        from tmrvc_train.models.f0_predictor import F0Predictor, length_regulate
        from tmrvc_train.models.content_synthesizer import ContentSynthesizer
        from tmrvc_train.models.converter import ConverterStudent
        from tmrvc_train.models.vocoder import VocoderStudent

        text_enc = TextEncoder().eval()
        dur_pred = DurationPredictor().eval()
        f0_pred = F0Predictor().eval()
        content_synth = ContentSynthesizer().eval()
        converter = ConverterStudent().eval()
        vocoder = VocoderStudent().eval()

        phoneme_ids = torch.randint(1, 200, (1, 10))
        language_ids = torch.zeros(1, dtype=torch.long)
        style_vec = torch.zeros(1, D_STYLE)
        spk_embed = torch.randn(1, 192)
        acoustic_params = torch.zeros(1, 32)

        with torch.no_grad():
            text_features = text_enc(phoneme_ids, language_ids)
            durations = dur_pred(text_features, style_vec)
            dur_int = torch.round(durations).long().clamp(min=1)
            expanded = length_regulate(text_features, dur_int.float())
            f0, voiced = f0_pred(expanded, style_vec)
            content = content_synth(expanded)

            # VC backend
            pred_features, _ = converter(content, spk_embed, acoustic_params)
            stft_mag, stft_phase, _ = vocoder(pred_features)

        T = content.shape[2]
        assert stft_mag.shape == (1, 513, T)
        assert stft_phase.shape == (1, 513, T)


class TestConverterFiLMMigration:
    """Test FiLM weight migration from VC (d_cond=224) to TTS (d_cond=256)."""

    def test_migrate_preserves_vc_behavior(self):
        from tmrvc_train.models.converter import (
            ConverterStudent,
            converter_from_vc_checkpoint,
        )

        vc_model = ConverterStudent().eval()
        tts_model = converter_from_vc_checkpoint(vc_model).eval()

        content = torch.randn(1, D_CONTENT, 10)
        spk_embed = torch.randn(1, 192)
        acoustic_params = torch.randn(1, 32)

        # VC forward
        with torch.no_grad():
            vc_out, _ = vc_model(content, spk_embed, acoustic_params)

        # TTS forward with zero style extension (should = VC output)
        style_params = torch.cat([acoustic_params, torch.zeros(1, 32)], dim=-1)
        with torch.no_grad():
            tts_out, _ = tts_model(content, spk_embed, style_params)

        # Should be identical since extra FiLM dims are zero-initialized
        torch.testing.assert_close(vc_out, tts_out, atol=1e-6, rtol=1e-5)

    def test_tts_model_accepts_extended_cond(self):
        from tmrvc_train.models.converter import converter_from_vc_checkpoint, ConverterStudent

        vc_model = ConverterStudent().eval()
        tts_model = converter_from_vc_checkpoint(vc_model).eval()

        content = torch.randn(1, D_CONTENT, 10)
        spk_embed = torch.randn(1, 192)
        style_params = torch.randn(1, 64)  # full 64-dim style

        with torch.no_grad():
            out, _ = tts_model(content, spk_embed, style_params)

        assert out.shape == (1, 513, 10)


class TestStyleEncoderONNX:
    def test_export_and_parity(self, tmp_path):
        ort = pytest.importorskip("onnxruntime")
        from tmrvc_train.models.style_encoder import StyleEncoder
        from tmrvc_export.export_tts import export_style_encoder, verify_style_encoder

        model = StyleEncoder()
        path = export_style_encoder(model, tmp_path / "style_encoder.onnx")
        assert path.exists()

        results = verify_style_encoder(model, path)
        for r in results:
            assert r["passed"], f"Parity failed: {r}"
