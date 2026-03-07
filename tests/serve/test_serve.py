"""Tests for the FastAPI UCLM v2 server schemas and utilities."""

import pytest
import torch
import numpy as np
from tmrvc_serve.schemas import (
    HealthResponse,
    TTSRequest,
    TTSResponse,
)
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.uclm_engine import UCLMEngine, EngineState


class TestTTSRequest:
    def test_valid_request(self):
        req = TTSRequest(text="こんにちは", character_id="sakura")
        assert req.text == "こんにちは"
        assert req.style_preset == "default"


class TestUCLMEngineLogic:
    def test_engine_initialization(self, tmp_path):
        from tmrvc_train.models import DisentangledUCLM, EmotionAwareCodec
        uclm = DisentangledUCLM()
        codec = EmotionAwareCodec()
        uclm_path = tmp_path / "uclm.pt"
        codec_path = tmp_path / "codec.pt"
        torch.save({"model": uclm.state_dict()}, uclm_path)
        torch.save({"model": codec.state_dict()}, codec_path)

        engine = UCLMEngine(device="cpu")
        engine.load_models(uclm_path=uclm_path, codec_path=codec_path)
        assert engine._loaded is True

    def test_vc_frame_processing(self, monkeypatch):
        engine = UCLMEngine(device="cpu")
        # Mock sub-components
        engine.codec_enc = torch.nn.Module()
        engine.vc_enc = torch.nn.Module()
        engine.voice_state_enc = torch.nn.Module()
        engine.uclm_core = torch.nn.Module()
        engine.codec_dec = torch.nn.Module()

        def mock_enc(audio, states): return torch.zeros(1, 8, 1), None, states
        def mock_vc(tokens): return torch.zeros(1, 512, 1), None
        def mock_vstate(v, s): return (torch.zeros(1, 1, 512),)
        def mock_uclm(c, b, s, st, cfg, kv): return torch.zeros(1, 8, 1, 1024), torch.zeros(1, 4, 1, 64), kv, torch.zeros(1, 1, 512)
        def mock_dec(a, b, v, s): return torch.zeros(1, 1, 240), s

        engine.codec_enc.forward = mock_enc
        engine.vc_enc.forward = mock_vc
        engine.voice_state_enc.forward = mock_vstate
        engine.uclm_core.forward = mock_uclm
        engine.codec_dec.forward = mock_dec
        engine._loaded = True

        state = EngineState()
        audio_in = torch.zeros(1, 1, 240)
        spk = torch.zeros(1, 192)
        style = StyleParams.neutral()

        audio_out, next_state = engine.vc_frame(audio_in, spk, style, state)
        assert audio_out.shape == (240,)
        assert isinstance(next_state, EngineState)

    def test_tts_returns_audio_and_metrics(self):
        """Verify that engine.tts() returns audio tensor and metrics dict.

        The v3 engine uses pointer-based causal generation internally.
        We mock the core model's streaming path used by tts().
        """
        engine = UCLMEngine(device="cpu")

        class _MockTextEncoder(torch.nn.Module):
            def forward(self, phoneme_ids, lang_ids, phoneme_lens):
                B, L = phoneme_ids.shape
                return torch.randn(B, 512, L)  # [B, d_model, L]

        class _MockUCLMCoreModel:
            text_encoder = _MockTextEncoder()

            def forward_streaming(self, content_features, b_ctx, speaker_embed, state_cond, cfg_scale=1.0, kv_caches=None, dialogue_context=None, acting_intent=None, prosody_latent=None):
                B = content_features.shape[0]
                return {
                    "logits_a": torch.zeros(B, 8, 1, 1024),
                    "logits_b": torch.zeros(B, 4, 1, 64),
                    "kv_cache_out": kv_caches,
                    "hidden_states": torch.zeros(B, 1, 512),
                }

            _call_count = 0

            def pointer_head(self, hidden_states):
                self._call_count += 1
                # Advance quickly so the loop terminates
                adv_logit = torch.tensor([[[-5.0]]])  # low prob to not advance
                progress = torch.tensor([[[0.5]]])
                if self._call_count > 3:
                    adv_logit = torch.tensor([[[10.0]]])  # high prob -> advance
                    progress = torch.tensor([[[1.0]]])
                return adv_logit, progress

        class _MockCodecDec(torch.nn.Module):
            def forward(self, a_t, b_t, v_state, states):
                t = a_t.shape[-1]
                return torch.zeros(1, 1, t * 240), states

        engine.uclm_core_model = _MockUCLMCoreModel()
        engine.codec_dec = _MockCodecDec()
        engine.uclm_core = torch.nn.Module()
        engine.vc_enc = torch.nn.Module()
        engine.voice_state_enc = torch.nn.Module()
        engine._loaded = True

        phonemes = torch.ones(1, 6, dtype=torch.long)
        spk = torch.zeros(1, 192)
        style = StyleParams.neutral()

        audio, metrics = engine.tts(phonemes=phonemes, speaker_embed=spk, style=style, temperature=0.0)
        assert isinstance(audio, torch.Tensor)
        assert "rtf" in metrics and "gen_time_ms" in metrics
