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
        torch.save({"model": uclm.state_dict()}, uclm_path)
        
        engine = UCLMEngine(device="cpu")
        engine.load_from_combined_checkpoint(uclm_path)
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
        def mock_vstate(v, s, d): return torch.zeros(1, 512)
        def mock_uclm(c, b, s, st, cfg, kv): return torch.zeros(1, 8, 1, 1024), torch.zeros(1, 4, 1, 64), kv
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
