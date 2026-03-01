"""Tests for the FastAPI UCLM v2 server schemas and utilities."""

import pytest
import torch
import numpy as np
from pathlib import Path


def _has_fastapi() -> bool:
    try:
        import fastapi
        return True
    except ImportError:
        return False


from tmrvc_serve.schemas import (
    CharacterCreateRequest,
    CharacterInfo,
    HealthResponse,
    TTSRequest,
    TTSResponse,
)
from tmrvc_core.dialogue_types import StyleParams


class TestTTSRequest:
    def test_valid_request(self):
        req = TTSRequest(text="こんにちは", character_id="sakura")
        assert req.text == "こんにちは"
        assert req.speed == 1.0
        assert req.emotion is None
        assert req.style_preset == "default"

    def test_with_uclm_style_options(self):
        req = TTSRequest(
            text="テスト",
            character_id="sakura",
            emotion="happy",
            style_preset="asmr_soft",
            speed=1.5,
            breathiness=0.8,
            tension=0.2,
        )
        assert req.emotion == "happy"
        assert req.style_preset == "asmr_soft"
        assert req.speed == 1.5
        assert req.breathiness == 0.8
        assert req.tension == 0.2


class TestStyleParamsV2:
    def test_to_vector_8dim(self):
        style = StyleParams(
            breathiness=0.5,
            tension=0.3,
            arousal=0.7,
            valence=0.1,
            roughness=0.0,
            voicing=1.0,
            energy=0.6,
            speech_rate=1.2,
        )
        vec = style.to_vector()
        assert len(vec) == 8
        assert vec[0] == 0.5
        assert vec[7] == 1.2

    def test_from_dict(self):
        d = {"breathiness": 0.9, "emotion": "whisper"}
        style = StyleParams.from_dict(d)
        assert style.breathiness == 0.9
        assert style.emotion == "whisper"
        assert style.tension == 0.0 # default


class TestUCLMEngineLogic:
    def test_engine_initialization(self, tmp_path):
        from tmrvc_serve.uclm_engine import UCLMEngine
        from tmrvc_train.models import DisentangledUCLM, EmotionAwareCodec
        
        uclm = DisentangledUCLM()
        codec = EmotionAwareCodec()
        
        uclm_path = tmp_path / "uclm.pt"
        codec_path = tmp_path / "codec.pt"
        
        torch.save({"model": uclm.state_dict()}, uclm_path)
        torch.save({"model": codec.state_dict()}, codec_path)
        
        engine = UCLMEngine(uclm_checkpoint=uclm_path, codec_checkpoint=codec_path, device="cpu")
        engine.load_models()
        assert engine._loaded is True
        assert engine.uclm is not None
        assert engine.codec is not None

    def test_vc_frame_processing(self, monkeypatch):
        from tmrvc_serve.uclm_engine import UCLMEngine
        
        engine = UCLMEngine(device="cpu")
        # Mock models
        engine.uclm = torch.nn.Module()
        engine.codec = torch.nn.Module()
        
        def mock_encode(audio):
            return torch.zeros(1, 8, 1), torch.zeros(1, 4, 1)
        def mock_vc_encoder(tokens):
            return torch.zeros(1, 1, 512), None
        def mock_voice_state_enc(v, ssl):
            return torch.zeros(1, 1, 512)
        def mock_forward_streaming(content, state, spk, kv_cache_in=None):
            return {
                "logits_a": torch.zeros(1, 8, 1, 1024),
                "logits_b": torch.zeros(1, 4, 1, 64),
                "kv_cache_out": torch.zeros(1, 1, 1)
            }
        def mock_decode(a, b, v):
            return torch.zeros(1, 1, 240)

        engine.codec.encode = mock_encode
        engine.uclm.vc_encoder = mock_vc_encoder
        engine.uclm.voice_state_enc = mock_voice_state_enc
        engine.uclm.forward_streaming = mock_forward_streaming
        engine.codec.decode = mock_decode
        engine._loaded = True
        
        audio_in = torch.zeros(1, 1, 240)
        spk = torch.zeros(1, 192)
        style = StyleParams.neutral()
        
        audio_out, kv = engine.vc_frame(audio_in, spk, style)
        assert audio_out.shape == (240,)
        assert kv is not None


class TestWavEncoding:
    @pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
    def test_audio_to_wav_base64(self):
        import base64
        from tmrvc_serve._helpers import _audio_to_wav_base64

        audio = np.zeros(2400, dtype=np.float32)
        result = _audio_to_wav_base64(audio, sr=24000)
        decoded = base64.b64decode(result)
        assert decoded[:4] == b"RIFF"


class TestStylePresetHelpersV2:
    @pytest.mark.skipif(not _has_fastapi(), reason="fastapi not installed")
    def test_asmr_soft_preset(self):
        from tmrvc_serve.style_resolver import _resolve_style_preset
        from tmrvc_serve.schemas import StylePreset

        style, cfg = _resolve_style_preset(None, StylePreset.ASMR_SOFT)
        assert style.emotion == "whisper"
        assert style.breathiness > 0.5
        assert style.voicing < 1.0
