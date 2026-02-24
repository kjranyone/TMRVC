"""Tests for the FastAPI TTS server schemas and utilities."""

import pytest


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
    WSAudioChunk,
    WSCommentMessage,
    WSResponseMessage,
    WSStyleInfo,
)


class TestTTSRequest:
    def test_valid_request(self):
        req = TTSRequest(text="こんにちは", character_id="sakura")
        assert req.text == "こんにちは"
        assert req.speed == 1.0
        assert req.emotion is None
        assert req.context is None

    def test_with_options(self):
        req = TTSRequest(
            text="テスト",
            character_id="sakura",
            emotion="happy",
            speed=1.5,
            situation="学校で",
        )
        assert req.emotion == "happy"
        assert req.speed == 1.5

    def test_speed_bounds(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            TTSRequest(text="test", character_id="x", speed=3.0)
        with pytest.raises(Exception):
            TTSRequest(text="test", character_id="x", speed=0.1)


class TestTTSResponse:
    def test_creation(self):
        resp = TTSResponse(
            audio_base64="AAAA",
            duration_sec=1.5,
        )
        assert resp.sample_rate == 24000


class TestCharacterSchemas:
    def test_character_info(self):
        info = CharacterInfo(id="sakura", name="桜")
        assert info.language == "ja"

    def test_character_create_request(self):
        req = CharacterCreateRequest(
            id="sakura",
            name="桜",
            personality="明るい",
        )
        assert req.id == "sakura"


class TestHealthResponse:
    def test_defaults(self):
        h = HealthResponse()
        assert h.status == "ok"
        assert h.models_loaded is False
        assert h.characters_count == 0


class TestWSSchemas:
    def test_comment_message(self):
        msg = WSCommentMessage(text="こんにちは", user="viewer1")
        assert msg.type == "comment"
        assert msg.priority == 2

    def test_response_message(self):
        msg = WSResponseMessage(text="やっほー", character_id="sakura")
        assert msg.type == "response"

    def test_audio_chunk(self):
        chunk = WSAudioChunk(data="base64data", frame_index=5, is_last=True)
        assert chunk.type == "audio_chunk"
        assert chunk.is_last is True

    def test_style_info(self):
        info = WSStyleInfo(emotion="happy", vad=[0.7, 0.5, 0.3])
        assert info.type == "style_info"
        assert len(info.vad) == 3


class TestWavEncoding:
    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_audio_to_wav_base64(self):
        import base64
        import numpy as np
        from tmrvc_serve.app import _audio_to_wav_base64

        audio = np.zeros(2400, dtype=np.float32)  # 100ms of silence
        result = _audio_to_wav_base64(audio, sr=24000)

        # Should be valid base64
        decoded = base64.b64decode(result)
        # WAV header starts with RIFF
        assert decoded[:4] == b"RIFF"
        assert decoded[8:12] == b"WAVE"
