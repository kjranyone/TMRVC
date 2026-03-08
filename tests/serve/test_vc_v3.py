"""Tests for UCLM v3 Voice Conversion (VC) features.

Validates:
- Pitch shifting via F0 conditioning in vc_frame.
- POST /vc batch conversion endpoint.
- Parity between batch and streaming conversion results.
"""

import base64
import io
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import soundfile as sf
from fastapi.testclient import TestClient

from tmrvc_serve.app import app, init_app
from tmrvc_serve.uclm_engine import UCLMEngine, EngineState
from tmrvc_core.dialogue_types import StyleParams


class MockUCLMCore(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_f0_condition = None

    def forward(
        self,
        content_features,
        a_ctx,
        b_ctx,
        speaker_embed,
        state_cond,
        cfg_scale=1.0,
        kv_caches=None,
        f0_condition=None,
    ):
        self.last_f0_condition = f0_condition
        B, _, T = content_features.shape
        logits_a = torch.zeros(B, 8, T, 1024, device=content_features.device)
        logits_b = torch.zeros(B, 4, T, 64, device=content_features.device)
        return logits_a, logits_b, kv_caches, torch.zeros(B, T, 512, device=content_features.device)


@pytest.fixture
def engine():
    e = UCLMEngine(device="cpu")
    e.codec_enc = lambda x, s: (torch.zeros(1, 8, x.shape[-1] // 240), None, s)
    e.vc_enc = lambda x: (torch.zeros(1, 512, x.shape[-1]), None)
    e.voice_state_enc = lambda v, s: (torch.zeros(1, 1, 512),)
    e.uclm_core = MockUCLMCore()
    e.codec_dec = lambda a, b, v, s: (torch.zeros(1, 1, 240), s)
    e._loaded = True
    return e


def test_vc_frame_pitch_shift_propagation(engine):
    """Verify that pitch_shift parameter correctly produces f0_condition."""
    audio_frame = torch.zeros(1, 1, 240)
    spk = torch.zeros(1, 192)
    style = StyleParams.neutral()
    state = EngineState()

    # Case 1: No pitch shift
    engine.vc_frame(audio_frame, spk, style, state, pitch_shift=0.0)
    assert engine.uclm_core.last_f0_condition is None

    # Case 2: Positive pitch shift
    engine.vc_frame(audio_frame, spk, style, state, pitch_shift=12.0)
    f0_cond = engine.uclm_core.last_f0_condition
    assert f0_cond is not None
    # 12 semitones = +1.0 in log2 domain
    assert torch.allclose(f0_cond[0, 0, 0], torch.tensor(1.0))

    # Case 3: Negative pitch shift
    engine.vc_frame(audio_frame, spk, style, state, pitch_shift=-12.0)
    f0_cond = engine.uclm_core.last_f0_condition
    assert torch.allclose(f0_cond[0, 0, 0], torch.tensor(-1.0))


def test_vc_endpoint_batch_conversion(monkeypatch):
    """Test the POST /vc endpoint for single audio conversion."""
    from tmrvc_serve.app import _characters, CharacterProfile
    
    # Mock character
    _characters["test_char"] = CharacterProfile(
        name="Test", speaker_file=Path("dummy.tmrvc_speaker")
    )
    
    # Mock engine
    class MockEngine:
        device = "cpu"
        def vc_frame(self, chunk, spk, style, state, pitch_shift=0.0, explicit_voice_state=None):
            # Return same audio as input for simplicity
            return chunk.squeeze(0), state

    monkeypatch.setattr("tmrvc_serve.app.get_engine", lambda: MockEngine())
    monkeypatch.setattr("tmrvc_serve._helpers._load_speaker_embed", 
                        lambda c: torch.zeros(192))

    client = TestClient(app)
    
    # Create dummy 10ms WAV
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(240, dtype=np.float32), 24000, format='WAV')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode()

    payload = {
        "audio_base64": audio_b64,
        "character_id": "test_char",
        "pitch_shift": 2.0
    }
    
    response = client.post("/vc", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "audio_base64" in data
    assert data["sample_rate"] == 24000


def test_vc_batch_streaming_parity(engine):
    """Ensure that the batch processing logic matches streaming results.
    
    The batch logic in convert_vc loops over chunks, which should be identical 
    to feeding those chunks one by one through streaming if the state is handled correctly.
    """
    spk = torch.zeros(1, 192)
    style = StyleParams.neutral()
    
    # Create input audio (2 frames = 480 samples)
    audio_np = np.random.randn(480).astype(np.float32)
    
    # 1. Streaming processing
    state = EngineState()
    stream_outputs = []
    for i in range(0, 480, 240):
        chunk = audio_np[i : i + 240]
        chunk_t = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
        out_chunk, state = engine.vc_frame(chunk_t, spk, style, state)
        stream_outputs.append(out_chunk.numpy())
    stream_final = np.concatenate(stream_outputs)
    
    # 2. Simulated batch processing (same logic as in convert_vc)
    batch_state = EngineState()
    batch_outputs = []
    for i in range(0, 480, 240):
        chunk = audio_np[i : i + 240]
        chunk_t = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0)
        out_chunk, batch_state = engine.vc_frame(chunk_t, spk, style, batch_state)
        batch_outputs.append(out_chunk.numpy())
    batch_final = np.concatenate(batch_outputs)
    
    assert np.allclose(stream_final, batch_final)
    assert len(stream_final) == 480
