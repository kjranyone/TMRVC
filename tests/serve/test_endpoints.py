"""Integration tests for tmrvc-serve endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import torch
import numpy as np

from tmrvc_serve.app import app, init_app, _characters
from tmrvc_core.dialogue_types import CharacterProfile

@pytest.fixture
def client(tmp_path):
    # Mock models to avoid heavy weight loading
    uclm_path = tmp_path / "uclm.pt"
    codec_path = tmp_path / "codec.pt"
    
    # Create minimal mock state dicts
    torch.save({"model": {}}, uclm_path)
    torch.save({"model": {}}, codec_path)
    
    # Initialize app with mocks
    init_app(uclm_checkpoint=uclm_path, codec_checkpoint=codec_path, device="cpu")
    
    # Add a mock character
    _characters["test_char"] = CharacterProfile(
        name="Test",
        language="ja"
    )
    
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """Verify health check returns 200."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

def test_tts_import_integrity(client, monkeypatch):
    """
    Test that /tts endpoint doesn't crash due to ImportErrors.
    We mock the engine.tts call to only test the route logic and imports.
    """
    from tmrvc_serve.uclm_engine import UCLMEngine
    
    def mock_tts(self, phonemes, speaker_profile=None, speaker_embed=None, style=None, cfg_scale=1.5, temperature=0.8, language_id=0, pace=1.0, hold_bias=0.0, boundary_bias=0.0, max_frames=1500, dialogue_context=None, acting_intent=None, phrase_pressure=0.0, breath_tendency=0.0):
        # Return a dummy 1-second audio tensor and dummy metrics
        return torch.zeros(24000), {"rtf": 0.1, "gen_time_ms": 100.0}
    
    monkeypatch.setattr(UCLMEngine, "tts", mock_tts)
    
    payload = {
        "text": "こんにちは",
        "character_id": "test_char",
        "emotion": "happy"
    }
    
    # This will trigger the generate_tts function in routes/tts.py
    # and catch the ImportError we just fixed.
    response = client.post("/tts", json=payload)
    
    # If imports are correct, it should at least try to run and either 
    # succeed (200) or fail with a known logic error (not a 500 ImportError).
    assert response.status_code == 200
    data = response.json()
    assert "audio_base64" in data
    assert data["style_used"]["emotion"] == "happy"

def test_character_list(client):
    """Verify characters endpoint works."""
    response = client.get("/characters")
    assert response.status_code == 200
    data = response.json()
    assert any(c["id"] == "test_char" for c in data)
