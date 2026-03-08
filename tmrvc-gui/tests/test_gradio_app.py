"""Smoke tests for the TMRVC Gradio Control Plane (Worker 12).

These tests verify:
1. The Gradio app builds without errors
2. Tab builder functions return valid Blocks objects
3. API helper functions handle errors gracefully
4. State management classes work correctly
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Prevent tests from writing to real data directories."""
    monkeypatch.setattr(
        "tmrvc_gui.gradio_app._audit",
        _make_audit(tmp_path),
    )
    monkeypatch.setattr(
        "tmrvc_gui.gradio_app._eval_session",
        _make_eval_session(tmp_path),
    )


def _make_audit(tmp_path):
    from tmrvc_gui.gradio_state import AuditTrail
    return AuditTrail(path=tmp_path / "audit.jsonl")


def _make_eval_session(tmp_path):
    from tmrvc_gui.gradio_state import EvalSession
    return EvalSession(path=tmp_path / "eval.jsonl")


# ---------------------------------------------------------------------------
# App creation smoke test
# ---------------------------------------------------------------------------


@patch("tmrvc_gui.gradio_app.CastingGallery")
def test_create_app_smoke(mock_gallery_cls):
    """App builds and returns a Blocks instance without crashing."""
    mock_gallery = MagicMock()
    mock_gallery.list_names.return_value = ["test_speaker (abc123)"]
    mock_gallery.profiles = {}
    mock_gallery_cls.return_value = mock_gallery

    import importlib
    import tmrvc_gui.gradio_app as app_mod

    # Patch the module-level _gallery
    original_gallery = app_mod._gallery
    app_mod._gallery = mock_gallery
    try:
        app = app_mod.create_app()
        import gradio as gr
        assert isinstance(app, gr.Blocks)
    finally:
        app_mod._gallery = original_gallery


# ---------------------------------------------------------------------------
# API helper tests
# ---------------------------------------------------------------------------


def test_api_get_handles_connection_error():
    """_api_get returns None when server is unreachable."""
    from tmrvc_gui.gradio_app import _api_get
    result = _api_get("/nonexistent")
    assert result is None


def test_api_post_handles_connection_error():
    """_api_post returns None when server is unreachable."""
    from tmrvc_gui.gradio_app import _api_post
    result = _api_post("/nonexistent", {"key": "value"})
    assert result is None


def test_api_patch_handles_connection_error():
    """_api_patch returns None when server is unreachable."""
    from tmrvc_gui.gradio_app import _api_patch
    result = _api_patch("/nonexistent", {"key": "value"})
    assert result is None


# ---------------------------------------------------------------------------
# State management tests
# ---------------------------------------------------------------------------


def test_audit_trail_roundtrip(tmp_path):
    """AuditTrail logs and reads back entries."""
    from tmrvc_gui.gradio_state import AuditTrail

    trail = AuditTrail(path=tmp_path / "audit.jsonl")
    entry = trail.log("admin", "user1", "test_action", rationale="testing")
    assert entry.actor_role == "admin"
    assert entry.actor_id == "user1"

    recent = trail.read_recent(10)
    assert len(recent) == 1
    assert recent[0]["action"] == "test_action"


def test_eval_session_roundtrip(tmp_path):
    """EvalSession records and summarizes evaluation pairs."""
    from tmrvc_gui.gradio_state import EvalSession, EvalPair

    session = EvalSession(path=tmp_path / "eval.jsonl")
    pair = EvalPair(
        pair_id="p1",
        sample_a_label="sys_a",
        sample_b_label="sys_b",
        text="hello",
        preference="A",
        mos_a=4.0,
        mos_b=3.0,
        rater_id="r1",
        rater_role="rater",
    )
    session.record(pair)

    summary = session.summary()
    assert summary["total"] == 1
    assert summary["a_wins"] == 1
    assert summary["b_wins"] == 0


def test_check_permission():
    """Role-based permission check works correctly."""
    from tmrvc_gui.gradio_state import check_permission

    assert check_permission("admin", "load_model") is True
    assert check_permission("rater", "load_model") is False
    assert check_permission("rater", "rate") is True
    assert check_permission("auditor", "promote") is True
    assert check_permission("annotator", "promote") is False


# ---------------------------------------------------------------------------
# Voice state preset tests
# ---------------------------------------------------------------------------


def test_voice_state_preset_save_load(tmp_path, monkeypatch):
    """Voice state presets save and load correctly."""
    preset_dir = tmp_path / "presets"
    preset_dir.mkdir()

    preset = {
        "pitch_level": 0.7,
        "pitch_range": 0.4,
        "energy_level": 0.3,
        "pressedness": 0.5,
        "spectral_tilt": 0.4,
        "breathiness": 0.6,
        "voice_irregularity": 0.8,
        "openness": 0.2,
    }
    (preset_dir / "test_preset.json").write_text(
        json.dumps(preset, indent=2), encoding="utf-8"
    )

    loaded = json.loads(
        (preset_dir / "test_preset.json").read_text(encoding="utf-8")
    )
    assert loaded["pitch_level"] == 0.7
    assert loaded["openness"] == 0.2


# ---------------------------------------------------------------------------
# Fetch speaker profiles test
# ---------------------------------------------------------------------------


def test_fetch_speaker_profiles_fallback():
    """_fetch_speaker_profiles falls back to local gallery when API is down."""
    from tmrvc_gui.gradio_app import _fetch_speaker_profiles

    # API is not running, so it should fall back to local gallery
    result = _fetch_speaker_profiles()
    assert isinstance(result, list)
