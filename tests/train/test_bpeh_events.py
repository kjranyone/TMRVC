"""Tests for BPEH event extraction and cache I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tmrvc_data.events import (
    _find_contiguous_regions,
    events_to_tensors,
    extract_events,
    load_events,
    save_events,
)


class TestFindContiguousRegions:
    def test_no_regions(self):
        assert _find_contiguous_regions(np.array([False, False, False])) == []

    def test_single_region(self):
        assert _find_contiguous_regions(np.array([False, True, True, False])) == [(1, 3)]

    def test_multiple_regions(self):
        mask = np.array([True, True, False, True, False, False, True, True, True])
        assert _find_contiguous_regions(mask) == [(0, 2), (3, 4), (6, 9)]

    def test_trailing_region(self):
        assert _find_contiguous_regions(np.array([False, True, True])) == [(1, 3)]

    def test_all_true(self):
        assert _find_contiguous_regions(np.array([True, True, True])) == [(0, 3)]


class TestExtractEvents:
    def test_silent_audio_yields_pauses(self):
        T = 100
        # Very low energy (silence) + no F0
        mel = np.full((80, T), -60.0, dtype=np.float32)
        f0 = np.zeros((1, T), dtype=np.float32)
        events = extract_events(mel, f0)
        # Should detect as pause (low energy, unvoiced)
        assert any(e["type"] == "pause" for e in events)

    def test_voiced_audio_no_breath(self):
        T = 50
        mel = np.random.randn(80, T).astype(np.float32)
        f0 = np.full((1, T), 200.0, dtype=np.float32)  # All voiced
        events = extract_events(mel, f0)
        # No breaths or pauses — all voiced
        assert len(events) == 0

    def test_breath_detection(self):
        T = 100
        mel = np.full((80, T), -60.0, dtype=np.float32)
        f0 = np.full((1, T), 200.0, dtype=np.float32)

        # Create a breathy segment (unvoiced, mid-band energy)
        f0[0, 20:35] = 0.0
        mel[13:40, 20:35] = -20.0  # Above breath threshold

        events = extract_events(mel, f0, breath_threshold_db=-40.0)
        breath_events = [e for e in events if e["type"] == "breath"]
        assert len(breath_events) >= 1
        assert breath_events[0]["start_frame"] == 20
        assert breath_events[0]["intensity"] > 0

    def test_pause_detection(self):
        T = 100
        mel = np.full((80, T), -10.0, dtype=np.float32)
        f0 = np.full((1, T), 200.0, dtype=np.float32)

        # Silent pause (unvoiced, very low energy)
        f0[0, 40:60] = 0.0
        mel[:, 40:60] = -80.0

        events = extract_events(mel, f0, breath_threshold_db=-40.0)
        pause_events = [e for e in events if e["type"] == "pause"]
        assert len(pause_events) >= 1

    def test_min_duration_filter(self):
        T = 50
        mel = np.full((80, T), -60.0, dtype=np.float32)
        f0 = np.full((1, T), 200.0, dtype=np.float32)
        # Very short unvoiced segment (1 frame = 10ms)
        f0[0, 10] = 0.0
        mel[:, 10] = -80.0

        events = extract_events(mel, f0, min_pause_ms=50.0)
        # Too short — should be filtered out
        assert len(events) == 0

    def test_f0_1d_input(self):
        T = 50
        mel = np.random.randn(80, T).astype(np.float32)
        f0 = np.full(T, 200.0, dtype=np.float32)  # 1D
        events = extract_events(mel, f0)
        assert isinstance(events, list)


class TestEventsToTensors:
    def test_empty_events(self):
        tensors = events_to_tensors([], 50)
        assert tensors["breath_onsets"].shape == (50,)
        assert tensors["breath_onsets"].sum() == 0
        assert tensors["pause_durations"].sum() == 0

    def test_breath_event(self):
        events = [{"type": "breath", "start_frame": 5, "dur_ms": 200, "intensity": 0.8}]
        tensors = events_to_tensors(events, 50)
        assert tensors["breath_onsets"][5] == 1.0
        assert tensors["breath_durations"][5] == 200.0
        assert tensors["breath_intensity"][5] == 0.8
        assert tensors["breath_onsets"][0] == 0.0

    def test_pause_event(self):
        events = [{"type": "pause", "start_frame": 10, "dur_ms": 150}]
        tensors = events_to_tensors(events, 50)
        assert tensors["pause_durations"][10] == 150.0
        assert tensors["breath_onsets"][10] == 0.0

    def test_out_of_bounds_ignored(self):
        events = [{"type": "breath", "start_frame": 100, "dur_ms": 200, "intensity": 0.5}]
        tensors = events_to_tensors(events, 50)
        assert tensors["breath_onsets"].sum() == 0


class TestSaveLoadEvents:
    def test_roundtrip(self, tmp_path):
        events = [
            {"type": "breath", "start_frame": 5, "dur_ms": 300, "intensity": 0.7},
            {"type": "pause", "start_frame": 40, "dur_ms": 150},
        ]
        save_events(events, tmp_path)
        loaded = load_events(tmp_path)
        assert len(loaded) == 2
        assert loaded[0]["type"] == "breath"
        assert loaded[1]["dur_ms"] == 150

    def test_load_missing_returns_empty(self, tmp_path):
        loaded = load_events(tmp_path / "nonexistent")
        assert loaded == []

    def test_json_format(self, tmp_path):
        events = [{"type": "breath", "start_frame": 0, "dur_ms": 100, "intensity": 0.5}]
        save_events(events, tmp_path)
        with open(tmp_path / "events.json", encoding="utf-8") as f:
            data = json.load(f)
        assert "events" in data
        assert len(data["events"]) == 1


class TestBuildBreathEventCacheCLI:
    def test_import(self):
        from scripts.build_breath_event_cache import main
        assert callable(main)
