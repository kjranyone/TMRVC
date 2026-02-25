"""Tests for VoiceSourceStatsTracker and compute_group_preset."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tmrvc_core.constants import N_ACOUSTIC_PARAMS, N_IR_PARAMS, N_VOICE_SOURCE_PARAMS
from tmrvc_train.voice_source_stats import VoiceSourceStatsTracker, compute_group_preset


@pytest.fixture
def tracker():
    return VoiceSourceStatsTracker()


class TestUpdate:
    def test_single_update(self, tracker):
        params = torch.zeros(1, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        tracker.update(params, ["spk001"])
        mean = tracker.get_speaker_mean("spk001")
        assert mean is not None
        np.testing.assert_array_almost_equal(
            mean, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )

    def test_running_mean(self, tracker):
        p1 = torch.zeros(1, N_ACOUSTIC_PARAMS)
        p1[0, N_IR_PARAMS:] = 2.0
        p2 = torch.zeros(1, N_ACOUSTIC_PARAMS)
        p2[0, N_IR_PARAMS:] = 4.0
        tracker.update(p1, ["spk001"])
        tracker.update(p2, ["spk001"])
        mean = tracker.get_speaker_mean("spk001")
        np.testing.assert_array_almost_equal(mean, np.full(N_VOICE_SOURCE_PARAMS, 3.0))

    def test_batch_update(self, tracker):
        params = torch.zeros(3, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = 1.0
        params[1, N_IR_PARAMS:] = 3.0
        params[2, N_IR_PARAMS:] = 5.0
        tracker.update(params, ["a", "a", "b"])
        mean_a = tracker.get_speaker_mean("a")
        mean_b = tracker.get_speaker_mean("b")
        np.testing.assert_array_almost_equal(mean_a, np.full(N_VOICE_SOURCE_PARAMS, 2.0))
        np.testing.assert_array_almost_equal(mean_b, np.full(N_VOICE_SOURCE_PARAMS, 5.0))

    def test_unknown_speaker(self, tracker):
        assert tracker.get_speaker_mean("nobody") is None


class TestGetAllMeans:
    def test_empty(self, tracker):
        assert tracker.get_all_means() == {}

    def test_multiple_speakers(self, tracker):
        params = torch.zeros(2, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = 1.0
        params[1, N_IR_PARAMS:] = 2.0
        tracker.update(params, ["a", "b"])
        means = tracker.get_all_means()
        assert "a" in means
        assert "b" in means
        assert len(means["a"]) == N_VOICE_SOURCE_PARAMS


class TestSaveLoad:
    def test_roundtrip(self, tmp_path, tracker):
        params = torch.zeros(2, N_ACOUSTIC_PARAMS)
        params[0, N_IR_PARAMS:] = 1.5
        params[1, N_IR_PARAMS:] = 2.5
        tracker.update(params, ["spk001", "spk002"])

        path = tmp_path / "stats.json"
        tracker.save(path)
        assert path.exists()

        loaded = VoiceSourceStatsTracker.load(path)
        np.testing.assert_array_almost_equal(
            loaded.get_speaker_mean("spk001"),
            tracker.get_speaker_mean("spk001"),
        )
        np.testing.assert_array_almost_equal(
            loaded.get_speaker_mean("spk002"),
            tracker.get_speaker_mean("spk002"),
        )

    def test_json_readable(self, tmp_path, tracker):
        params = torch.ones(1, N_ACOUSTIC_PARAMS)
        tracker.update(params, ["test"])
        path = tmp_path / "stats.json"
        tracker.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "param_names" in data
        assert "speakers" in data
        assert "test" in data["speakers"]


class TestComputeGroupPreset:
    @pytest.fixture
    def stats_path(self, tmp_path):
        tracker = VoiceSourceStatsTracker()
        for i in range(5):
            params = torch.zeros(1, N_ACOUSTIC_PARAMS)
            params[0, N_IR_PARAMS:] = float(i)
            tracker.update(params, [f"spk{i:03d}"])
        path = tmp_path / "stats.json"
        tracker.save(path)
        return path

    def test_single_pattern(self, stats_path):
        result = compute_group_preset(stats_path, ["spk001"])
        assert len(result["preset"]) == N_VOICE_SOURCE_PARAMS
        assert result["n_speakers"] == 1
        np.testing.assert_array_almost_equal(
            result["preset"], np.full(N_VOICE_SOURCE_PARAMS, 1.0),
        )

    def test_wildcard_pattern(self, stats_path):
        result = compute_group_preset(stats_path, ["spk*"])
        assert result["n_speakers"] == 5
        # Average of 0,1,2,3,4 = 2.0
        np.testing.assert_array_almost_equal(
            result["preset"], np.full(N_VOICE_SOURCE_PARAMS, 2.0),
        )

    def test_no_match_raises(self, stats_path):
        with pytest.raises(ValueError, match="No speakers matched"):
            compute_group_preset(stats_path, ["nonexistent*"])

    def test_save_output(self, stats_path, tmp_path):
        out_path = tmp_path / "preset.json"
        result = compute_group_preset(stats_path, ["spk00*"], output_path=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text(encoding="utf-8"))
        assert data["n_speakers"] == result["n_speakers"]
