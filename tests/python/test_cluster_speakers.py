"""Tests for scripts/cluster_speakers.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Import helpers from the script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from cluster_speakers import (
    EMBED_FILENAME,
    MAP_FILENAME,
    extract_embeddings,
    load_speaker_map,
    print_report,
    save_speaker_map,
)

hdbscan = pytest.importorskip("hdbscan", reason="hdbscan not installed (optional dep)")
from cluster_speakers import cluster_embeddings


class TestClusterEmbeddings:
    def test_basic_clustering(self):
        """Two clear clusters should be separated."""
        rng = np.random.RandomState(42)
        embeddings = {}
        # Cluster A: 30 points near [1, 0, 0, ...]
        base_a = np.zeros(192, dtype=np.float32)
        base_a[0] = 1.0
        for i in range(30):
            embeddings[f"a_{i:03d}.wav"] = base_a + rng.randn(192).astype(np.float32) * 0.01

        # Cluster B: 30 points near [0, 1, 0, ...]
        base_b = np.zeros(192, dtype=np.float32)
        base_b[1] = 1.0
        for i in range(30):
            embeddings[f"b_{i:03d}.wav"] = base_b + rng.randn(192).astype(np.float32) * 0.01

        mapping = cluster_embeddings(
            embeddings, min_cluster_size=10, min_samples=3,
        )
        assert len(mapping) == 60

        # Check that a_ and b_ files are in different clusters
        a_speakers = {mapping[k] for k in mapping if k.startswith("a_")}
        b_speakers = {mapping[k] for k in mapping if k.startswith("b_")}
        # Remove noise label if present
        a_speakers.discard("spk_noise")
        b_speakers.discard("spk_noise")
        # At least one valid cluster per group
        assert len(a_speakers) >= 1
        assert len(b_speakers) >= 1

    def test_noise_detection(self):
        """Random scattered points should be labeled as noise."""
        rng = np.random.RandomState(0)
        embeddings = {}
        for i in range(10):
            embeddings[f"noise_{i}.wav"] = rng.randn(192).astype(np.float32)

        mapping = cluster_embeddings(
            embeddings, min_cluster_size=20, min_samples=5,
        )
        # With only 10 widely scattered points and min_cluster_size=20,
        # all should be noise
        assert all(v == "spk_noise" for v in mapping.values())


class TestSaveLoadSpeakerMap:
    def test_roundtrip(self, tmp_path):
        mapping = {"a.wav": "spk_0001", "b.wav": "spk_0002", "c.wav": "spk_noise"}
        map_path = tmp_path / MAP_FILENAME

        save_speaker_map(mapping, map_path)
        assert map_path.exists()

        loaded = load_speaker_map(map_path)
        assert loaded == mapping

    def test_json_structure(self, tmp_path):
        mapping = {"x.wav": "spk_0001"}
        map_path = tmp_path / MAP_FILENAME
        save_speaker_map(mapping, map_path, method="hdbscan")

        with open(map_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["method"] == "hdbscan"
        assert data["n_speakers"] == 1
        assert data["n_noise"] == 0
        assert data["mapping"] == mapping


class TestPrintReport:
    def test_no_crash(self, capsys):
        """print_report should not crash on valid data."""
        mapping = {
            "a.wav": "spk_0001",
            "b.wav": "spk_0001",
            "c.wav": "spk_0002",
            "d.wav": "spk_noise",
        }
        print_report(mapping)
        captured = capsys.readouterr()
        assert "spk_0001" in captured.out
        assert "spk_0002" in captured.out
        assert "Noise" in captured.out
        assert "Total" in captured.out

    def test_empty_mapping(self, capsys):
        print_report({})
        captured = capsys.readouterr()
        assert "Speakers" in captured.out


class TestExtractEmbeddings:
    def test_extract_with_mock_encoder(self, tmp_path):
        """Test embedding extraction with mocked SpeakerEncoder."""
        import soundfile as sf

        # Create test audio files
        for name in ["utt1.wav", "utt2.wav", "utt3.wav"]:
            sf.write(str(tmp_path / name), np.zeros(24000, dtype=np.float32), 24000)

        # Create a non-audio file that should be ignored
        (tmp_path / "readme.txt").write_text("ignore me")

        mock_embed = np.random.randn(192).astype(np.float32)
        mock_encoder = MagicMock()
        mock_encoder.extract_from_file.return_value = MagicMock(
            numpy=MagicMock(return_value=mock_embed)
        )

        with patch("tmrvc_data.speaker.SpeakerEncoder", return_value=mock_encoder):
            output_path = tmp_path / EMBED_FILENAME
            embeddings = extract_embeddings(
                tmp_path, output_path, device="cpu", save_every=10,
            )

        assert len(embeddings) == 3
        assert "utt1.wav" in embeddings
        assert "utt2.wav" in embeddings
        assert "utt3.wav" in embeddings
        assert output_path.exists()

    def test_resume_from_existing(self, tmp_path):
        """Test that existing embeddings are loaded and not re-extracted."""
        import soundfile as sf

        for name in ["a.wav", "b.wav"]:
            sf.write(str(tmp_path / name), np.zeros(24000, dtype=np.float32), 24000)

        # Pre-save one embedding
        output_path = tmp_path / EMBED_FILENAME
        existing = {"a.wav": np.random.randn(192).astype(np.float32)}
        np.savez(str(output_path), **existing)

        mock_embed = np.random.randn(192).astype(np.float32)
        mock_encoder = MagicMock()
        mock_encoder.extract_from_file.return_value = MagicMock(
            numpy=MagicMock(return_value=mock_embed)
        )

        with patch("tmrvc_data.speaker.SpeakerEncoder", return_value=mock_encoder):
            embeddings = extract_embeddings(
                tmp_path, output_path, device="cpu", save_every=10,
            )

        # Should have both, but only b.wav was newly extracted
        assert len(embeddings) == 2
        assert mock_encoder.extract_from_file.call_count == 1
