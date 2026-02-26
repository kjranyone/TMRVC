"""Tests for scripts/inject_whisper_transcripts.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from inject_whisper_transcripts import (
    LANGUAGE_IDS,
    inject_to_cache,
    load_transcripts_dir,
)


class TestLoadTranscriptsDir:
    def test_load_single_file(self, tmp_path):
        spk_dir = tmp_path / "spk_0001"
        spk_dir.mkdir()
        tf = spk_dir / "transcripts.txt"
        tf.write_text("utt001|こんにちは\nutt002|さようなら\n", encoding="utf-8")

        result = load_transcripts_dir(tmp_path)
        assert result == {"utt001": "こんにちは", "utt002": "さようなら"}

    def test_load_multiple_speakers(self, tmp_path):
        for spk in ["spk_0001", "spk_0002"]:
            d = tmp_path / spk
            d.mkdir()
            (d / "transcripts.txt").write_text(
                f"{spk}_utt|テスト{spk}\n", encoding="utf-8"
            )

        result = load_transcripts_dir(tmp_path)
        assert len(result) == 2

    def test_skip_empty_lines(self, tmp_path):
        d = tmp_path / "spk"
        d.mkdir()
        (d / "transcripts.txt").write_text(
            "utt1|hello\n\n\nutt2|world\n", encoding="utf-8"
        )
        result = load_transcripts_dir(tmp_path)
        assert len(result) == 2

    def test_skip_malformed_lines(self, tmp_path):
        d = tmp_path / "spk"
        d.mkdir()
        (d / "transcripts.txt").write_text(
            "utt1|good\nno_pipe_here\nutt2|also_good\n", encoding="utf-8"
        )
        result = load_transcripts_dir(tmp_path)
        assert len(result) == 2
        assert "no_pipe_here" not in result

    def test_no_transcripts_returns_empty(self, tmp_path):
        result = load_transcripts_dir(tmp_path)
        assert result == {}

    def test_pipe_in_text(self, tmp_path):
        """Text containing | should keep everything after first pipe."""
        d = tmp_path / "spk"
        d.mkdir()
        (d / "transcripts.txt").write_text(
            "utt1|hello|world\n", encoding="utf-8"
        )
        result = load_transcripts_dir(tmp_path)
        assert result["utt1"] == "hello|world"


class TestInjectToCache:
    def _make_cache(self, cache_dir: Path, dataset: str, split: str, utts: list[dict]) -> None:
        """Helper: create cache meta.json files."""
        for utt in utts:
            utt_dir = cache_dir / dataset / split / utt["speaker_id"] / utt["utterance_id"]
            utt_dir.mkdir(parents=True, exist_ok=True)
            meta = {"utterance_id": utt["utterance_id"], "speaker_id": utt["speaker_id"]}
            (utt_dir / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8"
            )

    def test_inject_exact_match(self, tmp_path):
        cache_dir = tmp_path / "cache"
        self._make_cache(cache_dir, "galge", "train", [
            {"speaker_id": "galge_spk_0001", "utterance_id": "utt001"},
            {"speaker_id": "galge_spk_0001", "utterance_id": "utt002"},
        ])

        transcripts = {"utt001": "こんにちは", "utt002": "おはよう"}
        count = inject_to_cache(cache_dir, "galge", "train", transcripts, language_id=0)

        assert count == 2

        meta1 = json.loads(
            (cache_dir / "galge/train/galge_spk_0001/utt001/meta.json")
            .read_text(encoding="utf-8")
        )
        assert meta1["text"] == "こんにちは"
        assert meta1["language_id"] == 0

    def test_inject_suffix_match(self, tmp_path):
        """GenericAdapter prefixes IDs — test suffix matching."""
        cache_dir = tmp_path / "cache"
        self._make_cache(cache_dir, "galge", "train", [
            {"speaker_id": "galge_spk_0001", "utterance_id": "galge_ev001"},
        ])

        transcripts = {"ev001": "テスト"}
        count = inject_to_cache(cache_dir, "galge", "train", transcripts, language_id=0)
        assert count == 1

    def test_inject_no_match(self, tmp_path):
        cache_dir = tmp_path / "cache"
        self._make_cache(cache_dir, "galge", "train", [
            {"speaker_id": "galge_spk_0001", "utterance_id": "utt001"},
        ])

        transcripts = {"nonexistent": "never"}
        count = inject_to_cache(cache_dir, "galge", "train", transcripts, language_id=1)
        assert count == 0

    def test_inject_missing_cache_dir(self, tmp_path):
        count = inject_to_cache(tmp_path / "nope", "galge", "train", {"x": "y"}, 0)
        assert count == 0

    def test_language_ids(self):
        assert LANGUAGE_IDS["ja"] == 0
        assert LANGUAGE_IDS["en"] == 1


class TestPrepBulkVoiceSpeakerMap:
    """Test scan_audio_files_with_map from prepare_bulk_voice.py."""

    def test_scan_with_map(self, tmp_path):
        import soundfile as sf
        import numpy as np

        from prepare_bulk_voice import scan_audio_files_with_map

        # Create flat audio files
        for name in ["a.wav", "b.wav", "noise.wav"]:
            sf.write(str(tmp_path / name), np.zeros(2400, dtype=np.float32), 24000)

        speaker_map = {
            "version": 1, "method": "hdbscan",
            "n_speakers": 2, "n_noise": 1,
            "mapping": {
                "a.wav": "spk_0001",
                "b.wav": "spk_0002",
                "noise.wav": "spk_noise",
            },
        }
        map_path = tmp_path / "_speaker_map.json"
        map_path.write_text(json.dumps(speaker_map), encoding="utf-8")

        result = scan_audio_files_with_map(tmp_path, map_path)
        assert "spk_0001" in result
        assert "spk_0002" in result
        assert "spk_noise" not in result
        assert len(result["spk_0001"]) == 1
        assert result["spk_0001"][0].name == "a.wav"

    def test_scan_missing_files(self, tmp_path):
        from prepare_bulk_voice import scan_audio_files_with_map

        speaker_map = {
            "version": 1, "method": "hdbscan",
            "n_speakers": 1, "n_noise": 0,
            "mapping": {"missing.wav": "spk_0001"},
        }
        map_path = tmp_path / "_speaker_map.json"
        map_path.write_text(json.dumps(speaker_map), encoding="utf-8")

        result = scan_audio_files_with_map(tmp_path, map_path)
        assert len(result) == 0
