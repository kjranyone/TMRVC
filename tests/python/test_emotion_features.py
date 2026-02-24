"""Tests for emotion dataset parsers."""

import pytest
from pathlib import Path

from tmrvc_core.dialogue_types import EMOTION_CATEGORIES
from tmrvc_data.emotion_features import (
    EXPRESSO_STYLE_MAP,
    JVNV_EMOTION_MAP,
    EMOV_EMOTION_MAP,
    RAVDESS_EMOTION_MAP,
    RAVDESS_VAD,
    EmotionEntry,
    parse_dataset,
)


class TestEmotionEntry:
    def test_valid_entry(self):
        entry = EmotionEntry(
            utterance_id="test_001",
            speaker_id="spk1",
            audio_path=Path("test.wav"),
            text="hello",
            emotion="happy",
        )
        assert entry.validate()

    def test_invalid_emotion(self):
        entry = EmotionEntry(
            utterance_id="test_001",
            speaker_id="spk1",
            audio_path=Path("test.wav"),
            text="hello",
            emotion="nonexistent",
        )
        assert not entry.validate()


class TestExpressoMapping:
    def test_all_mapped_emotions_valid(self):
        for style, emotion in EXPRESSO_STYLE_MAP.items():
            assert emotion in EMOTION_CATEGORIES, f"Invalid: {style} -> {emotion}"

    def test_key_styles(self):
        assert EXPRESSO_STYLE_MAP["default"] == "neutral"
        assert EXPRESSO_STYLE_MAP["happy"] == "happy"
        assert EXPRESSO_STYLE_MAP["angry"] == "angry"
        assert EXPRESSO_STYLE_MAP["whisper"] == "whisper"


class TestJVNVMapping:
    def test_all_mapped_emotions_valid(self):
        for jvnv_em, emotion in JVNV_EMOTION_MAP.items():
            assert emotion in EMOTION_CATEGORIES, f"Invalid: {jvnv_em} -> {emotion}"

    def test_six_emotions(self):
        assert len(JVNV_EMOTION_MAP) == 6

    def test_key_mappings(self):
        assert JVNV_EMOTION_MAP["anger"] == "angry"
        assert JVNV_EMOTION_MAP["happiness"] == "happy"
        assert JVNV_EMOTION_MAP["sadness"] == "sad"


class TestEmoVMapping:
    def test_all_mapped_emotions_valid(self):
        for emov_em, emotion in EMOV_EMOTION_MAP.items():
            assert emotion in EMOTION_CATEGORIES

    def test_five_emotions(self):
        assert len(EMOV_EMOTION_MAP) == 5


class TestRAVDESSMapping:
    def test_all_mapped_emotions_valid(self):
        for code, emotion in RAVDESS_EMOTION_MAP.items():
            assert emotion in EMOTION_CATEGORIES

    def test_eight_emotions(self):
        assert len(RAVDESS_EMOTION_MAP) == 8

    def test_vad_values_in_range(self):
        for code, (v, a, d) in RAVDESS_VAD.items():
            assert -1.0 <= v <= 1.0
            assert -1.0 <= a <= 1.0
            assert -1.0 <= d <= 1.0


class TestParseDataset:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            parse_dataset("nonexistent", "/tmp")

    def test_empty_dir_returns_empty(self, tmp_path):
        entries = parse_dataset("expresso", tmp_path)
        assert entries == []

    def test_ravdess_empty_dir(self, tmp_path):
        entries = parse_dataset("ravdess", tmp_path)
        assert entries == []

    def test_ravdess_with_file(self, tmp_path):
        # Create a RAVDESS-format filename
        actor_dir = tmp_path / "Actor_01"
        actor_dir.mkdir()
        wav = actor_dir / "03-01-03-01-01-01-01.wav"
        wav.write_bytes(b"\x00" * 100)  # dummy wav

        entries = parse_dataset("ravdess", tmp_path)
        assert len(entries) == 1
        assert entries[0].emotion == "happy"  # emotion code 03
        assert entries[0].speaker_id == "Actor_01"

    def test_expresso_with_file(self, tmp_path):
        wav = tmp_path / "ex01_happy_00001.wav"
        wav.write_bytes(b"\x00" * 100)

        entries = parse_dataset("expresso", tmp_path)
        assert len(entries) == 1
        assert entries[0].emotion == "happy"
        assert entries[0].speaker_id == "ex01"
