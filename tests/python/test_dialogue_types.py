"""Tests for dialogue types and style parameters."""

import pytest

from tmrvc_core.dialogue_types import (
    EMOTION_CATEGORIES,
    EMOTION_TO_ID,
    CharacterProfile,
    DialogueTurn,
    Script,
    ScriptEntry,
    StyleParams,
)


class TestEmotionCategories:
    def test_twelve_categories(self):
        assert len(EMOTION_CATEGORIES) == 12

    def test_required_emotions_present(self):
        required = ["happy", "sad", "angry", "fearful", "surprised",
                     "disgusted", "neutral", "bored", "excited",
                     "tender", "sarcastic", "whisper"]
        for emotion in required:
            assert emotion in EMOTION_CATEGORIES

    def test_emotion_to_id_consistency(self):
        for emotion, idx in EMOTION_TO_ID.items():
            assert EMOTION_CATEGORIES[idx] == emotion

    def test_no_duplicate_categories(self):
        assert len(EMOTION_CATEGORIES) == len(set(EMOTION_CATEGORIES))


class TestStyleParams:
    def test_neutral_default(self):
        s = StyleParams.neutral()
        assert s.emotion == "neutral"
        assert s.valence == 0.0
        assert s.arousal == 0.0

    def test_to_vector_length(self):
        s = StyleParams.neutral()
        vec = s.to_vector()
        assert len(vec) == 32

    def test_to_vector_neutral_mostly_zero(self):
        s = StyleParams.neutral()
        vec = s.to_vector()
        # VAD should be zero
        assert vec[0] == 0.0
        assert vec[1] == 0.0
        assert vec[2] == 0.0
        # Neutral emotion one-hot at index 9+6=15
        neutral_id = EMOTION_TO_ID["neutral"]
        assert vec[9 + neutral_id] == 1.0

    def test_to_vector_happy(self):
        s = StyleParams(emotion="happy", valence=0.7, arousal=0.5)
        vec = s.to_vector()
        assert vec[0] == pytest.approx(0.7)
        assert vec[1] == pytest.approx(0.5)
        happy_id = EMOTION_TO_ID["happy"]
        assert vec[9 + happy_id] == 1.0

    def test_to_vector_clamps_values(self):
        s = StyleParams(valence=5.0, arousal=-3.0)
        vec = s.to_vector()
        assert vec[0] == pytest.approx(1.0)
        assert vec[1] == pytest.approx(-1.0)

    def test_from_dict(self):
        d = {
            "emotion": "angry",
            "valence": -0.3,
            "arousal": 0.8,
            "dominance": 0.5,
            "reasoning": "test",
        }
        s = StyleParams.from_dict(d)
        assert s.emotion == "angry"
        assert s.valence == -0.3
        assert s.reasoning == "test"

    def test_from_dict_defaults(self):
        s = StyleParams.from_dict({})
        assert s.emotion == "neutral"
        assert s.valence == 0.0

    def test_unknown_emotion_in_vector(self):
        s = StyleParams(emotion="nonexistent")
        vec = s.to_vector()
        # Falls back to neutral
        neutral_id = EMOTION_TO_ID["neutral"]
        assert vec[9 + neutral_id] == 1.0

    def test_prosody_in_vector(self):
        s = StyleParams(speech_rate=0.5, energy=-0.3, pitch_range=0.8)
        vec = s.to_vector()
        assert vec[6] == pytest.approx(0.5)
        assert vec[7] == pytest.approx(-0.3)
        assert vec[8] == pytest.approx(0.8)


class TestCharacterProfile:
    def test_basic_creation(self):
        c = CharacterProfile(
            name="テスト",
            personality="明るい",
            voice_description="高い声",
        )
        assert c.name == "テスト"
        assert c.language == "ja"
        assert c.speaker_file is None

    def test_default_style_is_neutral(self):
        c = CharacterProfile(name="test")
        assert c.default_style.emotion == "neutral"

    def test_unsupported_language_raises(self):
        with pytest.raises(ValueError, match="Unsupported language"):
            CharacterProfile(name="test", language="other")


class TestDialogueTurn:
    def test_basic_turn(self):
        t = DialogueTurn(speaker="Alice", text="Hello!")
        assert t.speaker == "Alice"
        assert t.emotion is None
        assert t.timestamp is None

    def test_with_emotion(self):
        t = DialogueTurn(speaker="Bob", text="Oh no!", emotion="sad")
        assert t.emotion == "sad"


class TestScript:
    def test_empty_script(self):
        s = Script()
        assert s.title == ""
        assert s.entries == []
        assert s.characters == {}

    def test_script_with_entries(self):
        s = Script(
            title="Test Scene",
            situation="A test scenario",
            characters={
                "alice": CharacterProfile(name="Alice"),
            },
            entries=[
                ScriptEntry(speaker="alice", text="Hello!", hint="cheerful"),
            ],
        )
        assert len(s.entries) == 1
        assert s.entries[0].hint == "cheerful"
        assert "alice" in s.characters
