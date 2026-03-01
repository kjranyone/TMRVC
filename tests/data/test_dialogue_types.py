"""Tests for UCLM v2 dialogue and style types."""

import pytest
from tmrvc_core.dialogue_types import StyleParams


class TestStyleParams:
    def test_to_vector_length(self):
        s = StyleParams.neutral()
        vec = s.to_vector()
        # Contract Section 3.1: 8-dim physical voice state
        assert len(vec) == 8

    def test_to_vector_neutral_values(self):
        s = StyleParams.neutral()
        vec = s.to_vector()
        # Default neutral values
        assert vec[0] == 0.0 # breathiness
        assert vec[5] == 1.0 # voicing
        assert vec[7] == 1.0 # speech_rate

    def test_to_vector_happy_mapping(self):
        # In v2, happy is just a label, but physical params control it
        s = StyleParams(emotion="happy", arousal=0.7, valence=0.5)
        vec = s.to_vector()
        assert vec[2] == 0.7 # arousal
        assert vec[3] == 0.5 # valence

    def test_to_vector_clamps_values(self):
        s = StyleParams(valence=5.0, arousal=-3.0)
        vec = s.to_vector()
        assert vec[3] == 1.0 # clamped valence
        assert vec[2] == 0.0 # clamped arousal

    def test_from_dict(self):
        d = {
            "emotion": "whisper",
            "breathiness": 0.8,
            "voicing": 0.2,
            "speech_rate": 1.5
        }
        s = StyleParams.from_dict(d)
        assert s.emotion == "whisper"
        assert s.breathiness == 0.8
        assert s.voicing == 0.2
        assert s.speech_rate == 1.5
