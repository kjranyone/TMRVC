"""Tests for v4 dialogue and style types (12-D physical voice state)."""

import pytest
from tmrvc_core.dialogue_types import StyleParams


class TestStyleParams:
    def test_to_vector_length(self):
        s = StyleParams.neutral()
        vec = s.to_vector()
        # v4 contract: 12-D physical voice state
        assert len(vec) == 12

    def test_to_vector_neutral_values(self):
        s = StyleParams.neutral()
        vec = s.to_vector()
        # Default neutral values match voice_state registry defaults
        assert vec[0] == 0.5   # pitch_level
        assert vec[1] == 0.3   # pitch_range
        assert vec[2] == 0.5   # energy_level
        assert vec[3] == 0.35  # pressedness (tension)
        assert vec[4] == 0.5   # spectral_tilt
        assert vec[5] == 0.2   # breathiness
        assert vec[6] == 0.15  # voice_irregularity (roughness)
        assert vec[7] == 0.5   # openness
        assert vec[8] == 0.2   # aperiodicity
        assert vec[9] == 0.5   # formant_shift
        assert vec[10] == 0.4  # vocal_effort
        assert vec[11] == 0.1  # creak

    def test_to_vector_custom_values(self):
        s = StyleParams(
            emotion="happy",
            breathiness=0.7,
            tension=0.6,
            energy=0.8,
        )
        vec = s.to_vector()
        assert vec[5] == 0.7  # breathiness at idx 5
        assert vec[3] == 0.6  # tension (pressedness) at idx 3
        assert vec[2] == 0.8  # energy (energy_level) at idx 2

    def test_to_vector_clamps_values(self):
        s = StyleParams(breathiness=5.0, tension=-3.0, creak=2.0)
        vec = s.to_vector()
        assert vec[5] == 1.0  # clamped breathiness
        assert vec[3] == 0.0  # clamped tension (pressedness)
        assert vec[11] == 1.0 # clamped creak

    def test_to_vector_new_dimensions(self):
        s = StyleParams(
            spectral_tilt=0.8,
            openness=0.6,
            aperiodicity=0.3,
            formant_shift=0.7,
            vocal_effort=0.9,
            creak=0.4,
        )
        vec = s.to_vector()
        assert vec[4] == 0.8   # spectral_tilt
        assert vec[7] == 0.6   # openness
        assert vec[8] == 0.3   # aperiodicity
        assert vec[9] == 0.7   # formant_shift
        assert vec[10] == 0.9  # vocal_effort
        assert vec[11] == 0.4  # creak

    def test_from_dict(self):
        d = {
            "emotion": "whisper",
            "breathiness": 0.8,
            "voicing": 0.2,
            "speech_rate": 1.5,
            "spectral_tilt": 0.7,
            "creak": 0.3,
        }
        s = StyleParams.from_dict(d)
        assert s.emotion == "whisper"
        assert s.breathiness == 0.8
        assert s.voicing == 0.2
        assert s.speech_rate == 1.5
        assert s.spectral_tilt == 0.7
        assert s.creak == 0.3

    def test_from_dict_defaults_new_fields(self):
        # Legacy dict without new v4 fields should get registry defaults
        d = {"emotion": "neutral", "breathiness": 0.5}
        s = StyleParams.from_dict(d)
        assert s.spectral_tilt == 0.5
        assert s.openness == 0.5
        assert s.aperiodicity == 0.2
        assert s.formant_shift == 0.5
        assert s.vocal_effort == 0.4
        assert s.creak == 0.1
        assert s.pitch_level == 0.5
        assert s.pitch_range == 0.3

    def test_semantic_fields_not_in_vector(self):
        """arousal, valence, voicing, speech_rate are semantic/control
        fields and must not appear in the 12-D physical vector."""
        s = StyleParams(arousal=0.99, valence=-0.5, voicing=0.77, speech_rate=1.8)
        vec = s.to_vector()
        assert len(vec) == 12
        # None of the semantic values should appear in the vector
        assert 0.99 not in vec  # arousal
        assert -0.5 not in vec  # valence
        assert 0.77 not in vec  # voicing
        assert 1.8 not in vec   # speech_rate
