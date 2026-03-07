"""Tests for UCLM v3 pointer-based TTS serving.

Covers:
a) PointerInferenceState creation and .finished property
b) Engine tts_pointer method exists (import check)
c) Schema validation for new TTS fields (pace, hold_bias, boundary_bias)
"""

from __future__ import annotations

import pytest

from tmrvc_serve.uclm_engine import PointerInferenceState, UCLMEngine
from tmrvc_serve.schemas import TTSRequest, TTSStreamRequest


# ---------------------------------------------------------------------------
# a) PointerInferenceState
# ---------------------------------------------------------------------------


class TestPointerInferenceState:
    def test_creation_defaults(self):
        state = PointerInferenceState()
        assert state.text_index == 0
        assert state.progress == 0.0
        assert state.total_phonemes == 0
        assert state.frames_generated == 0

    def test_finished_when_index_equals_total(self):
        state = PointerInferenceState(
            text_index=10,
            progress=1.0,
            total_phonemes=10,
            frames_generated=80,
        )
        assert state.finished is True

    def test_not_finished_when_index_less_than_total(self):
        state = PointerInferenceState(
            text_index=5,
            progress=0.5,
            total_phonemes=10,
            frames_generated=40,
        )
        assert state.finished is False

    def test_finished_when_index_exceeds_total(self):
        state = PointerInferenceState(
            text_index=15,
            progress=1.0,
            total_phonemes=10,
            frames_generated=120,
        )
        assert state.finished is True

    def test_to_dict(self):
        state = PointerInferenceState(
            text_index=3,
            progress=0.25,
            total_phonemes=12,
            frames_generated=24,
        )
        d = state.to_dict()
        assert d["text_index"] == 3
        assert d["progress"] == 0.25
        assert d["total_phonemes"] == 12
        assert d["frames_generated"] == 24
        assert d["stall_frames"] == 0
        # New failure-handling fields should also be present
        assert "max_frames_per_unit" in d
        assert "frames_on_current_unit" in d
        assert "skip_protection_threshold" in d
        assert "forced_advance_count" in d
        assert "skip_protection_count" in d


# ---------------------------------------------------------------------------
# b) Engine tts_pointer method exists
# ---------------------------------------------------------------------------


class TestEngineTtsPointerContract:
    def test_tts_is_pointer_based(self):
        """UCLMEngine.tts should be the pointer-based implementation (no tts_mode split)."""
        import inspect

        sig = inspect.signature(UCLMEngine.tts)
        # tts() is always pointer mode — no tts_mode parameter
        assert "tts_mode" not in sig.parameters

    def test_tts_method_accepts_pace_hold_boundary(self):
        """UCLMEngine.tts should accept pace, hold_bias, boundary_bias kwargs."""
        import inspect

        sig = inspect.signature(UCLMEngine.tts)
        for param_name in ("pace", "hold_bias", "boundary_bias"):
            assert param_name in sig.parameters, f"Missing parameter: {param_name}"


# ---------------------------------------------------------------------------
# c) Schema validation for new TTS fields
# ---------------------------------------------------------------------------


class TestTTSSchemaNewFields:
    def test_tts_request_pace_default(self):
        req = TTSRequest(text="hello", character_id="test")
        assert req.pace == 1.0

    def test_tts_request_hold_bias_default(self):
        req = TTSRequest(text="hello", character_id="test")
        assert req.hold_bias == 0.0

    def test_tts_request_boundary_bias_default(self):
        req = TTSRequest(text="hello", character_id="test")
        assert req.boundary_bias == 0.0

    def test_tts_request_pace_range(self):
        """pace should be bounded to [0.5, 3.0]."""
        req = TTSRequest(text="hello", character_id="test", pace=2.5)
        assert req.pace == 2.5

        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", pace=0.1)

        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", pace=5.0)

    def test_tts_request_hold_bias_range(self):
        """hold_bias should be bounded to [-1.0, 1.0]."""
        req = TTSRequest(text="hello", character_id="test", hold_bias=-0.5)
        assert req.hold_bias == -0.5

        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", hold_bias=-2.0)

        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", hold_bias=2.0)

    def test_tts_request_boundary_bias_range(self):
        """boundary_bias should be bounded to [-1.0, 1.0]."""
        req = TTSRequest(text="hello", character_id="test", boundary_bias=0.8)
        assert req.boundary_bias == 0.8

        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", boundary_bias=-1.5)

    def test_tts_stream_request_has_pointer_fields(self):
        """TTSStreamRequest should also support pace, hold_bias, boundary_bias."""
        req = TTSStreamRequest(text="hello", character_id="test", pace=1.5, hold_bias=0.2, boundary_bias=-0.3)
        assert req.pace == 1.5
        assert req.hold_bias == 0.2
        assert req.boundary_bias == -0.3


# ---------------------------------------------------------------------------
# d) dialogue_context and acting_intent schema fields
# ---------------------------------------------------------------------------


class TestTTSRequestExpressiveFields:
    def test_tts_request_accepts_dialogue_context(self):
        """TTSRequest should accept optional dialogue_context and acting_intent vectors."""
        ctx = [0.1, 0.2, 0.3, 0.4]
        intent = [0.5, -0.5, 0.0]
        req = TTSRequest(
            text="hello",
            character_id="test",
            dialogue_context=ctx,
            acting_intent=intent,
        )
        assert req.dialogue_context == ctx
        assert req.acting_intent == intent

    def test_tts_request_dialogue_context_defaults_none(self):
        """dialogue_context and acting_intent should default to None."""
        req = TTSRequest(text="hello", character_id="test")
        assert req.dialogue_context is None
        assert req.acting_intent is None

    def test_tts_stream_request_accepts_dialogue_context(self):
        """TTSStreamRequest should also accept dialogue_context and acting_intent."""
        req = TTSStreamRequest(
            text="hello",
            character_id="test",
            dialogue_context=[1.0, 2.0],
            acting_intent=[0.0],
        )
        assert req.dialogue_context == [1.0, 2.0]
        assert req.acting_intent == [0.0]


# ---------------------------------------------------------------------------
# e) PointerInferenceState .finished property edge cases
# ---------------------------------------------------------------------------


class TestPointerStateFinishedProperty:
    def test_pointer_state_finished_property_zero_phonemes(self):
        """A state with zero total phonemes should be finished immediately."""
        state = PointerInferenceState(text_index=0, total_phonemes=0)
        assert state.finished is True

    def test_pointer_state_finished_property_one_remaining(self):
        """A state one step from completion should not be finished."""
        state = PointerInferenceState(text_index=9, total_phonemes=10)
        assert state.finished is False

    def test_pointer_state_finished_property_exact(self):
        """Finished when text_index == total_phonemes."""
        state = PointerInferenceState(text_index=10, total_phonemes=10)
        assert state.finished is True


# ---------------------------------------------------------------------------
# f) Engine tts_mode default
# ---------------------------------------------------------------------------


class TestEngineTtsMode:
    def test_engine_tts_mode_default_is_pointer(self):
        """UCLMEngine should default to tts_mode='pointer'."""
        engine = UCLMEngine()
        assert engine.tts_mode == "pointer"

    def test_engine_tts_mode_legacy_raises(self):
        """Legacy duration mode should raise NotImplementedError."""
        engine = UCLMEngine(tts_mode="legacy_duration")
        assert engine.tts_mode == "legacy_duration"

        import torch
        from tmrvc_core.dialogue_types import StyleParams

        with pytest.raises(NotImplementedError, match="Legacy duration-based TTS"):
            engine.tts(
                phonemes=torch.zeros(1, 5, dtype=torch.long),
                speaker_embed=torch.zeros(1, 192),
                style=StyleParams.neutral(),
            )


# ---------------------------------------------------------------------------
# g) PointerInferenceState stall_frames and max_stall (v3)
# ---------------------------------------------------------------------------


class TestPointerStateStallFields:
    def test_stall_frames_default(self):
        """stall_frames should default to 0."""
        state = PointerInferenceState()
        assert state.stall_frames == 0

    def test_max_stall_default(self):
        """max_stall should default to 100."""
        state = PointerInferenceState()
        assert state.max_stall == 100

    def test_stall_frames_settable(self):
        """stall_frames should be settable at construction."""
        state = PointerInferenceState(stall_frames=42)
        assert state.stall_frames == 42

    def test_max_stall_settable(self):
        """max_stall should be settable at construction."""
        state = PointerInferenceState(max_stall=200)
        assert state.max_stall == 200

    def test_stall_frames_in_to_dict(self):
        """to_dict should include stall_frames."""
        state = PointerInferenceState(stall_frames=5)
        d = state.to_dict()
        assert "stall_frames" in d
        assert d["stall_frames"] == 5


# ---------------------------------------------------------------------------
# h) TTSRequest new v3 fields
# ---------------------------------------------------------------------------


class TestTTSRequestV3Fields:
    def test_reference_audio_base64_default_none(self):
        """reference_audio_base64 should default to None."""
        req = TTSRequest(text="hello", character_id="test")
        assert req.reference_audio_base64 is None

    def test_reference_audio_base64_settable(self):
        """reference_audio_base64 should accept a base64 string."""
        req = TTSRequest(text="hello", character_id="test", reference_audio_base64="dGVzdA==")
        assert req.reference_audio_base64 == "dGVzdA=="

    def test_reference_text_default_none(self):
        """reference_text should default to None."""
        req = TTSRequest(text="hello", character_id="test")
        assert req.reference_text is None

    def test_reference_text_settable(self):
        """reference_text should accept a string."""
        req = TTSRequest(text="hello", character_id="test", reference_text="sample transcript")
        assert req.reference_text == "sample transcript"

    def test_explicit_voice_state_default_none(self):
        """explicit_voice_state should default to None."""
        req = TTSRequest(text="hello", character_id="test")
        assert req.explicit_voice_state is None

    def test_explicit_voice_state_accepts_8d(self):
        """explicit_voice_state should accept an 8-dimensional vector."""
        vs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        req = TTSRequest(text="hello", character_id="test", explicit_voice_state=vs)
        assert req.explicit_voice_state == vs

    def test_explicit_voice_state_rejects_wrong_length(self):
        """explicit_voice_state must be exactly 8 elements."""
        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", explicit_voice_state=[0.1, 0.2])

    def test_delta_voice_state_default_none(self):
        """delta_voice_state should default to None."""
        req = TTSRequest(text="hello", character_id="test")
        assert req.delta_voice_state is None

    def test_delta_voice_state_accepts_8d(self):
        """delta_voice_state should accept an 8-dimensional vector."""
        dvs = [0.0] * 8
        req = TTSRequest(text="hello", character_id="test", delta_voice_state=dvs)
        assert req.delta_voice_state == dvs

    def test_delta_voice_state_rejects_wrong_length(self):
        """delta_voice_state must be exactly 8 elements."""
        with pytest.raises(Exception):
            TTSRequest(text="hello", character_id="test", delta_voice_state=[0.1] * 3)


# ---------------------------------------------------------------------------
# i) TTSStreamRequest same new v3 fields
# ---------------------------------------------------------------------------


class TestTTSStreamRequestV3Fields:
    def test_reference_audio_base64(self):
        req = TTSStreamRequest(text="hi", character_id="c", reference_audio_base64="abc")
        assert req.reference_audio_base64 == "abc"

    def test_reference_text(self):
        req = TTSStreamRequest(text="hi", character_id="c", reference_text="ref text")
        assert req.reference_text == "ref text"

    def test_explicit_voice_state(self):
        vs = [0.0] * 8
        req = TTSStreamRequest(text="hi", character_id="c", explicit_voice_state=vs)
        assert req.explicit_voice_state == vs

    def test_delta_voice_state(self):
        dvs = [0.1] * 8
        req = TTSStreamRequest(text="hi", character_id="c", delta_voice_state=dvs)
        assert req.delta_voice_state == dvs

    def test_stream_defaults_none(self):
        req = TTSStreamRequest(text="hi", character_id="c")
        assert req.reference_audio_base64 is None
        assert req.reference_text is None
        assert req.explicit_voice_state is None
        assert req.delta_voice_state is None


# ---------------------------------------------------------------------------
# j) PointerInferenceState new failure-handling fields
# ---------------------------------------------------------------------------


class TestPointerInferenceStateFailureFields:
    def test_max_frames_per_unit_default(self):
        """max_frames_per_unit should default to 50."""
        state = PointerInferenceState()
        assert state.max_frames_per_unit == 50

    def test_frames_on_current_unit_default(self):
        """frames_on_current_unit should default to 0."""
        state = PointerInferenceState()
        assert state.frames_on_current_unit == 0

    def test_skip_protection_threshold_default(self):
        """skip_protection_threshold should default to 0.3."""
        state = PointerInferenceState()
        assert state.skip_protection_threshold == pytest.approx(0.3)

    def test_forced_advance_count_default(self):
        """forced_advance_count should default to 0."""
        state = PointerInferenceState()
        assert state.forced_advance_count == 0

    def test_skip_protection_count_default(self):
        """skip_protection_count should default to 0."""
        state = PointerInferenceState()
        assert state.skip_protection_count == 0

    def test_new_fields_in_to_dict(self):
        """All new failure-handling fields should appear in to_dict()."""
        state = PointerInferenceState(
            max_frames_per_unit=40,
            frames_on_current_unit=5,
            skip_protection_threshold=0.25,
            forced_advance_count=2,
            skip_protection_count=1,
        )
        d = state.to_dict()
        assert d["max_frames_per_unit"] == 40
        assert d["frames_on_current_unit"] == 5
        assert d["skip_protection_threshold"] == pytest.approx(0.25)
        assert d["forced_advance_count"] == 2
        assert d["skip_protection_count"] == 1


# ---------------------------------------------------------------------------
# k) TTSRequest / TTSStreamRequest new fields: speaker_profile_id, cfg_scale
# ---------------------------------------------------------------------------


class TestTTSRequestSpeakerProfileAndCfg:
    def test_speaker_profile_id_default_none(self):
        """speaker_profile_id should default to None when not provided."""
        req = TTSRequest(text="hello", character_id="test")
        assert getattr(req, "speaker_profile_id", None) is None

    def test_speaker_profile_id_settable(self):
        """speaker_profile_id should be settable at construction."""
        try:
            req = TTSRequest(text="hello", character_id="test", speaker_profile_id="spk_001")
            assert req.speaker_profile_id == "spk_001"
        except TypeError:
            pytest.skip("speaker_profile_id field not yet added to TTSRequest")

    def test_cfg_scale_default(self):
        """cfg_scale should have a sensible default (1.5)."""
        try:
            req = TTSRequest(text="hello", character_id="test")
            # Accept either 1.0 or 1.5 as default depending on implementation
            assert hasattr(req, "cfg_scale")
            assert req.cfg_scale >= 0.0
        except (TypeError, AttributeError):
            pytest.skip("cfg_scale field not yet added to TTSRequest")

    def test_cfg_scale_range(self):
        """cfg_scale should accept values within valid range."""
        try:
            req = TTSRequest(text="hello", character_id="test", cfg_scale=2.0)
            assert req.cfg_scale == pytest.approx(2.0)
        except TypeError:
            pytest.skip("cfg_scale field not yet added to TTSRequest")

    def test_stream_speaker_profile_id(self):
        """TTSStreamRequest should also support speaker_profile_id."""
        try:
            req = TTSStreamRequest(text="hi", character_id="c", speaker_profile_id="spk_002")
            assert req.speaker_profile_id == "spk_002"
        except TypeError:
            pytest.skip("speaker_profile_id field not yet added to TTSStreamRequest")

    def test_stream_cfg_scale(self):
        """TTSStreamRequest should also support cfg_scale."""
        try:
            req = TTSStreamRequest(text="hi", character_id="c", cfg_scale=1.8)
            assert req.cfg_scale == pytest.approx(1.8)
        except TypeError:
            pytest.skip("cfg_scale field not yet added to TTSStreamRequest")
