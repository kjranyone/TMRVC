"""Tests for inline stage-direction analysis in text utils."""

from __future__ import annotations

from tmrvc_core.dialogue_types import StyleParams
from tmrvc_core.text_utils import analyze_inline_stage_directions


class TestInlineStageAnalysis:
    def test_no_stage_blocks_is_noop(self):
        text = "Hello there."
        analysis = analyze_inline_stage_directions(text, language="en")
        assert analysis.spoken_text == text
        assert analysis.stage_directions == []
        assert analysis.style_overlay is None
        assert analysis.speed_scale == 1.0

    def test_prefix_and_suffix_breath_controls(self):
        text = "(deep breath) Hello there. (long exhale)"
        analysis = analyze_inline_stage_directions(text, language="en")
        assert analysis.spoken_text == "Hello there."
        assert len(analysis.stage_directions) == 2
        assert analysis.leading_silence_ms > 0
        assert analysis.trailing_silence_ms > 0
        assert analysis.speed_scale < 1.0
        assert isinstance(analysis.style_overlay, StyleParams)
        assert analysis.style_overlay.energy < 0.0

    def test_middle_pause_affects_sentence_pause_delta(self):
        text = "Hello (pause) there."
        analysis = analyze_inline_stage_directions(text, language="en")
        assert analysis.spoken_text == "Hello there."
        assert analysis.leading_silence_ms == 0
        assert analysis.trailing_silence_ms == 0
        assert analysis.sentence_pause_ms_delta > 0

    def test_stage_only_text_falls_back_to_original_text(self):
        text = "(whisper)"
        analysis = analyze_inline_stage_directions(text, language="en")
        assert analysis.spoken_text == text
        assert isinstance(analysis.style_overlay, StyleParams)
        assert analysis.style_overlay.emotion == "whisper"
