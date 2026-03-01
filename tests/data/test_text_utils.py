"""Tests for UCLM v2 text utilities and style inference."""

import pytest
from tmrvc_core.text_utils import analyze_inline_stage_directions, infer_sentence_style
from tmrvc_core.dialogue_types import StyleParams


class TestInferSentenceStyle:
    def test_neutral_stays_neutral(self):
        style = infer_sentence_style("普通の文章です。", language="ja")
        assert style.emotion == "neutral"
        assert style.arousal == 0.0

    def test_exclamation_boosts_arousal(self):
        style = infer_sentence_style("すごい！！", language="ja")
        assert style.arousal > 0.0
        assert style.energy > 0.0

    def test_happy_keyword(self):
        style = infer_sentence_style("ありがとう！", language="ja")
        assert style.emotion == "happy"
        assert style.valence > 0.0


class TestAnalyzeInlineStageDirections:
    def test_whisper_direction(self):
        # [whisper] hello -> emotion=whisper, high breathiness, low voicing
        res = analyze_inline_stage_directions("[whisper] こんにちは", language="ja")
        assert res.spoken_text == "こんにちは"
        assert res.style_overlay.emotion == "whisper"
        assert res.style_overlay.breathiness > 0.5
        assert res.style_overlay.voicing < 0.5

    def test_prefix_silence(self):
        res = analyze_inline_stage_directions("(200ms) テスト", language="ja")
        assert res.leading_silence_ms == 200
        assert res.spoken_text == "テスト"

    def test_multiple_overlays(self):
        # Combining emotion and physical tweak
        res = analyze_inline_stage_directions("[happy, tension=0.8] やった！", language="ja")
        assert res.style_overlay.emotion == "happy"
        assert res.style_overlay.tension == pytest.approx(0.8)
