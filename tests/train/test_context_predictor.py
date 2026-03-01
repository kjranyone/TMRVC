"""Tests for ContextStylePredictor (UCLM v2)."""

import pytest
from tmrvc_train.context_predictor import ContextStylePredictor
from tmrvc_core.dialogue_types import CharacterProfile


class TestRuleBasedPrediction:
    @pytest.fixture
    def predictor(self):
        return ContextStylePredictor()

    def test_neutral_text(self, predictor):
        style = predictor.predict_rule_based("これはテストです。")
        assert style.emotion == "neutral"
        assert style.breathiness == 0.0

    def test_whisper_keyword(self, predictor):
        style = predictor.predict_rule_based("ひそひそ話をする。")
        assert style.emotion == "whisper"
        assert style.breathiness > 0.5
        assert style.voicing < 0.5

    def test_happy_keyword(self, predictor):
        style = predictor.predict_rule_based("ありがとう！嬉しい！")
        assert style.emotion == "happy"
        assert style.valence > 0.0

    def test_exclamation_increases_arousal(self, predictor):
        style = predictor.predict_rule_based("すごい！！")
        assert style.arousal > 0.0
        assert style.energy > 0.0

    def test_ellipsis_decreases_energy(self, predictor):
        style = predictor.predict_rule_based("そうなんだ…")
        assert style.energy < 0.0
        assert style.arousal < 0.0
