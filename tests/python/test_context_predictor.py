"""Tests for ContextStylePredictor."""

import json

import pytest

from tmrvc_core.dialogue_types import CharacterProfile, DialogueTurn, StyleParams
from tmrvc_train.context_predictor import ContextStylePredictor


class TestRuleBasedPrediction:
    def setup_method(self):
        self.predictor = ContextStylePredictor(api_key=None)

    def test_neutral_text(self):
        style = self.predictor.predict_rule_based("普通のテキストです。")
        assert style.emotion == "neutral"

    def test_exclamation_increases_arousal(self):
        style = self.predictor.predict_rule_based("すごい！")
        assert style.arousal > 0.0
        assert style.energy > 0.0

    def test_question_increases_pitch_range(self):
        style = self.predictor.predict_rule_based("本当ですか？")
        assert style.pitch_range > 0.0

    def test_ellipsis_decreases_arousal(self):
        style = self.predictor.predict_rule_based("そう…")
        assert style.arousal < 0.0
        assert style.speech_rate < 0.0

    def test_happy_keyword(self):
        style = self.predictor.predict_rule_based("嬉しいです！")
        assert style.emotion == "happy"
        assert style.valence > 0.0

    def test_sad_keyword(self):
        style = self.predictor.predict_rule_based("悲しいことがあった")
        assert style.emotion == "sad"
        assert style.valence < 0.0

    def test_angry_keyword(self):
        style = self.predictor.predict_rule_based("怒りを感じる")
        assert style.emotion == "angry"

    def test_surprised_keyword(self):
        style = self.predictor.predict_rule_based("えっ！？本当に？")
        assert style.emotion == "surprised"
        assert style.arousal > 0.0

    def test_whisper_keyword(self):
        style = self.predictor.predict_rule_based("内緒だけどね")
        assert style.emotion == "whisper"
        assert style.energy < 0.0

    def test_thank_you_keyword(self):
        style = self.predictor.predict_rule_based("ありがとうございます")
        assert style.emotion == "happy"

    def test_character_defaults_applied(self):
        character = CharacterProfile(
            name="テスト",
            default_style=StyleParams(valence=0.3, energy=0.2),
        )
        style = self.predictor.predict_rule_based("普通のテキスト。", character)
        assert style.valence == pytest.approx(0.3)
        assert style.energy == pytest.approx(0.2)

    def test_character_defaults_plus_keywords(self):
        character = CharacterProfile(
            name="テスト",
            default_style=StyleParams(valence=0.1),
        )
        style = self.predictor.predict_rule_based("嬉しい！", character)
        # Character default + happy keyword valence + exclamation
        assert style.valence > 0.1
        assert style.emotion == "happy"

    def test_multiple_punctuation_additive(self):
        style = self.predictor.predict_rule_based("本当に！？")
        assert style.arousal > 0.0
        assert style.pitch_range > 0.0


class TestResponseParsing:
    def setup_method(self):
        self.predictor = ContextStylePredictor(api_key=None)

    def test_parse_valid_json(self):
        response_text = json.dumps({
            "emotion": "happy",
            "valence": 0.7,
            "arousal": 0.5,
            "dominance": 0.3,
            "speech_rate": 0.1,
            "energy": 0.2,
            "pitch_range": 0.4,
            "reasoning": "greeting response",
        })
        style = self.predictor._parse_response(response_text)
        assert style.emotion == "happy"
        assert style.valence == pytest.approx(0.7)

    def test_parse_json_in_markdown(self):
        response_text = '```json\n{"emotion": "sad", "valence": -0.5}\n```'
        style = self.predictor._parse_response(response_text)
        assert style.emotion == "sad"

    def test_parse_invalid_json(self):
        style = self.predictor._parse_response("not json at all")
        assert style.emotion == "neutral"

    def test_parse_unknown_emotion(self):
        response_text = json.dumps({"emotion": "nonexistent_emotion"})
        style = self.predictor._parse_response(response_text)
        assert style.emotion == "neutral"

    def test_parse_partial_response(self):
        response_text = json.dumps({"emotion": "angry"})
        style = self.predictor._parse_response(response_text)
        assert style.emotion == "angry"
        assert style.valence == 0.0  # default


class TestPromptBuilding:
    def setup_method(self):
        self.predictor = ContextStylePredictor(api_key=None)

    def test_basic_prompt(self):
        character = CharacterProfile(
            name="桜",
            personality="明るい",
            voice_description="高い声",
        )
        prompt = self.predictor._build_user_prompt(
            character, [], "こんにちは！",
        )
        assert "桜" in prompt
        assert "明るい" in prompt
        assert "こんにちは！" in prompt

    def test_prompt_with_history(self):
        character = CharacterProfile(name="テスト")
        history = [
            DialogueTurn(speaker="Alice", text="Hi!"),
            DialogueTurn(speaker="テスト", text="Hello!"),
        ]
        prompt = self.predictor._build_user_prompt(
            character, history, "Nice day!",
        )
        assert "Alice" in prompt
        assert "Hi!" in prompt

    def test_prompt_with_situation(self):
        character = CharacterProfile(name="テスト")
        prompt = self.predictor._build_user_prompt(
            character, [], "こんにちは", situation="学校の教室で",
        )
        assert "学校の教室で" in prompt

    def test_prompt_truncates_history(self):
        character = CharacterProfile(name="テスト")
        history = [DialogueTurn(speaker=f"s{i}", text=f"t{i}") for i in range(20)]
        predictor = ContextStylePredictor(api_key=None, max_history=5)
        prompt = predictor._build_user_prompt(character, history, "test")
        # Only last 5 turns should appear
        assert "s15" in prompt
        assert "s19" in prompt
        assert "s0" not in prompt

    def test_prompt_with_emotion_tags(self):
        character = CharacterProfile(name="テスト")
        history = [
            DialogueTurn(speaker="A", text="Bad news.", emotion="sad"),
        ]
        prompt = self.predictor._build_user_prompt(
            character, history, "Really?",
        )
        assert "[sad]" in prompt


class TestSyncPrediction:
    def test_no_api_key_falls_back(self):
        predictor = ContextStylePredictor(api_key=None)
        character = CharacterProfile(name="テスト")
        style = predictor.predict_sync(character, [], "嬉しい！")
        assert style.emotion == "happy"

    def test_predict_sync_returns_style_params(self):
        predictor = ContextStylePredictor(api_key=None)
        character = CharacterProfile(name="テスト")
        style = predictor.predict_sync(character, [], "テスト")
        assert isinstance(style, StyleParams)
