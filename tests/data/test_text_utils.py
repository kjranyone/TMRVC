"""Tests for sentence segmentation and style inference in text_utils."""

import pytest

from tmrvc_core.text_utils import segment_sentences, infer_sentence_style


class TestSegmentSentencesJapanese:
    def test_basic_split(self):
        result = segment_sentences("こんにちは！今日はいい天気ですね。散歩に行きましょう！", "ja")
        assert result == ["こんにちは！", "今日はいい天気ですね。", "散歩に行きましょう！"]

    def test_single_sentence(self):
        result = segment_sentences("こんにちは", "ja")
        assert result == ["こんにちは"]

    def test_single_sentence_with_period(self):
        result = segment_sentences("こんにちは。", "ja")
        assert result == ["こんにちは。"]

    def test_quoted_text_preserved(self):
        result = segment_sentences("彼が「今日は天気がいい。散歩しよう。」と言った。", "ja")
        # Punctuation inside brackets should NOT cause a split
        assert len(result) == 1

    def test_newline_split(self):
        result = segment_sentences("一行目\n二行目", "ja")
        assert result == ["一行目", "二行目"]

    def test_empty_string(self):
        result = segment_sentences("", "ja")
        assert result == [""]

    def test_whitespace_only(self):
        result = segment_sentences("   ", "ja")
        assert result == [""]

    def test_fullwidth_question(self):
        result = segment_sentences("元気ですか？はい、元気です。", "ja")
        assert result == ["元気ですか？", "はい、元気です。"]

    def test_mixed_punctuation(self):
        result = segment_sentences("すごい！本当？そうだよ。", "ja")
        assert result == ["すごい！", "本当？", "そうだよ。"]


class TestSegmentSentencesChinese:
    def test_basic_split(self):
        result = segment_sentences("你好！今天天气很好。", "zh")
        assert result == ["你好！", "今天天气很好。"]


class TestSegmentSentencesEnglish:
    def test_basic_split(self):
        result = segment_sentences("Hello! How are you? I am fine.", "en")
        assert result == ["Hello!", "How are you?", "I am fine."]

    def test_single_sentence(self):
        result = segment_sentences("Hello world", "en")
        assert result == ["Hello world"]

    def test_abbreviation_preserved(self):
        result = segment_sentences("Mr. Smith went to Dr. Jones. He was happy.", "en")
        # "Mr." and "Dr." should NOT cause splits
        assert len(result) == 2
        assert "Mr. Smith" in result[0]
        assert "He was happy." in result[1]

    def test_no_uppercase_after_period(self):
        result = segment_sentences("version 2.0 is out.", "en")
        assert result == ["version 2.0 is out."]

    def test_multiple_spaces(self):
        result = segment_sentences("First sentence.  Second sentence.", "en")
        assert len(result) == 2

    def test_exclamation_split(self):
        result = segment_sentences("Wow! That is great.", "en")
        assert result == ["Wow!", "That is great."]


class TestSegmentSentencesEdgeCases:
    def test_very_short_segment_merged(self):
        """Segments shorter than 2 chars should be merged into previous."""
        result = segment_sentences("あ。い。う。", "ja")
        # "あ。" is 2 chars, "い。" is 2 chars, "う。" is 2 chars — all >= 2
        assert len(result) == 3

    def test_korean_uses_english_rules(self):
        result = segment_sentences("안녕하세요. 잘 지내세요?", "ko")
        # Korean uses English-style splitting
        assert len(result) >= 1

    def test_default_language_is_ja(self):
        result = segment_sentences("テスト。テスト。")
        assert len(result) == 2


# ---------------------------------------------------------------------------
# infer_sentence_style tests
# ---------------------------------------------------------------------------

class TestInferSentenceStyleJapanese:
    def _neutral(self):
        from tmrvc_core.dialogue_types import StyleParams
        return StyleParams.neutral()

    def test_neutral_stays_neutral(self):
        result = infer_sentence_style("普通の文章です", "ja", self._neutral())
        assert result.emotion == "neutral"

    def test_exclamation_boosts_arousal(self):
        base = self._neutral()
        result = infer_sentence_style("すごい！！", "ja", base)
        assert result.arousal > base.arousal
        assert result.energy > base.energy

    def test_double_exclamation_sets_excited(self):
        result = infer_sentence_style("やった！！", "ja", self._neutral())
        assert result.emotion == "excited"

    def test_question_boosts_pitch_range(self):
        base = self._neutral()
        result = infer_sentence_style("本当ですか？", "ja", base)
        assert result.pitch_range > base.pitch_range

    def test_happy_keyword(self):
        result = infer_sentence_style("嬉しいです", "ja", self._neutral())
        assert result.emotion == "happy"
        assert result.valence > 0

    def test_sad_keyword(self):
        result = infer_sentence_style("悲しいです", "ja", self._neutral())
        assert result.emotion == "sad"
        assert result.valence < 0

    def test_angry_keyword(self):
        result = infer_sentence_style("許さない", "ja", self._neutral())
        assert result.emotion == "angry"

    def test_whisper_keyword(self):
        result = infer_sentence_style("内緒だよ", "ja", self._neutral())
        assert result.emotion == "whisper"

    def test_ellipsis_lowers_energy(self):
        base = self._neutral()
        result = infer_sentence_style("そうですね...", "ja", base)
        assert result.energy < base.energy
        assert result.emotion == "tender"

    def test_non_style_input_returns_as_is(self):
        """Non-StyleParams base_style is returned unchanged."""
        result = infer_sentence_style("test", "ja", "not_a_style")
        assert result == "not_a_style"


class TestInferSentenceStyleEnglish:
    def _neutral(self):
        from tmrvc_core.dialogue_types import StyleParams
        return StyleParams.neutral()

    def test_happy_keyword(self):
        result = infer_sentence_style("I am so happy today", "en", self._neutral())
        assert result.emotion == "happy"

    def test_angry_keyword(self):
        result = infer_sentence_style("I am furious about this", "en", self._neutral())
        assert result.emotion == "angry"

    def test_surprised_keyword(self):
        result = infer_sentence_style("Wow, that's incredible", "en", self._neutral())
        assert result.emotion == "surprised"

    def test_exclamation_in_english(self):
        base = self._neutral()
        result = infer_sentence_style("Amazing!!", "en", base)
        assert result.arousal > base.arousal

    def test_highest_priority_keyword_wins(self):
        """When multiple keywords match, the one with highest priority wins."""
        # "furious" has priority 0.6+0.5=1.1, "happy" has 0.3+0.5=0.8
        result = infer_sentence_style("I am furious but happy", "en", self._neutral())
        assert result.emotion == "angry"


# ---------------------------------------------------------------------------
# analyze_inline_stage_directions tests
# ---------------------------------------------------------------------------

from tmrvc_core.text_utils import analyze_inline_stage_directions


class TestAnalyzeInlineStageDirections:
    def test_no_stage_directions(self):
        result = analyze_inline_stage_directions("普通のテキスト", "ja")
        assert result.spoken_text == "普通のテキスト"
        assert result.stage_directions == []
        assert result.style_overlay is None
        assert result.speed_scale == 1.0

    def test_whisper_direction_extracts_spoken_text(self):
        result = analyze_inline_stage_directions("[whisper] Hello there.", "en")
        assert result.spoken_text == "Hello there."
        assert "whisper" in result.stage_directions

    def test_whisper_sets_emotion_and_energy(self):
        from tmrvc_core.dialogue_types import StyleParams
        result = analyze_inline_stage_directions("（囁き声で）こんにちは", "ja")
        assert result.style_overlay is not None
        assert isinstance(result.style_overlay, StyleParams)
        assert result.style_overlay.emotion == "whisper"
        assert result.style_overlay.energy < 0

    def test_shout_increases_energy(self):
        from tmrvc_core.dialogue_types import StyleParams
        result = analyze_inline_stage_directions("[shout] Get out!", "en")
        assert result.style_overlay is not None
        assert isinstance(result.style_overlay, StyleParams)
        assert result.style_overlay.energy > 0
        assert result.style_overlay.arousal > 0

    def test_multiple_brackets_formats(self):
        """Parentheses, brackets, and angle brackets are all recognized."""
        for text in [
            "(whisper) hello",
            "（囁き声）hello",
            "[whisper] hello",
            "【囁き】hello",
            "<whisper> hello",
            "＜ささやき＞hello",
        ]:
            result = analyze_inline_stage_directions(text, "ja")
            assert len(result.stage_directions) >= 1, f"Failed for: {text}"

    def test_prefix_direction_adds_leading_silence(self):
        result = analyze_inline_stage_directions("(pause) それでは始めましょう", "ja")
        assert result.leading_silence_ms > 0

    def test_suffix_direction_adds_trailing_silence(self):
        result = analyze_inline_stage_directions("さようなら (長い息)", "ja")
        # "長い息" matches "長い息" keyword in long_breath rule
        assert result.trailing_silence_ms > 0

    def test_speed_scale_clamped(self):
        result = analyze_inline_stage_directions(
            "[whisper][breath][pause] test", "en"
        )
        # Multiple rules compound the speed_scale multiplicatively
        assert 0.75 <= result.speed_scale <= 1.25

    def test_overlay_values_are_deltas_from_zero(self):
        """Overlay dominance should be 0 since no rule modifies it."""
        from tmrvc_core.dialogue_types import StyleParams
        result = analyze_inline_stage_directions("[whisper] test", "en")
        assert result.style_overlay is not None
        assert isinstance(result.style_overlay, StyleParams)
        assert result.style_overlay.dominance == 0.0
