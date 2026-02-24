"""Tests for YAML script parser."""

import pytest
import yaml

from tmrvc_core.dialogue_types import CharacterProfile, Script, ScriptEntry, StyleParams
from tmrvc_data.script_parser import load_script_from_string, script_to_yaml


SAMPLE_SCRIPT = """\
title: "Scene 1 - 再会"
situation: "10年ぶりに駅で再会した幼馴染"
characters:
  sakura:
    name: "桜"
    personality: "明るく感情的な25歳女性"
    voice_description: "高めの声"
    language: "ja"
  yuki:
    name: "ゆき"
    personality: "落ち着いた26歳女性"
dialogue:
  - speaker: sakura
    text: "ゆきちゃん！本当に久しぶり！"
    hint: "歓喜"
  - speaker: yuki
    text: "さくら...まさか来てくれるなんて。"
    hint: "驚き"
    emotion: "surprised"
"""


class TestLoadScript:
    def test_basic_parse(self):
        script = load_script_from_string(SAMPLE_SCRIPT)
        assert script.title == "Scene 1 - 再会"
        assert script.situation == "10年ぶりに駅で再会した幼馴染"

    def test_characters(self):
        script = load_script_from_string(SAMPLE_SCRIPT)
        assert "sakura" in script.characters
        assert "yuki" in script.characters
        assert script.characters["sakura"].name == "桜"
        assert script.characters["sakura"].language == "ja"

    def test_dialogue_entries(self):
        script = load_script_from_string(SAMPLE_SCRIPT)
        assert len(script.entries) == 2
        assert script.entries[0].speaker == "sakura"
        assert script.entries[0].text == "ゆきちゃん！本当に久しぶり！"
        assert script.entries[0].hint == "歓喜"

    def test_emotion_override(self):
        script = load_script_from_string(SAMPLE_SCRIPT)
        assert script.entries[1].style_override is not None
        assert script.entries[1].style_override.emotion == "surprised"

    def test_empty_script(self):
        script = load_script_from_string("title: test\n")
        assert script.title == "test"
        assert script.entries == []
        assert script.characters == {}

    def test_short_character_form(self):
        yaml_str = """\
characters:
  narrator: "ナレーター"
dialogue:
  - speaker: narrator
    text: "昔々あるところに"
"""
        script = load_script_from_string(yaml_str)
        assert script.characters["narrator"].name == "ナレーター"

    def test_invalid_yaml_type(self):
        with pytest.raises(ValueError, match="YAML mapping"):
            load_script_from_string("- just a list\n")

    def test_missing_speaker_text_skipped(self):
        yaml_str = """\
dialogue:
  - speaker: alice
  - text: "no speaker"
  - speaker: bob
    text: "valid"
"""
        script = load_script_from_string(yaml_str)
        assert len(script.entries) == 1
        assert script.entries[0].speaker == "bob"

    def test_default_style(self):
        yaml_str = """\
characters:
  test:
    name: "テスト"
    default_style:
      emotion: "happy"
      valence: 0.5
dialogue:
  - speaker: test
    text: "hello"
"""
        script = load_script_from_string(yaml_str)
        assert script.characters["test"].default_style.emotion == "happy"
        assert script.characters["test"].default_style.valence == 0.5


class TestScriptToYaml:
    def test_roundtrip(self):
        script = load_script_from_string(SAMPLE_SCRIPT)
        yaml_str = script_to_yaml(script)
        script2 = load_script_from_string(yaml_str)
        assert script2.title == script.title
        assert len(script2.entries) == len(script.entries)

    def test_empty_script(self):
        script = Script()
        yaml_str = script_to_yaml(script)
        assert yaml_str.strip() == "{}"

    def test_with_characters(self):
        script = Script(
            title="Test",
            characters={"a": CharacterProfile(name="Alice", language="en")},
            entries=[ScriptEntry(speaker="a", text="Hi")],
        )
        yaml_str = script_to_yaml(script)
        assert "Alice" in yaml_str
        assert "language: en" in yaml_str
