"""YAML script parser for TTS batch generation.

Parses structured script files (台本) into :class:`Script` objects
for batch TTS generation with character-specific voices and styles.

Script format::

    title: "Scene 1 - 再会"
    situation: "10年ぶりに駅で再会した幼馴染"
    characters:
      sakura:
        name: "桜"
        personality: "明るく感情的な25歳女性"
        voice_description: "高めの声、やや息混じり"
        speaker_file: "models/sakura.tmrvc_speaker"
        language: "ja"
      yuki:
        name: "ゆき"
        personality: "落ち着いた控えめな26歳女性"
        speaker_file: "models/yuki.tmrvc_speaker"
    dialogue:
      - speaker: sakura
        text: "ゆきちゃん！本当に久しぶり！"
        hint: "歓喜、涙ぐみ"
      - speaker: yuki
        text: "さくら...まさか来てくれるなんて。"
        hint: "驚き、感動"
        emotion: "surprised"
        speed: 0.9

Usage::

    from tmrvc_data.script_parser import load_script, load_script_from_string

    script = load_script("scripts/scene1.yaml")
    for entry in script.entries:
        print(f"{entry.speaker}: {entry.text}")
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml

from tmrvc_core.dialogue_types import (
    CharacterProfile,
    Script,
    ScriptEntry,
    StyleParams,
)

logger = logging.getLogger(__name__)


def load_script(path: str | Path) -> Script:
    """Load a script from a YAML file.

    Args:
        path: Path to the YAML script file.

    Returns:
        Parsed :class:`Script` object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the YAML structure is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Script file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Script must be a YAML mapping, got {type(data).__name__}")

    return _parse_script(data, base_dir=path.parent)


def load_script_from_string(text: str, base_dir: Path | None = None) -> Script:
    """Load a script from a YAML string.

    Args:
        text: YAML string.
        base_dir: Base directory for resolving relative paths.

    Returns:
        Parsed :class:`Script` object.
    """
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Script must be a YAML mapping, got {type(data).__name__}")
    return _parse_script(data, base_dir=base_dir or Path("."))


def _parse_script(data: dict, base_dir: Path) -> Script:
    """Parse a script dict into a Script object."""
    title = data.get("title", "")
    situation = data.get("situation", "")

    # Parse characters
    characters: dict[str, CharacterProfile] = {}
    for char_id, char_data in (data.get("characters") or {}).items():
        if isinstance(char_data, str):
            # Short form: just a name
            characters[char_id] = CharacterProfile(name=char_data)
            continue

        speaker_file = None
        if char_data.get("speaker_file"):
            sp = Path(char_data["speaker_file"])
            if not sp.is_absolute():
                sp = base_dir / sp
            speaker_file = sp

        default_style = StyleParams.neutral()
        if char_data.get("default_style"):
            default_style = StyleParams.from_dict(char_data["default_style"])

        characters[char_id] = CharacterProfile(
            name=char_data.get("name", char_id),
            personality=char_data.get("personality", ""),
            voice_description=char_data.get("voice_description", ""),
            default_style=default_style,
            speaker_file=speaker_file,
            language=char_data.get("language", "ja"),
        )

    # Parse dialogue entries
    entries: list[ScriptEntry] = []
    for entry_data in data.get("dialogue") or []:
        if not isinstance(entry_data, dict):
            logger.warning("Skipping non-dict dialogue entry: %s", entry_data)
            continue

        speaker = entry_data.get("speaker", "")
        text = entry_data.get("text", "")
        if not speaker or not text:
            logger.warning("Skipping entry with missing speaker/text: %s", entry_data)
            continue

        # Validate speaker exists in characters (warn but don't fail)
        if characters and speaker not in characters:
            logger.warning("Speaker '%s' not in characters list", speaker)

        hint = entry_data.get("hint")

        # Speed override
        speed = float(entry_data.get("speed", 1.0))

        # Style override from explicit fields
        style_override = None
        if entry_data.get("emotion"):
            style_dict = {}
            if entry_data.get("emotion"):
                style_dict["emotion"] = entry_data["emotion"]
            if entry_data.get("valence") is not None:
                style_dict["valence"] = float(entry_data["valence"])
            if entry_data.get("arousal") is not None:
                style_dict["arousal"] = float(entry_data["arousal"])
            if entry_data.get("energy") is not None:
                style_dict["energy"] = float(entry_data["energy"])
            if entry_data.get("speech_rate") is not None:
                style_dict["speech_rate"] = float(entry_data["speech_rate"])
            if entry_data.get("pitch_range") is not None:
                style_dict["pitch_range"] = float(entry_data["pitch_range"])
            style_override = StyleParams.from_dict(style_dict)

        entries.append(ScriptEntry(
            speaker=speaker,
            text=text,
            hint=hint,
            style_override=style_override,
            speed=speed,
        ))

    return Script(
        title=title,
        situation=situation,
        characters=characters,
        entries=entries,
    )


def script_to_yaml(script: Script) -> str:
    """Serialize a Script object to YAML string.

    Args:
        script: Script to serialize.

    Returns:
        YAML string.
    """
    data: dict = {}
    if script.title:
        data["title"] = script.title
    if script.situation:
        data["situation"] = script.situation

    if script.characters:
        chars = {}
        for cid, cp in script.characters.items():
            cd: dict = {"name": cp.name}
            if cp.personality:
                cd["personality"] = cp.personality
            if cp.voice_description:
                cd["voice_description"] = cp.voice_description
            if cp.speaker_file:
                cd["speaker_file"] = str(cp.speaker_file)
            if cp.language != "ja":
                cd["language"] = cp.language
            chars[cid] = cd
        data["characters"] = chars

    if script.entries:
        dialogue = []
        for entry in script.entries:
            ed: dict = {"speaker": entry.speaker, "text": entry.text}
            if entry.hint:
                ed["hint"] = entry.hint
            if entry.style_override:
                if entry.style_override.emotion != "neutral":
                    ed["emotion"] = entry.style_override.emotion
            dialogue.append(ed)
        data["dialogue"] = dialogue

    return yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
