"""ContextStylePredictor: LLM-based emotion/style inference from dialogue context.

Uses Claude API to predict appropriate emotion and style parameters
based on character profile, conversation history, and the next utterance text.

Falls back to rule-based heuristics when API is unavailable.

Usage::

    predictor = ContextStylePredictor(api_key="sk-ant-...")
    style = await predictor.predict(character, history, "ゆきちゃん！久しぶり！")

    # Sync version
    style = predictor.predict_sync(character, history, "ゆきちゃん！久しぶり！")

    # Rule-based fallback (no API needed)
    style = predictor.predict_rule_based("えっ！？本当に？")
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tmrvc_core.dialogue_types import (
    EMOTION_CATEGORIES,
    CharacterProfile,
    DialogueTurn,
    StyleParams,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
あなたは音声合成システムの感情コントローラーです。
以下のキャラクターと会話コンテキストを踏まえ、
次の発話に最適な感情パラメータを JSON で出力してください。

出力は以下の JSON 形式のみで、余計な説明は不要です:
{
  "emotion": "<12カテゴリから1つ>",
  "valence": <-1.0 ~ 1.0>,
  "arousal": <-1.0 ~ 1.0>,
  "dominance": <-1.0 ~ 1.0>,
  "speech_rate": <-1.0 ~ 1.0>,
  "energy": <-1.0 ~ 1.0>,
  "pitch_range": <-1.0 ~ 1.0>,
  "reasoning": "<推論の根拠を簡潔に>"
}

感情カテゴリ: happy, sad, angry, fearful, surprised, disgusted, neutral, bored, excited, tender, sarcastic, whisper"""

# Rule-based punctuation/keyword → style adjustment mapping
_PUNCTUATION_RULES: list[tuple[str, dict[str, Any]]] = [
    ("！！", {"arousal": 0.5, "energy": 0.4, "pitch_range": 0.3}),
    ("！", {"arousal": 0.3, "energy": 0.2}),
    ("？？", {"pitch_range": 0.4, "arousal": 0.2}),
    ("？", {"pitch_range": 0.3}),
    ("…", {"arousal": -0.2, "speech_rate": -0.3}),
    ("。。。", {"arousal": -0.2, "speech_rate": -0.3}),
    ("〜", {"pitch_range": 0.2, "valence": 0.1}),
    ("♪", {"valence": 0.4, "arousal": 0.2}),
]

_KEYWORD_RULES: list[tuple[str, dict[str, Any]]] = [
    ("ありがとう", {"emotion": "happy", "valence": 0.5}),
    ("ごめん", {"emotion": "sad", "valence": -0.3}),
    ("すみません", {"emotion": "sad", "valence": -0.2}),
    ("怒", {"emotion": "angry", "arousal": 0.5}),
    ("悲し", {"emotion": "sad", "valence": -0.5}),
    ("嬉し", {"emotion": "happy", "valence": 0.6}),
    ("楽し", {"emotion": "happy", "valence": 0.5}),
    ("怖", {"emotion": "fearful", "arousal": 0.3}),
    ("驚", {"emotion": "surprised", "arousal": 0.4}),
    ("えっ", {"emotion": "surprised", "arousal": 0.3}),
    ("はぁ", {"emotion": "bored", "arousal": -0.3}),
    ("ふふ", {"emotion": "happy", "valence": 0.3, "energy": -0.2}),
    ("うう", {"emotion": "sad", "valence": -0.4}),
    ("ひそひそ", {"emotion": "whisper", "energy": -0.5}),
    ("内緒", {"emotion": "whisper", "energy": -0.4}),
]


class ContextStylePredictor:
    """Predict emotion/style from dialogue context using Claude API.

    Args:
        api_key: Anthropic API key. If None, only rule-based fallback is available.
        model: Claude model to use.
        max_history: Maximum dialogue turns to include in context.
        timeout: API request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5-20251001",
        max_history: int = 10,
        timeout: float = 10.0,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_history = max_history
        self.timeout = timeout
        self._client: Any | None = None

    def _get_client(self) -> Any:
        """Lazily create Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package is required for ContextStylePredictor. "
                    "Install with: pip install anthropic"
                ) from e
            self._client = anthropic.Anthropic(
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def _build_user_prompt(
        self,
        character: CharacterProfile,
        history: list[DialogueTurn],
        next_text: str,
        situation: str | None = None,
    ) -> str:
        """Build the user prompt for the LLM."""
        parts: list[str] = []

        parts.append(f"## キャラクター\n名前: {character.name}")
        if character.personality:
            parts.append(f"性格: {character.personality}")
        if character.voice_description:
            parts.append(f"声の特徴: {character.voice_description}")

        if situation:
            parts.append(f"\n## 状況\n{situation}")

        if history:
            recent = history[-self.max_history :]
            lines = []
            for turn in recent:
                emotion_tag = f" [{turn.emotion}]" if turn.emotion else ""
                lines.append(f"{turn.speaker}{emotion_tag}: {turn.text}")
            parts.append(f"\n## 会話履歴\n" + "\n".join(lines))

        parts.append(f'\n## 次の発話\n「{next_text}」')

        return "\n".join(parts)

    async def predict(
        self,
        character: CharacterProfile,
        history: list[DialogueTurn],
        next_text: str,
        situation: str | None = None,
    ) -> StyleParams:
        """Predict style from context using Claude API (async).

        Falls back to rule-based prediction on API failure.
        """
        if self.api_key is None:
            logger.debug("No API key, using rule-based fallback")
            return self.predict_rule_based(next_text, character)

        try:
            client = self._get_client()
            user_prompt = self._build_user_prompt(
                character, history, next_text, situation,
            )

            # Use async client if available
            try:
                import anthropic
                async_client = anthropic.AsyncAnthropic(
                    api_key=self.api_key,
                    timeout=self.timeout,
                )
                response = await async_client.messages.create(
                    model=self.model,
                    max_tokens=256,
                    system=_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            except Exception:
                # Fallback to sync in async context
                return self.predict_sync(character, history, next_text, situation)

            return self._parse_response(response.content[0].text)

        except Exception as e:
            logger.warning("API call failed, using rule-based fallback: %s", e)
            return self.predict_rule_based(next_text, character)

    def predict_sync(
        self,
        character: CharacterProfile,
        history: list[DialogueTurn],
        next_text: str,
        situation: str | None = None,
    ) -> StyleParams:
        """Predict style from context using Claude API (sync).

        Falls back to rule-based prediction on API failure.
        """
        if self.api_key is None:
            logger.debug("No API key, using rule-based fallback")
            return self.predict_rule_based(next_text, character)

        try:
            client = self._get_client()
            user_prompt = self._build_user_prompt(
                character, history, next_text, situation,
            )

            response = client.messages.create(
                model=self.model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            return self._parse_response(response.content[0].text)

        except Exception as e:
            logger.warning("API call failed, using rule-based fallback: %s", e)
            return self.predict_rule_based(next_text, character)

    def _parse_response(self, text: str) -> StyleParams:
        """Parse LLM JSON response into StyleParams."""
        # Extract JSON from response (handle markdown code blocks)
        json_match = re.search(r"\{[^}]+\}", text, re.DOTALL)
        if not json_match:
            logger.warning("No JSON found in LLM response: %s", text[:200])
            return StyleParams.neutral()

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in LLM response: %s", text[:200])
            return StyleParams.neutral()

        # Validate emotion category
        emotion = data.get("emotion", "neutral")
        if emotion not in EMOTION_CATEGORIES:
            logger.warning("Unknown emotion '%s', defaulting to neutral", emotion)
            emotion = "neutral"
        data["emotion"] = emotion

        return StyleParams.from_dict(data)

    def predict_rule_based(
        self,
        text: str,
        character: CharacterProfile | None = None,
    ) -> StyleParams:
        """Predict style using rule-based heuristics (no API needed).

        Analyzes punctuation, keywords, and character defaults to estimate
        appropriate emotion parameters.
        """
        params: dict[str, Any] = {
            "emotion": "neutral",
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.0,
            "speech_rate": 0.0,
            "energy": 0.0,
            "pitch_range": 0.0,
            "reasoning": "rule-based",
        }

        # Apply character defaults
        if character and character.default_style:
            ds = character.default_style
            params["valence"] = ds.valence
            params["arousal"] = ds.arousal
            params["dominance"] = ds.dominance
            params["speech_rate"] = ds.speech_rate
            params["energy"] = ds.energy
            params["pitch_range"] = ds.pitch_range

        # Apply keyword rules (first match sets emotion)
        emotion_set = False
        for keyword, adjustments in _KEYWORD_RULES:
            if keyword in text:
                for key, value in adjustments.items():
                    if key == "emotion" and not emotion_set:
                        params["emotion"] = value
                        emotion_set = True
                    elif key != "emotion":
                        params[key] = _clamp(params[key] + value, -1.0, 1.0)

        # Apply punctuation rules (additive)
        for pattern, adjustments in _PUNCTUATION_RULES:
            if pattern in text:
                for key, value in adjustments.items():
                    if key != "emotion":
                        params[key] = _clamp(params[key] + value, -1.0, 1.0)

        return StyleParams.from_dict(params)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
