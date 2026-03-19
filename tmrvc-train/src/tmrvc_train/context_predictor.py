"""ContextStylePredictor: LLM-based emotion/style inference from dialogue context.

Uses the open-weight Qwen LLM backend to predict appropriate emotion and style
parameters based on character profile, conversation history, and the next
utterance text.

Model selection (from track_architecture.md SS5a):
  Primary:  Qwen/Qwen3.5-35B-A3B (MoE, 3B active)
  Fallback: Qwen/Qwen3.5-4B (dense)

Falls back to rule-based heuristics when no GPU or model is available.
"""

from __future__ import annotations

import asyncio
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
次の発話に最適な音響パラメータを JSON で出力してください。

出力は以下の JSON 形式のみで、余計な説明は不要です:
{
  "emotion": "<12カテゴリから1つ>",
  "breathiness": <0.0 ~ 1.0>,
  "tension": <0.0 ~ 1.0>,
  "arousal": <0.0 ~ 1.0>,
  "valence": <-1.0 ~ 1.0>,
  "roughness": <0.0 ~ 1.0>,
  "voicing": <0.0 ~ 1.0>,
  "energy": <0.0 ~ 1.0>,
  "speech_rate": <0.5 ~ 2.0>,
  "reasoning": "<推論の根拠を簡潔に>"
}

感情カテゴリ: happy, sad, angry, fearful, surprised, disgusted, neutral, bored, excited, tender, sarcastic, whisper"""

# Rule-based punctuation/keyword → style adjustment mapping
_PUNCTUATION_RULES: list[tuple[str, dict[str, Any]]] = [
    ("！！", {"arousal": 0.5, "energy": 0.4, "tension": 0.3}),
    ("！", {"arousal": 0.3, "energy": 0.2}),
    ("？？", {"arousal": 0.2, "tension": 0.1}),
    ("？", {"arousal": 0.1}),
    ("…", {"arousal": -0.2, "speech_rate": 0.8, "energy": -0.2}),
    ("。。。", {"arousal": -0.2, "speech_rate": 0.7, "energy": -0.3}),
    ("〜", {"valence": 0.1, "tension": -0.1}),
    ("♪", {"valence": 0.4, "arousal": 0.2, "energy": 0.1}),
]

_KEYWORD_RULES: list[tuple[str, dict[str, Any]]] = [
    ("ありがとう", {"emotion": "happy", "valence": 0.5, "tension": -0.2}),
    ("ごめん", {"emotion": "sad", "valence": -0.3, "arousal": -0.1}),
    ("すみません", {"emotion": "sad", "valence": -0.2}),
    ("怒", {"emotion": "angry", "arousal": 0.5, "tension": 0.6}),
    ("悲し", {"emotion": "sad", "valence": -0.5, "energy": -0.3}),
    ("嬉し", {"emotion": "happy", "valence": 0.6, "arousal": 0.3}),
    ("楽し", {"emotion": "happy", "valence": 0.5, "arousal": 0.4}),
    ("怖", {"emotion": "fearful", "arousal": 0.3, "tension": 0.4}),
    ("驚", {"emotion": "surprised", "arousal": 0.4, "tension": 0.2}),
    ("えっ", {"emotion": "surprised", "arousal": 0.3}),
    ("はぁ", {"emotion": "bored", "arousal": -0.3, "energy": -0.3}),
    ("ふふ", {"emotion": "happy", "valence": 0.3, "breathiness": 0.3, "energy": -0.2}),
    ("うう", {"emotion": "sad", "valence": -0.4, "tension": 0.2}),
    ("ひそひそ", {"emotion": "whisper", "breathiness": 0.7, "energy": -0.5, "voicing": 0.3}),
    ("内緒", {"emotion": "whisper", "breathiness": 0.6, "energy": -0.4, "voicing": 0.4}),
]


class ContextStylePredictor:
    """Predict emotion/style from dialogue context using Qwen LLM backend.

    Uses the open-weight Qwen models via the LLMBackend from tmrvc-serve.
    Falls back to rule-based heuristics when no GPU or model is available.

    Args:
        model: Qwen model name to use.
        max_history: Maximum dialogue turns to include in context.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-35B-A3B",
        max_history: int = 10,
    ) -> None:
        self.model = model
        self.max_history = max_history
        self._backend: Any | None = None
        self._backend_loaded: bool = False

    def _get_backend(self) -> Any:
        """Lazily create and load the LLMBackend."""
        if self._backend is None:
            from tmrvc_serve.llm_backend import LLMBackend

            self._backend = LLMBackend(model_name=self.model)
            self._backend_loaded = self._backend.load()
        return self._backend

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

    def _generate(self, user_prompt: str) -> str:
        """Generate a response using the LLM backend.

        Prepends the system prompt to the user prompt so the backend
        receives full context regardless of its internal system prompt.
        """
        backend = self._get_backend()
        full_prompt = f"{_SYSTEM_PROMPT}\n\n{user_prompt}"
        return backend.generate(full_prompt, max_tokens=256, temperature=0.0)

    async def predict(
        self,
        character: CharacterProfile,
        history: list[DialogueTurn],
        next_text: str,
        situation: str | None = None,
    ) -> StyleParams:
        """Predict style from context using Qwen LLM backend (async).

        Falls back to rule-based prediction when the model is unavailable.
        """
        try:
            backend = self._get_backend()
            if backend.backend_type == "rule_based":
                logger.debug("LLM backend is rule-based, using rule-based fallback")
                return self.predict_rule_based(next_text, character)

            user_prompt = self._build_user_prompt(
                character, history, next_text, situation,
            )

            # Run synchronous generation in a thread to avoid blocking
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(
                None, self._generate, user_prompt,
            )

            return self._parse_response(response_text)

        except Exception as e:
            logger.warning("LLM generation failed, using rule-based fallback: %s", e)
            return self.predict_rule_based(next_text, character)

    def predict_sync(
        self,
        character: CharacterProfile,
        history: list[DialogueTurn],
        next_text: str,
        situation: str | None = None,
    ) -> StyleParams:
        """Predict style from context using Qwen LLM backend (sync).

        Falls back to rule-based prediction when the model is unavailable.
        """
        try:
            backend = self._get_backend()
            if backend.backend_type == "rule_based":
                logger.debug("LLM backend is rule-based, using rule-based fallback")
                return self.predict_rule_based(next_text, character)

            user_prompt = self._build_user_prompt(
                character, history, next_text, situation,
            )

            response_text = self._generate(user_prompt)
            return self._parse_response(response_text)

        except Exception as e:
            logger.warning("LLM generation failed, using rule-based fallback: %s", e)
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
        """Predict style using rule-based heuristics (no LLM needed).

        Analyzes punctuation, keywords, and character defaults to estimate
        appropriate emotion parameters.
        """
        params: dict[str, Any] = {
            "emotion": "neutral",
            "breathiness": 0.0,
            "tension": 0.0,
            "arousal": 0.0,
            "valence": 0.0,
            "roughness": 0.0,
            "voicing": 1.0,
            "energy": 0.0,
            "speech_rate": 1.0,
            "reasoning": "rule-based",
        }

        # Apply character defaults
        if character and character.default_style:
            ds = character.default_style
            params["breathiness"] = ds.breathiness
            params["tension"] = ds.tension
            params["arousal"] = ds.arousal
            params["valence"] = ds.valence
            params["roughness"] = ds.roughness
            params["voicing"] = ds.voicing
            params["energy"] = ds.energy
            params["speech_rate"] = ds.speech_rate

        # Apply keyword rules (first match sets emotion)
        emotion_set = False
        for keyword, adjustments in _KEYWORD_RULES:
            if keyword in text:
                for key, value in adjustments.items():
                    if key == "emotion" and not emotion_set:
                        params["emotion"] = value
                        emotion_set = True
                    elif key != "emotion":
                        if key == "speech_rate":
                            params[key] = _clamp(params[key] * (1.0 + (value - 1.0)), 0.5, 2.0)
                        elif key == "voicing" and params["emotion"] == "whisper":
                            params[key] = value # Force direct value
                        else:
                            params[key] = _clamp(params[key] + value, -1.0, 1.0)

        # Apply punctuation rules (additive)
        for pattern, adjustments in _PUNCTUATION_RULES:
            if pattern in text:
                for key, value in adjustments.items():
                    if key != "emotion":
                        if key == "speech_rate":
                            params[key] = _clamp(params[key] * (1.0 + (value - 1.0)), 0.5, 2.0)
                        else:
                            params[key] = _clamp(params[key] + value, -1.0, 1.0)

        return StyleParams.from_dict(params)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
