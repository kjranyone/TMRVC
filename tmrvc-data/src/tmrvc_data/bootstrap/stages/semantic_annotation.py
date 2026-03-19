"""Stage 10: LLM-based semantic and acting annotation.

Uses Qwen3.5-9B (or compatible LLM) to generate:
- scene_summary: 1-sentence scene summary
- dialogue_intent: utterance intent (inform/request/comfort/scold/etc.)
- emotion_description: free-form emotion description
- acting_hint: free-form acting direction

Output is JSON-parsed; failures produce empty strings with confidence=0.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)

# System prompt for annotation LLM
_SYSTEM_PROMPT = """You are an expert voice acting director and speech analyst.
Given a transcript and optional context, produce a JSON object with exactly these keys:
- scene_summary: A single sentence describing the scene or context.
- dialogue_intent: The speaker's intent. Choose from: inform, request, comfort, scold, apologize, persuade, question, exclaim, narrate, greet, farewell, other.
- emotion_description: A brief description of the speaker's emotional state.
- acting_hint: A concise acting direction for a voice actor performing this line.

Respond ONLY with valid JSON. No markdown, no explanation."""

_USER_PROMPT_TEMPLATE = """Transcript: {transcript}
Language: {language}
Duration: {duration:.1f}s"""

# Canonical intent vocabulary
INTENT_VOCABULARY = frozenset({
    "inform", "request", "comfort", "scold", "apologize", "persuade",
    "question", "exclaim", "narrate", "greet", "farewell", "other",
})


class SemanticAnnotationStage:
    """LLM-based semantic annotation using Qwen3.5-9B.

    Lazy-loads the LLM on first use.  Falls back to heuristic
    annotation when no LLM backend is available.
    """

    def __init__(
        self,
        config: Optional[BootstrapConfig] = None,
        *,
        model_id: str = "Qwen/Qwen3.5-9B",
        batch_size: int = 8,
    ) -> None:
        self.config = config or BootstrapConfig()
        self.model_id = model_id
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._backend: Optional[str] = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Generate semantic annotations for all non-rejected utterances."""
        # Filter to annotate-able utterances
        to_annotate = [
            u for u in utterances
            if not u.is_rejected and u.text_transcript
        ]

        if to_annotate:
            # Process in batches
            for batch_start in range(0, len(to_annotate), self.batch_size):
                batch = to_annotate[batch_start:batch_start + self.batch_size]
                self._annotate_batch(batch)

        # Mark all utterances as completed
        for utt in utterances:
            if not utt.acting_annotations:
                utt.acting_annotations = {
                    "scene_summary": "",
                    "dialogue_intent": "",
                    "emotion_description": "",
                    "acting_hint": "",
                }
            utt.stage_completed = BootstrapStage.SEMANTIC_ANNOTATION

        logger.info(
            "SemanticAnnotation: annotated %d / %d utterances",
            len(to_annotate), len(utterances),
        )
        return utterances

    # ------------------------------------------------------------------
    # Batch annotation
    # ------------------------------------------------------------------

    def _annotate_batch(
        self, utterances: List[BootstrapUtterance],
    ) -> None:
        """Annotate a batch of utterances."""
        # Try LLM backends in order
        try:
            self._annotate_with_transformers(utterances)
            return
        except (ImportError, Exception) as exc:
            logger.debug("Transformers LLM unavailable: %s", exc)

        try:
            self._annotate_with_vllm(utterances)
            return
        except (ImportError, Exception) as exc:
            logger.debug("vLLM unavailable: %s", exc)

        # Fallback to heuristic annotation
        self._annotate_heuristic(utterances)

    def _annotate_with_transformers(
        self, utterances: List[BootstrapUtterance],
    ) -> None:
        """Annotate using HuggingFace transformers."""
        import torch

        if self._model is None or self._backend != "transformers":
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                device_map=self.config.device,
            )
            self._backend = "transformers"

        for utt in utterances:
            prompt = _USER_PROMPT_TEMPLATE.format(
                transcript=utt.text_transcript,
                language=utt.language or "unknown",
                duration=utt.duration_sec,
            )

            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]

            try:
                input_text = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                inputs = self._tokenizer(
                    input_text, return_tensors="pt",
                ).to(self._model.device)

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.3,
                        do_sample=True,
                        top_p=0.9,
                    )

                response = self._tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip()

                annotations = self._parse_json_response(response)
                utt.acting_annotations = annotations

            except Exception as exc:
                logger.warning(
                    "LLM annotation failed for %s: %s", utt.utterance_id, exc,
                )
                utt.acting_annotations = self._empty_annotations()
                utt.warnings.append(f"semantic_annotation_error:{exc}")

    def _annotate_with_vllm(
        self, utterances: List[BootstrapUtterance],
    ) -> None:
        """Annotate using vLLM for faster batch inference."""
        from vllm import LLM, SamplingParams

        if self._model is None or self._backend != "vllm":
            self._model = LLM(model=self.model_id)
            self._backend = "vllm"

        sampling_params = SamplingParams(
            temperature=0.3, top_p=0.9, max_tokens=256,
        )

        prompts = []
        for utt in utterances:
            prompt = _USER_PROMPT_TEMPLATE.format(
                transcript=utt.text_transcript,
                language=utt.language or "unknown",
                duration=utt.duration_sec,
            )
            full_prompt = f"<|system|>\n{_SYSTEM_PROMPT}\n<|user|>\n{prompt}\n<|assistant|>\n"
            prompts.append(full_prompt)

        outputs = self._model.generate(prompts, sampling_params)

        for utt, output in zip(utterances, outputs):
            response = output.outputs[0].text.strip()
            annotations = self._parse_json_response(response)
            utt.acting_annotations = annotations

    def _annotate_heuristic(
        self, utterances: List[BootstrapUtterance],
    ) -> None:
        """Heuristic annotation fallback when no LLM is available."""
        logger.info(
            "Using heuristic annotation (no LLM available). "
            "For better results, install transformers and download %s",
            self.model_id,
        )

        for utt in utterances:
            text = utt.text_transcript.strip()
            annotations = {
                "scene_summary": "",
                "dialogue_intent": self._guess_intent(text),
                "emotion_description": self._guess_emotion(text, utt.language),
                "acting_hint": "",
            }
            utt.acting_annotations = annotations

    # ------------------------------------------------------------------
    # Parsing and heuristics
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_response(response: str) -> Dict[str, str]:
        """Parse JSON from LLM response, with fallback extraction."""
        # Try direct JSON parse
        try:
            data = json.loads(response)
            return SemanticAnnotationStage._validate_annotations(data)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return SemanticAnnotationStage._validate_annotations(data)
            except json.JSONDecodeError:
                pass

        # Try extracting any JSON-like object
        brace_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if brace_match:
            try:
                data = json.loads(brace_match.group(0))
                return SemanticAnnotationStage._validate_annotations(data)
            except json.JSONDecodeError:
                pass

        logger.debug("Failed to parse LLM response as JSON: %s", response[:200])
        return SemanticAnnotationStage._empty_annotations()

    @staticmethod
    def _validate_annotations(data: Dict[str, Any]) -> Dict[str, str]:
        """Validate and normalise annotation fields."""
        result = {
            "scene_summary": str(data.get("scene_summary", "")),
            "dialogue_intent": str(data.get("dialogue_intent", "")),
            "emotion_description": str(data.get("emotion_description", "")),
            "acting_hint": str(data.get("acting_hint", "")),
        }

        # Normalize dialogue_intent to canonical vocabulary
        intent = result["dialogue_intent"].lower().strip()
        if intent not in INTENT_VOCABULARY:
            # Try to find closest match
            for canonical in INTENT_VOCABULARY:
                if canonical in intent or intent in canonical:
                    result["dialogue_intent"] = canonical
                    break
            else:
                result["dialogue_intent"] = "other"
        else:
            result["dialogue_intent"] = intent

        return result

    @staticmethod
    def _empty_annotations() -> Dict[str, str]:
        """Return empty annotation dict."""
        return {
            "scene_summary": "",
            "dialogue_intent": "",
            "emotion_description": "",
            "acting_hint": "",
        }

    @staticmethod
    def _guess_intent(text: str) -> str:
        """Heuristic intent detection from text."""
        if not text:
            return "other"

        # Question detection
        if text.endswith("?") or text.endswith("？"):
            return "question"
        if any(w in text.lower() for w in ["what", "why", "how", "when", "where", "who"]):
            return "question"
        if any(w in text for w in ["何", "なぜ", "どう", "いつ", "どこ", "誰"]):
            return "question"

        # Exclamation detection
        if text.endswith("!") or text.endswith("！"):
            return "exclaim"

        # Greeting detection
        greetings = ["hello", "hi", "hey", "こんにちは", "おはよう", "お疲れ"]
        if any(text.lower().startswith(g) for g in greetings):
            return "greet"

        # Farewell detection
        farewells = ["goodbye", "bye", "さよなら", "じゃあね", "また"]
        if any(text.lower().startswith(f) for f in farewells):
            return "farewell"

        # Apology detection
        apologies = ["sorry", "apologize", "すみません", "ごめん", "申し訳"]
        if any(a in text.lower() for a in apologies):
            return "apologize"

        # Request detection
        requests = ["please", "could you", "お願い", "ください", "してくれ"]
        if any(r in text.lower() for r in requests):
            return "request"

        return "inform"

    @staticmethod
    def _guess_emotion(text: str, language: str = "") -> str:
        """Heuristic emotion detection from text."""
        if not text:
            return ""

        text_lower = text.lower()

        # Simple keyword-based emotion detection
        emotion_keywords = {
            "happy": ["happy", "glad", "嬉しい", "楽しい", "幸せ", "よかった"],
            "sad": ["sad", "sorry", "悲しい", "寂しい", "辛い", "残念"],
            "angry": ["angry", "furious", "怒", "むかつく", "許せない"],
            "surprised": ["surprised", "amazing", "びっくり", "驚", "えっ"],
            "fearful": ["scared", "afraid", "怖い", "恐", "不安"],
            "calm": ["calm", "peaceful", "穏やか", "静か", "落ち着"],
            "excited": ["excited", "awesome", "やった", "すごい", "最高"],
        }

        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion

        return ""
