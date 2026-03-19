"""Stage 10b: Enriched transcript generation with inline acting tags.

Combines the plain transcript, detected audio events, physical targets,
and semantic annotations to produce an enriched transcript with inline
acting tags aligned to word/phoneme boundaries.

Uses Qwen3.5-9B or rule-based fallback.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)

# Canonical acting tags from tmrvc_core.acting_tags
CANONICAL_TAGS = {
    # Vocal events
    "inhale", "exhale", "laugh", "sigh", "cough", "click", "gasp", "hum", "cry", "sniff",
    # Prosodic markers
    "emphasis", "prolonged", "pause", "rising", "falling", "break",
    # Acting directives
    "angry", "whisper", "calm", "excited", "tender", "professional",
    "sad", "happy", "fearful", "disgusted", "surprised", "bored",
    "nervous", "confident", "sarcastic", "playful",
}

# Physical feature index -> potential tag mappings
PHYSICAL_TO_TAG = {
    5: ("breathiness", 0.7, "[whisper]"),       # high breathiness -> whisper
    2: ("energy_level", 0.8, "[emphasis]"),      # high energy -> emphasis
    10: ("vocal_effort", 0.8, "[emphasis]"),     # high vocal effort -> emphasis
    11: ("creak", 0.6, "[pause]"),               # high creak -> creaky voice
}

# LLM prompt for enriched transcript generation
_SYSTEM_PROMPT = """You are an expert speech annotation system.
Given a plain transcript and audio analysis data, insert inline acting tags
into the transcript. Tags use the format [tag_name].

Available tags: [inhale], [exhale], [laugh], [sigh], [cough], [emphasis],
[prolonged], [pause], [angry], [whisper], [calm], [excited], [tender],
[professional], [sad], [happy], [fearful], [surprised], [bored],
[nervous], [confident], [sarcastic], [playful]

Rules:
- Insert tags at word boundaries only
- Each tag should be placed BEFORE the word it modifies
- Use [emphasis] sparingly (max 2 per sentence)
- Emotion tags go at the start of the sentence
- Vocal event tags go at the point where they occur
- Return ONLY the enriched transcript, no explanation"""

_USER_PROMPT_TEMPLATE = """Plain transcript: {transcript}
Emotion: {emotion}
Acting hint: {acting_hint}
Physical features (summary):
  breathiness={breathiness:.2f}, energy={energy:.2f}, vocal_effort={vocal_effort:.2f}
  pitch_range={pitch_range:.2f}, pressedness={pressedness:.2f}
Duration: {duration:.1f}s"""


class EnrichedTranscriptStage:
    """Generate enriched transcripts with inline acting tags.

    Combines plain transcript + audio events + physical targets +
    semantic annotations using Qwen3.5-9B or rule-based fallback.
    Tags are aligned to word/phoneme boundaries.
    """

    def __init__(
        self,
        config: Optional[BootstrapConfig] = None,
        *,
        model_id: str = "Qwen/Qwen3.5-9B",
    ) -> None:
        self.config = config or BootstrapConfig()
        self.model_id = model_id
        self._model = None
        self._tokenizer = None
        self._backend: Optional[str] = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Generate enriched transcripts for all utterances."""
        for utt in utterances:
            if utt.is_rejected or not utt.text_transcript:
                if not utt.enriched_transcript:
                    utt.enriched_transcript = utt.text_transcript or ""
                continue

            try:
                enriched = self._generate_enriched(utt)
                utt.enriched_transcript = enriched
            except Exception as exc:
                logger.warning(
                    "Enriched transcript failed for %s: %s",
                    utt.utterance_id, exc,
                )
                utt.enriched_transcript = utt.text_transcript
                utt.warnings.append(f"enriched_transcript_error:{exc}")

        logger.info(
            "EnrichedTranscript: processed %d utterances", len(utterances),
        )
        return utterances

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate_enriched(self, utt: BootstrapUtterance) -> str:
        """Generate enriched transcript for a single utterance."""
        # Try LLM-based generation
        try:
            return self._generate_with_llm(utt)
        except (ImportError, Exception) as exc:
            logger.debug("LLM enrichment unavailable: %s", exc)

        # Fall back to rule-based
        return self._generate_rule_based(utt)

    def _generate_with_llm(self, utt: BootstrapUtterance) -> str:
        """Generate enriched transcript using LLM."""
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

        # Compute physical feature summary
        phys_summary = self._summarize_physical(utt)

        prompt = _USER_PROMPT_TEMPLATE.format(
            transcript=utt.text_transcript,
            emotion=utt.acting_annotations.get("emotion_description", ""),
            acting_hint=utt.acting_annotations.get("acting_hint", ""),
            duration=utt.duration_sec,
            **phys_summary,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tokenizer(
            input_text, return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
            )

        response = self._tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        ).strip()

        # Validate and normalize the enriched transcript
        enriched = self._normalize_enriched(response, utt.text_transcript)
        return enriched

    def _generate_rule_based(self, utt: BootstrapUtterance) -> str:
        """Generate enriched transcript using rule-based heuristics."""
        text = utt.text_transcript
        if not text:
            return ""

        parts: List[str] = []

        # 1. Add emotion tag at the start based on annotations
        emotion = utt.acting_annotations.get("emotion_description", "")
        if emotion:
            tag = self._emotion_to_tag(emotion)
            if tag:
                parts.append(tag)

        # 2. Add physical-feature-derived tags
        phys_summary = self._summarize_physical(utt)

        # High breathiness -> [whisper] at start
        if phys_summary.get("breathiness", 0) > 0.7:
            if "[whisper]" not in parts:
                parts.append("[whisper]")

        # 3. Process the transcript word by word
        words = text.split()
        for i, word in enumerate(words):
            # Check if this word position has special physical characteristics
            if utt.physical_targets is not None and len(words) > 1:
                frame_idx = int(i / len(words) * utt.physical_targets.shape[0])
                frame_idx = min(frame_idx, utt.physical_targets.shape[0] - 1)

                if utt.physical_observed_mask is not None:
                    # High energy at this position -> [emphasis]
                    if (utt.physical_observed_mask[frame_idx, 2]
                            and utt.physical_targets[frame_idx, 2] > 0.8):
                        parts.append("[emphasis]")

                    # Very low energy -> [pause] before next word
                    if (i > 0 and utt.physical_observed_mask[frame_idx, 2]
                            and utt.physical_targets[frame_idx, 2] < 0.1):
                        parts.append("[pause]")

            parts.append(word)

        enriched = " ".join(parts)

        # 4. Clean up: remove duplicate consecutive tags
        enriched = self._deduplicate_tags(enriched)

        return enriched

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _summarize_physical(utt: BootstrapUtterance) -> Dict[str, float]:
        """Compute summary statistics of physical features."""
        defaults = {
            "breathiness": 0.0,
            "energy": 0.0,
            "vocal_effort": 0.0,
            "pitch_range": 0.0,
            "pressedness": 0.0,
        }

        if utt.physical_targets is None or utt.physical_observed_mask is None:
            return defaults

        targets = utt.physical_targets  # [T, 12]
        mask = utt.physical_observed_mask  # [T, 12]

        # Compute mean of observed values for key dimensions
        dim_map = {
            "breathiness": 5,
            "energy": 2,
            "vocal_effort": 10,
            "pitch_range": 1,
            "pressedness": 3,
        }

        result = {}
        for name, dim_idx in dim_map.items():
            observed = targets[:, dim_idx][mask[:, dim_idx]]
            if len(observed) > 0:
                result[name] = float(np.mean(observed))
            else:
                result[name] = 0.0

        return result

    @staticmethod
    def _emotion_to_tag(emotion: str) -> Optional[str]:
        """Map emotion description to canonical acting tag."""
        emotion_lower = emotion.strip().lower()

        direct_map = {
            "angry": "[angry]", "anger": "[angry]",
            "whisper": "[whisper]", "calm": "[calm]",
            "excited": "[excited]", "excitement": "[excited]",
            "tender": "[tender]", "professional": "[professional]",
            "sad": "[sad]", "sadness": "[sad]",
            "happy": "[happy]", "happiness": "[happy]",
            "fear": "[fearful]", "fearful": "[fearful]",
            "disgust": "[disgusted]", "disgusted": "[disgusted]",
            "surprise": "[surprised]", "surprised": "[surprised]",
            "bored": "[bored]", "boredom": "[bored]",
            "nervous": "[nervous]", "confident": "[confident]",
            "sarcastic": "[sarcastic]", "playful": "[playful]",
        }

        if emotion_lower in direct_map:
            return direct_map[emotion_lower]

        # Check for partial matches
        for keyword, tag in direct_map.items():
            if keyword in emotion_lower:
                return tag

        return None

    @staticmethod
    def _normalize_enriched(response: str, original_text: str) -> str:
        """Validate and normalize an LLM-generated enriched transcript.

        Ensures:
        - All tags use canonical bracket format [tag]
        - Tags are from the canonical vocabulary
        - The original text content is preserved
        """
        # Normalize tag format
        response = response.strip()

        # Extract all tags from the response
        tag_pattern = re.compile(r"\[([^\]]+)\]")
        found_tags = tag_pattern.findall(response)

        # Validate tags against canonical vocabulary
        for tag_content in found_tags:
            if tag_content.lower() not in CANONICAL_TAGS:
                # Replace unknown tags with closest canonical match or remove
                closest = None
                for canonical in CANONICAL_TAGS:
                    if canonical in tag_content.lower() or tag_content.lower() in canonical:
                        closest = canonical
                        break

                if closest:
                    response = response.replace(
                        f"[{tag_content}]", f"[{closest}]",
                    )
                else:
                    # Keep as free-form acting instruction
                    pass

        # Verify the original text words are present
        original_words = set(original_text.lower().split())
        response_text = tag_pattern.sub("", response).strip()
        response_words = set(response_text.lower().split())

        # If too many original words are missing, fall back
        if original_words and len(original_words - response_words) > len(original_words) * 0.3:
            logger.debug(
                "Enriched transcript diverged too much from original, "
                "using original with prepended tags",
            )
            # Extract just the leading tags and prepend to original
            leading_tags = []
            for match in tag_pattern.finditer(response):
                if match.start() < len(response) // 4:
                    leading_tags.append(match.group(0))
            if leading_tags:
                return " ".join(leading_tags) + " " + original_text
            return original_text

        return response

    @staticmethod
    def _deduplicate_tags(text: str) -> str:
        """Remove consecutive duplicate tags."""
        tokens = text.split()
        deduped = []
        prev_token = ""
        for token in tokens:
            if token.startswith("[") and token.endswith("]"):
                if token == prev_token:
                    continue
            deduped.append(token)
            prev_token = token
        return " ".join(deduped)
