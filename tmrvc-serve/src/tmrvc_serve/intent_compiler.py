
"""Intent Compiler for expressive speech synthesis.

Converts raw prompt/tags into canonical IntentCompilerOutput with 12-D physical targets.

Uses the open-weight LLM backend (Qwen/Qwen3.5-35B-A3B primary,
Qwen/Qwen3.5-4B fallback) for intent compilation, with rule-based
fallback when no GPU is available.

See: plan/track_serving.md SS8
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import List, Optional

import torch

from tmrvc_core.types import (
    ActingTextureMacro,
    IntentCompilerOutput,
    PacingControls,
)
from tmrvc_serve.llm_backend import LLMBackend

logger = logging.getLogger(__name__)


class IntentCompiler:
    """Compiles high-level intentions into low-level model controls.

    Backed by an open-weight LLM (Qwen3.5) with deterministic inference,
    falling back to rule-based compilation when no GPU is available.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        use_vllm: bool = False,
        seed: int = 42,
    ):
        self._llm = LLMBackend(
            model_name=model_name,
            device=device,
            use_vllm=use_vllm,
            seed=seed,
        )
        self._initialized = False

    def ensure_loaded(self) -> None:
        """Load the LLM backend if not already loaded."""
        if not self._initialized:
            self._llm.load()
            self._initialized = True

    def compile(
        self,
        prompt: str,
        context: Optional[dict] = None,
    ) -> IntentCompilerOutput:
        """Compile a natural language prompt into an Intent record.

        Args:
            prompt: Natural language acting instruction or tagged prompt.
            context: Optional context dict (scene, character, history).

        Returns:
            Canonical IntentCompilerOutput with 12-D physical targets,
            acting macro controls, pacing, inline tags, and provenance.
        """
        self.ensure_loaded()
        compile_id = str(uuid.uuid4())

        # Build the LLM prompt with context
        llm_prompt = self._build_prompt(prompt, context)

        # Generate via LLM (or rule-based fallback)
        raw_output = self._llm.generate(llm_prompt, max_tokens=512, temperature=0.0)

        # Parse LLM JSON output into structured fields
        parsed = self._parse_llm_output(raw_output)

        # Build physical targets tensor [1, 12]
        physical_values = parsed.get("physical_targets", [0.5] * 12)
        if len(physical_values) != 12:
            physical_values = (physical_values + [0.5] * 12)[:12]
        physical_targets = torch.tensor([physical_values], dtype=torch.float32)

        # Build acting macro
        macro_data = parsed.get("acting_macro", {})
        acting_macro = ActingTextureMacro(
            intensity=_clamp01(macro_data.get("intensity", 0.5)),
            instability=_clamp01(macro_data.get("instability", 0.2)),
            tenderness=_clamp01(macro_data.get("tenderness", 0.3)),
            tension=_clamp01(macro_data.get("tension", 0.3)),
            spontaneity=_clamp01(macro_data.get("spontaneity", 0.5)),
            reference_mix=_clamp01(macro_data.get("reference_mix", 0.0)),
        )

        # Build pacing controls
        pacing_data = parsed.get("pacing", {})
        pacing = PacingControls(
            pace=_clamp(pacing_data.get("pace", 1.0), 0.3, 3.0),
            hold_bias=_clamp(pacing_data.get("hold_bias", 0.0), -1.0, 1.0),
            boundary_bias=_clamp(pacing_data.get("boundary_bias", 0.0), -1.0, 1.0),
            phrase_pressure=_clamp(pacing_data.get("phrase_pressure", 0.0), -1.0, 1.0),
            breath_tendency=_clamp(pacing_data.get("breath_tendency", 0.0), -1.0, 1.0),
        )

        # Inline tags and warnings
        inline_tags = parsed.get("inline_tags", [])
        warnings = parsed.get("warnings", [])

        logger.info(
            "Compiled prompt [%s] -> ID: %s (backend=%s)",
            prompt[:80],
            compile_id,
            self._llm.backend_type,
        )

        return IntentCompilerOutput(
            compile_id=compile_id,
            source_prompt=prompt,
            inline_tags=inline_tags,
            physical_targets=physical_targets,
            acting_macro=acting_macro,
            pacing=pacing,
            warnings=warnings,
            provenance=self._llm.provenance,
            metadata={
                "compiler_version": "2.0.0",
                "model_version": self._llm.provenance,
                "backend_type": self._llm.backend_type,
                "context": context or {},
            },
        )

    def _build_prompt(self, prompt: str, context: Optional[dict] = None) -> str:
        """Build the full prompt for the LLM including any context."""
        parts: list[str] = []

        if context:
            if context.get("scene"):
                parts.append(f"Scene context: {context['scene']}")
            if context.get("character"):
                parts.append(f"Character: {context['character']}")
            if context.get("history"):
                parts.append(f"Recent dialogue: {context['history']}")

        parts.append(f"Acting instruction: {prompt}")
        return "\n".join(parts)

    def _parse_llm_output(self, raw: str) -> dict:
        """Parse the LLM JSON output, tolerating markdown fences and extras.

        Returns a dict with keys: physical_targets, acting_macro, pacing,
        inline_tags, warnings.  On parse failure, returns safe defaults.
        """
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```\s*$", "", cleaned)

        # Try to extract JSON object
        json_match = re.search(r"\{[\s\S]*\}", cleaned)
        if not json_match:
            logger.warning("No JSON found in LLM output: %s", raw[:200])
            return self._safe_defaults()

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            logger.warning("Invalid JSON in LLM output: %s", raw[:200])
            return self._safe_defaults()

        # Validate expected structure
        result: dict = {}

        # physical_targets: list of 12 floats
        pt = data.get("physical_targets")
        if isinstance(pt, list) and len(pt) == 12:
            result["physical_targets"] = [_clamp01(float(v)) for v in pt]
        else:
            result["physical_targets"] = [0.5] * 12

        # acting_macro: dict of 6 floats
        am = data.get("acting_macro")
        if isinstance(am, dict):
            result["acting_macro"] = am
        else:
            result["acting_macro"] = {}

        # pacing: dict
        pac = data.get("pacing")
        if isinstance(pac, dict):
            result["pacing"] = pac
        else:
            result["pacing"] = {}

        # inline_tags: list of strings
        tags = data.get("inline_tags")
        if isinstance(tags, list):
            result["inline_tags"] = [str(t) for t in tags]
        else:
            result["inline_tags"] = []

        # warnings: list of strings
        warns = data.get("warnings")
        if isinstance(warns, list):
            result["warnings"] = [str(w) for w in warns]
        else:
            result["warnings"] = []

        return result

    @staticmethod
    def _safe_defaults() -> dict:
        """Return safe default values when LLM output cannot be parsed."""
        return {
            "physical_targets": [0.5] * 12,
            "acting_macro": {},
            "pacing": {},
            "inline_tags": [],
            "warnings": ["COMPILATION_FAILED: LLM output could not be parsed, using defaults"],
        }


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
