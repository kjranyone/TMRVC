"""Open-weight LLM backend for Intent Compiler.

Replaces the Claude API dependency with local open-weight inference.

Model selection (from track_architecture.md SS5a):
- Primary: Qwen/Qwen3.5-35B-A3B (MoE, 3B active)
- Fallback: Qwen/Qwen3.5-4B (dense)

Requirements:
- Deterministic output for a given model + prompt (temperature=0, seed fixed)
- Model version recorded in IntentCompilerOutput.provenance
- Rule-based fallback when no GPU available
"""

from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Model configuration
PRIMARY_MODEL = "Qwen/Qwen3.5-35B-A3B"
FALLBACK_MODEL = "Qwen/Qwen3.5-4B"

# Intent compiler system prompt
INTENT_COMPILER_SYSTEM_PROMPT = """You are an Intent Compiler for a programmable expressive speech engine.

Given a natural language acting instruction, you must output a structured JSON response with:
1. physical_targets: 12-D physical voice control values [0.0-1.0] for:
   pitch_level, pitch_range, energy_level, pressedness, spectral_tilt, breathiness,
   voice_irregularity, openness, aperiodicity, formant_shift, vocal_effort, creak
2. acting_macro: 6 macro acting controls [0.0-1.0] for:
   intensity, instability, tenderness, tension, spontaneity, reference_mix
3. pacing: pace (0.3-3.0), hold_bias (-1.0 to 1.0), boundary_bias (-1.0 to 1.0),
   phrase_pressure (-1.0 to 1.0), breath_tendency (-1.0 to 1.0)
4. inline_tags: list of acting tags to inject into the transcript
5. warnings: list of ambiguity warnings

Output ONLY valid JSON. No explanation."""


class LLMBackend:
    """Unified LLM backend for the Intent Compiler.

    Supports:
    - Local inference via transformers/vllm
    - Rule-based fallback (no GPU)
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        use_vllm: bool = False,
        seed: int = 42,
    ):
        self.model_name = model_name or PRIMARY_MODEL
        self.device = device
        self.use_vllm = use_vllm
        self.seed = seed
        self._model = None
        self._tokenizer = None
        self._backend_type: str = "none"  # none, transformers, vllm, rule_based

    def load(self) -> bool:
        """Attempt to load the LLM model.

        Returns True if model loaded, False if falling back to rules.
        """
        # Try vllm first if requested
        if self.use_vllm:
            if self._try_load_vllm():
                return True

        # Try transformers
        if self._try_load_transformers():
            return True

        # Try fallback model with transformers
        if self.model_name != FALLBACK_MODEL:
            logger.info(
                "Primary model %s unavailable, trying fallback %s",
                self.model_name,
                FALLBACK_MODEL,
            )
            original = self.model_name
            self.model_name = FALLBACK_MODEL
            if self._try_load_transformers():
                return True
            self.model_name = original

        # Fallback to rule-based
        logger.warning(
            "No GPU or model available. Using rule-based fallback for Intent Compiler."
        )
        self._backend_type = "rule_based"
        return False

    def _try_load_vllm(self) -> bool:
        """Try loading model with vllm for efficient inference."""
        try:
            from vllm import LLM, SamplingParams  # noqa: F401

            self._model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                seed=self.seed,
                dtype="auto",
            )
            self._backend_type = "vllm"
            logger.info("Loaded %s via vllm", self.model_name)
            return True
        except Exception as e:
            logger.info("vllm not available: %s", e)
            return False

    def _try_load_transformers(self) -> bool:
        """Try loading model with transformers."""
        try:
            import torch

            if not torch.cuda.is_available():
                logger.info("No CUDA available for transformers backend")
                return False

            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self._model.eval()
            self._backend_type = "transformers"
            logger.info("Loaded %s via transformers", self.model_name)
            return True
        except Exception as e:
            logger.info("transformers loading failed: %s", e)
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            prompt: User prompt text
            max_tokens: Maximum output tokens
            temperature: Sampling temperature (0 = deterministic)

        Returns:
            Generated text response
        """
        if self._backend_type == "vllm":
            return self._generate_vllm(prompt, max_tokens, temperature)
        elif self._backend_type == "transformers":
            return self._generate_transformers(prompt, max_tokens, temperature)
        else:
            return self._generate_rule_based(prompt)

    def _generate_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams

        if temperature == 0:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0,
                seed=self.seed,
            )
        else:
            params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                seed=self.seed,
            )

        messages = [
            {"role": "system", "content": INTENT_COMPILER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        outputs = self._model.chat(messages, sampling_params=params)
        return outputs[0].outputs[0].text

    def _generate_transformers(
        self, prompt: str, max_tokens: int, temperature: float
    ) -> str:
        import torch

        messages = [
            {"role": "system", "content": INTENT_COMPILER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        torch.manual_seed(self.seed)
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                top_p=1.0,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def _generate_rule_based(self, prompt: str) -> str:
        """Rule-based fallback when no GPU/model available."""
        prompt_lower = prompt.lower()

        # Default physical targets
        physical = [0.5] * 12
        acting_macro = {
            "intensity": 0.5,
            "instability": 0.2,
            "tenderness": 0.3,
            "tension": 0.3,
            "spontaneity": 0.5,
            "reference_mix": 0.0,
        }
        pacing = {
            "pace": 1.0,
            "hold_bias": 0.0,
            "boundary_bias": 0.0,
            "phrase_pressure": 0.0,
            "breath_tendency": 0.0,
        }
        inline_tags: list[str] = []
        warnings = ["Rule-based fallback: limited acting interpretation"]

        # Simple keyword matching
        if "angry" in prompt_lower or "\u6012" in prompt_lower:
            physical[2] = 0.8  # energy
            physical[3] = 0.7  # pressedness
            acting_macro["intensity"] = 0.8
            acting_macro["tension"] = 0.7
            inline_tags.append("[angry]")

        if "whisper" in prompt_lower or "\u56c1" in prompt_lower:
            physical[2] = 0.2  # energy
            physical[5] = 0.8  # breathiness
            acting_macro["intensity"] = 0.3
            inline_tags.append("[whisper]")

        if "gentle" in prompt_lower or "\u512a\u3057" in prompt_lower:
            acting_macro["tenderness"] = 0.8
            acting_macro["intensity"] = 0.3
            inline_tags.append("[tender]")

        if "fast" in prompt_lower or "\u901f" in prompt_lower:
            pacing["pace"] = 1.5

        if "slow" in prompt_lower or "\u3086\u3063\u304f\u308a" in prompt_lower:
            pacing["pace"] = 0.7

        if "sad" in prompt_lower or "\u60b2\u3057" in prompt_lower:
            physical[2] = 0.3  # energy
            acting_macro["tenderness"] = 0.6
            acting_macro["tension"] = 0.4
            pacing["pace"] = 0.8
            inline_tags.append("[sad]")

        if "excited" in prompt_lower or "\u5143\u6c17" in prompt_lower:
            physical[2] = 0.8  # energy
            physical[0] = 0.7  # pitch_level
            acting_macro["intensity"] = 0.8
            acting_macro["spontaneity"] = 0.7
            pacing["pace"] = 1.3
            inline_tags.append("[excited]")

        result = {
            "physical_targets": physical,
            "acting_macro": acting_macro,
            "pacing": pacing,
            "inline_tags": inline_tags,
            "warnings": warnings,
        }

        return json.dumps(result)

    @property
    def provenance(self) -> str:
        """Model provenance string for IntentCompilerOutput."""
        if self._backend_type == "rule_based":
            return "intent_compiler_rule_based_v1"
        return f"intent_compiler_{self.model_name}_{self._backend_type}"

    @property
    def is_loaded(self) -> bool:
        return self._backend_type != "none"

    @property
    def backend_type(self) -> str:
        return self._backend_type
