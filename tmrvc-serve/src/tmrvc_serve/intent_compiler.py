
"""Intent Compiler for expressive speech synthesis (Worker 04).

Converts raw prompt/tags into canonical IntentCompilerOutput.
"""

from __future__ import annotations

import uuid
import logging
from typing import List, Optional
import torch

from tmrvc_core.types import IntentCompilerOutput, PacingControls

logger = logging.getLogger(__name__)

class IntentCompiler:
    """Compiles high-level intentions into low-level model controls."""

    def compile(self, prompt: str, context: Optional[dict] = None) -> IntentCompilerOutput:
        """Compile a natural language prompt into an Intent record."""
        compile_id = str(uuid.uuid4())
        
        # SOTA: Basic parsing of pacing and emotion from tags (v0)
        # In a real SOTA implementation, this would use a small LLM or rule-base.
        pacing = PacingControls()
        explicit_vs = torch.zeros((1, 8))
        
        # Simple heuristic parsing for v0
        warnings = []
        if "[fast]" in prompt:
            pacing.pace = 1.3
        if "[slow]" in prompt:
            pacing.pace = 0.7
        if "[stable]" in prompt:
            pacing.hold_bias = 3.0
        if "[energetic]" in prompt:
            explicit_vs[0, 2] = 0.8 # energy_level
            
        logger.info("Compiled prompt [%s] -> ID: %s", prompt, compile_id)
        
        return IntentCompilerOutput(
            compile_id=compile_id,
            source_prompt=prompt,
            explicit_voice_state=explicit_vs,
            pacing=pacing,
            warnings=warnings,
            metadata={
                "compiler_version": "0.1.0-alpha",
                "context": context or {}
            }
        )
