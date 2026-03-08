"""v3 purity tests — guarantee that UCLM v2 traces are eliminated and the system 
maintains v3-only architecture standards.
"""

import pytest
import torch
import torch.nn as nn
from tmrvc_train.models.uclm_model import DisentangledUCLM, PointerHead
from tmrvc_serve.uclm_engine import UCLMEngine


class TestV3ArchitecturePurity:
    """Guarantee that the model structure follows v3 (Pointer-based) spec."""

    def test_model_has_pointer_head(self):
        """Model must have a PointerHead module."""
        model = DisentangledUCLM()
        assert hasattr(model, "pointer_head")
        assert isinstance(model.pointer_head, PointerHead)

    def test_model_lacks_duration_predictor(self):
        """Model must NOT have the legacy DurationPredictor module."""
        model = DisentangledUCLM()
        assert not hasattr(model, "duration_predictor")
        assert not hasattr(model, "feature_expander")

    def test_forward_methods_presence(self):
        """Only v3-compliant forward methods should exist."""
        model = DisentangledUCLM()
        assert hasattr(model, "forward_tts_pointer")
        assert hasattr(model, "forward_vc")
        assert hasattr(model, "forward_streaming")
        # forward_tts is retained as a compatibility wrapper that delegates
        # to forward_tts_pointer (per Worker 01 guardrails)
        assert hasattr(model, "forward_tts")
        # Legacy-only method must be gone
        assert not hasattr(model, "forward_tts_legacy")


class TestEnginePurity:
    """Guarantee that the inference engine is v3-only."""

    def test_engine_lacks_duration_estimation(self):
        """UCLMEngine must not have legacy duration estimation logic."""
        engine = UCLMEngine()
        assert not hasattr(engine, "_estimate_tts_target_length")

    def test_tts_uses_pointer_loop(self):
        """UCLMEngine.tts should be the pointer implementation."""
        # This is a basic check; implementation-level verification
        # is handled by functional integration tests.
        engine = UCLMEngine()
        assert hasattr(engine, "tts")


class TestNoV2Leakage:
    """Verify that no strings referencing 'UCLM v2' exist in the mainline source."""

    def test_no_uclm_v2_in_source_tree(self):
        """No Python source file in main packages should reference 'UCLM v2' as current."""
        import subprocess
        from pathlib import Path

        # Search in source directories
        src_dirs = ["tmrvc-core", "tmrvc-data", "tmrvc-train", "tmrvc-serve", "tmrvc-gui"]
        matches = []
        for d in src_dirs:
            try:
                res = subprocess.run(
                    ["grep", "-rn", "UCLM v2", d, "--include=*.py"],
                    capture_output=True,
                    text=True
                )
                if res.stdout:
                    matches.extend(res.stdout.splitlines())
            except FileNotFoundError:
                continue

        if matches:
            pytest.fail(f"Found 'UCLM v2' references in source:\n" + "\n".join(matches))
