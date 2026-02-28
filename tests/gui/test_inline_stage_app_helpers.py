"""Tests for inline-stage helper functions in serve app."""

from __future__ import annotations

import numpy as np
import pytest

from tmrvc_core.dialogue_types import StyleParams


def _has_fastapi() -> bool:
    try:
        import fastapi  # noqa: F401
        return True
    except ImportError:
        return False


class TestInlineStageAppHelpers:
    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_append_silence_adds_expected_samples(self):
        from tmrvc_serve.app import _append_silence

        audio = np.zeros(2400, dtype=np.float32)  # 100ms
        out = _append_silence(audio, leading_ms=100, trailing_ms=50)
        # 100ms + 100ms + 50ms = 250ms @ 24kHz => 6000 samples
        assert len(out) == 6000

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_resolve_sentence_pause_clamps_to_non_negative(self):
        from tmrvc_serve.app import _resolve_sentence_pause

        assert _resolve_sentence_pause(120, 180) == 300
        assert _resolve_sentence_pause(120, -999) == 0

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_apply_inline_stage_overlay_blends_style(self):
        from tmrvc_serve.app import _apply_inline_stage_overlay

        base = StyleParams.neutral()
        overlay = StyleParams(emotion="whisper", energy=-0.8, reasoning="inline")
        merged = _apply_inline_stage_overlay(base, overlay)
        assert merged is not None
        assert merged.energy < 0.0
        assert "inline_stage" in merged.reasoning

    @pytest.mark.skipif(
        not _has_fastapi(),
        reason="fastapi not installed",
    )
    def test_apply_inline_stage_overlay_ignores_non_style_overlay(self):
        from tmrvc_serve.app import _apply_inline_stage_overlay

        base = StyleParams.neutral()
        merged = _apply_inline_stage_overlay(base, {"emotion": "whisper"})
        assert merged is base
