"""Worker 06: External baseline registry validation.

Ensures the baseline registry is well-formed and tracks freeze status.
The baseline must be fully frozen (no TBD fields) before Stage D sign-off.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "docs" / "design" / "external-baseline-registry.md"

REQUIRED_FIELDS = {
    "baseline_id",
    "model_name",
    "artifact_id",
    "tokenizer_version",
    "prompt_rule",
    "reference_lengths_sec",
    "inference_settings",
    "evaluation_set_version",
    "date_frozen",
    "notes",
}


class TestRegistryStructure:
    def test_file_exists(self):
        assert REGISTRY_PATH.exists()

    def test_contains_entry_template(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        for field in REQUIRED_FIELDS:
            assert f"`{field}`" in text, f"Template missing field: {field}"

    def test_contains_candidate_baselines(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "CosyVoice" in text
        assert "F5-TTS" in text
        assert "MaskGCT" in text
        assert "Qwen3-TTS" in text

    def test_no_or_newer_successor_in_active_entries(self):
        """Plan forbids 'or newer successor' in active baseline entries."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        active_start = text.find("## Active Entries")
        candidate_start = text.find("## Candidate Baselines")
        if active_start == -1 or candidate_start == -1:
            pytest.skip("Could not isolate Active Entries section")
        active_section = text[active_start:candidate_start].lower()
        # The phrase may appear in instructions but must not appear in actual entries
        # Check table rows only (lines starting with |)
        table_lines = [l for l in active_section.split("\n") if l.strip().startswith("|")]
        for line in table_lines:
            assert "or newer successor" not in line, (
                f"Active entry uses forbidden 'or newer successor': {line}"
            )


class TestBaselineFreezeStatus:
    """Track whether the baseline is frozen.

    These tests document the current freeze status.
    The xfail test will start passing once a real baseline is frozen.
    """

    def test_pending_freeze_placeholder_exists(self):
        """Verify the placeholder entry is present (pre-Stage-D state)."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "baseline_pending_freeze" in text

    @pytest.mark.xfail(
        reason="Baseline not yet frozen — Stage D prerequisite (Worker 06 task 13)",
        strict=False,
    )
    def test_no_tbd_fields_in_active_entries(self):
        """All active baseline entries must have no TBD fields at sign-off."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        # Extract only the Active Entries section
        active_start = text.find("## Active Entries")
        active_end = text.find("## Candidate Baselines")
        if active_start == -1 or active_end == -1:
            pytest.fail("Could not find Active Entries section")
        active_section = text[active_start:active_end]
        assert "`TBD`" not in active_section, (
            "Active baseline entry still has TBD fields. "
            "Freeze a real baseline before Stage D sign-off."
        )

    def test_selection_criteria_documented(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "publicly available model weights" in text
        assert "proprietary-only" in text
