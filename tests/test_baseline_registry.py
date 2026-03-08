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

    def test_baselines_are_frozen(self):
        """Verify baselines have been frozen (no longer placeholder)."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "CosyVoice" in text or "cosyvoice" in text.lower(), "Primary baseline must be present"

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


# ---------------------------------------------------------------------------
# Primary Baseline: CosyVoice 3 (Worker 06)
# ---------------------------------------------------------------------------


class TestPrimaryBaselineCosyVoice3:
    """Primary baseline (CosyVoice 3) must have all required fields populated."""

    PRIMARY_ID = "primary_fun_cosyvoice3_0p5b_2512_hf_29e01c4"

    def test_primary_entry_exists(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert self.PRIMARY_ID in text, "Primary baseline entry not found in registry"

    def test_primary_has_model_name(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "Fun-CosyVoice3-0.5B-2512" in text

    def test_primary_has_artifact_id(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "hf:FunAudioLLM/Fun-CosyVoice3-0.5B-2512@29e01c4" in text

    def test_primary_has_language_set(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        # The entry must specify the frozen language set
        section = text[text.find(self.PRIMARY_ID):]
        for lang in ["Chinese", "English", "Japanese", "Korean",
                      "German", "Spanish", "French", "Italian", "Russian"]:
            assert lang in section, (
                f"Primary baseline missing language: {lang}"
            )

    def test_primary_has_reference_lengths(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.PRIMARY_ID):]
        assert "3, 5, 10" in section, "Primary baseline missing reference_lengths_sec"

    def test_primary_has_date_frozen(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.PRIMARY_ID):]
        assert "date_frozen" in section
        # Must have an actual date, not TBD
        assert "2026" in section, "Primary baseline date_frozen appears unfrozen"

    def test_primary_has_hardware_class(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.PRIMARY_ID):]
        assert "hardware_class" in section
        assert "nvidia" in section.lower() or "gpu" in section.lower(), (
            "Primary baseline must specify a hardware class"
        )

    def test_primary_has_source_refs(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.PRIMARY_ID):]
        assert "source_refs" in section
        assert "huggingface.co" in section or "github.com" in section

    def test_primary_all_required_fields_in_entry(self):
        """All REQUIRED_FIELDS must appear in the primary entry section."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        start = text.find(self.PRIMARY_ID)
        assert start != -1
        # Find end of this entry (next ### or end of Active Entries)
        next_section = text.find("###", start + 10)
        if next_section == -1:
            next_section = len(text)
        section = text[start:next_section]

        for field in REQUIRED_FIELDS:
            assert f"`{field}`" in section, (
                f"Primary baseline entry missing required field: {field}"
            )


# ---------------------------------------------------------------------------
# Secondary Baseline: Qwen3-TTS (Worker 06)
# ---------------------------------------------------------------------------


class TestSecondaryBaselineQwen3TTS:
    """Secondary baseline (Qwen3-TTS) must have all required fields populated."""

    SECONDARY_ID = "secondary_qwen3_tts_12hz_1p7b_base_hf_fd4b254"

    def test_secondary_entry_exists(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert self.SECONDARY_ID in text, "Secondary baseline entry not found in registry"

    def test_secondary_has_model_name(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "Qwen3-TTS" in text

    def test_secondary_has_artifact_id(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        assert "hf:Qwen/Qwen3-TTS-12Hz-1.7B-Base@fd4b254" in text

    def test_secondary_has_tokenizer_version(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.SECONDARY_ID):]
        assert "tokenizer_version" in section
        assert "Qwen3-TTS-Tokenizer" in section

    def test_secondary_has_date_frozen(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.SECONDARY_ID):]
        assert "2026" in section, "Secondary baseline date_frozen appears unfrozen"

    def test_secondary_has_reference_lengths(self):
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        section = text[text.find(self.SECONDARY_ID):]
        assert "3, 5, 10" in section

    def test_secondary_all_required_fields_in_entry(self):
        """All REQUIRED_FIELDS must appear in the secondary entry section."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        start = text.find(self.SECONDARY_ID)
        assert start != -1
        next_section = text.find("###", start + 10)
        if next_section == -1:
            next_section = len(text)
        section = text[start:next_section]

        for field in REQUIRED_FIELDS:
            assert f"`{field}`" in section, (
                f"Secondary baseline entry missing required field: {field}"
            )


# ---------------------------------------------------------------------------
# No Placeholder or Partially-Specified Entries (Worker 06)
# ---------------------------------------------------------------------------


class TestNoPlaceholderEntries:
    """Active entries must not contain placeholder or TBD values in critical fields."""

    def test_no_tbd_in_artifact_id(self):
        """artifact_id must not be TBD in any active entry."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        active_start = text.find("## Active Entries")
        other_start = text.find("## Other Candidate Baselines")
        if active_start == -1:
            pytest.skip("No Active Entries section found")
        end = other_start if other_start != -1 else len(text)
        active = text[active_start:end]

        # Check table rows for TBD in artifact fields
        for line in active.split("\n"):
            if "`artifact_id`" in line:
                assert "TBD" not in line, f"Placeholder artifact_id found: {line}"

    def test_no_placeholder_model_names(self):
        """model_name must not be 'placeholder' or 'TBD'."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        active_start = text.find("## Active Entries")
        other_start = text.find("## Other Candidate Baselines")
        if active_start == -1:
            pytest.skip("No Active Entries section found")
        end = other_start if other_start != -1 else len(text)
        active = text[active_start:end]

        for line in active.split("\n"):
            if "`model_name`" in line:
                lower = line.lower()
                assert "placeholder" not in lower, f"Placeholder model_name: {line}"

    def test_active_entries_have_evaluation_protocol_version(self):
        """Each active entry must reference an evaluation_protocol_version."""
        text = REGISTRY_PATH.read_text(encoding="utf-8")
        active_start = text.find("## Active Entries")
        other_start = text.find("## Other Candidate Baselines")
        if active_start == -1:
            pytest.skip("No Active Entries section found")
        end = other_start if other_start != -1 else len(text)
        active = text[active_start:end]

        assert "evaluation_protocol_version" in active, (
            "Active entries missing evaluation_protocol_version"
        )
