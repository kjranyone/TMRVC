"""Tests for phoneme_aliases.yaml loading and structure.

Covers:
- YAML file loads without errors
- Alias mapping has expected structure (required fields per entry)
- Inventory migration policy field exists with required keys
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ALIASES_PATH = Path(__file__).resolve().parents[2] / "configs" / "phoneme_aliases.yaml"


@pytest.fixture(scope="module")
def aliases_data():
    """Load the phoneme_aliases.yaml file once per module."""
    assert ALIASES_PATH.exists(), f"phoneme_aliases.yaml not found at {ALIASES_PATH}"
    with open(ALIASES_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Loading test
# ---------------------------------------------------------------------------


class TestPhonemeAliasesLoading:
    def test_file_exists(self):
        assert ALIASES_PATH.exists()

    def test_loads_without_error(self, aliases_data):
        assert aliases_data is not None

    def test_top_level_keys(self, aliases_data):
        assert "aliases" in aliases_data
        assert "inventory_policy" in aliases_data


# ---------------------------------------------------------------------------
# Alias mapping structure
# ---------------------------------------------------------------------------

_REQUIRED_ALIAS_FIELDS = {
    "source_symbol",
    "normalized_symbol",
    "canonical_symbol",
    "language",
    "source_backend",
    "action",
    "note",
}


class TestAliasStructure:
    def test_aliases_is_list(self, aliases_data):
        assert isinstance(aliases_data["aliases"], list)

    def test_aliases_non_empty(self, aliases_data):
        assert len(aliases_data["aliases"]) > 0

    def test_each_alias_has_required_fields(self, aliases_data):
        for i, entry in enumerate(aliases_data["aliases"]):
            missing = _REQUIRED_ALIAS_FIELDS - set(entry.keys())
            assert not missing, (
                f"Alias entry {i} missing fields: {missing}"
            )

    def test_action_values_valid(self, aliases_data):
        valid_actions = {"map", "drop", "unk"}
        for i, entry in enumerate(aliases_data["aliases"]):
            assert entry["action"] in valid_actions, (
                f"Alias entry {i} has invalid action: {entry['action']}"
            )

    def test_language_is_string(self, aliases_data):
        for entry in aliases_data["aliases"]:
            assert isinstance(entry["language"], str)


# ---------------------------------------------------------------------------
# Inventory migration policy
# ---------------------------------------------------------------------------


class TestInventoryPolicy:
    def test_policy_exists(self, aliases_data):
        assert "inventory_policy" in aliases_data

    def test_version_field(self, aliases_data):
        policy = aliases_data["inventory_policy"]
        assert "version" in policy
        assert isinstance(policy["version"], str)

    def test_compatible_with_field(self, aliases_data):
        policy = aliases_data["inventory_policy"]
        assert "compatible_with" in policy

    def test_addition_rule_field(self, aliases_data):
        policy = aliases_data["inventory_policy"]
        assert "addition_rule" in policy
        assert policy["addition_rule"] == "append_only"

    def test_unk_fallback_field(self, aliases_data):
        policy = aliases_data["inventory_policy"]
        assert "unk_fallback" in policy
        assert policy["unk_fallback"] is True
