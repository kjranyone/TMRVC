from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(
    reason="scripts/annotate/run_forced_alignment.py removed in v3 cleanup"
)


def test_equal_duration_split_with_bos_eos_enforces_zero_edges():
    pass


def test_equal_duration_split_with_bos_eos_returns_none_when_no_content():
    pass


def test_cli_fails_without_textgrid_and_without_heuristic():
    pass
