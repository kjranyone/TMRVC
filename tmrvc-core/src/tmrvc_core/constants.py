"""Universal TMRVC UCLM Constants.

This module re-exports constants from _generated_constants.py (auto-generated from YAML).

Single source of truth: configs/constants.yaml
Run: python scripts/codegen/generate_constants.py
"""

from pathlib import Path as _Path

_YAML_PATH = _Path(__file__).resolve().parents[3] / "configs" / "constants.yaml"
_GEN_PATH = _Path(__file__).with_name("_generated_constants.py")

if not _YAML_PATH.exists():
    raise FileNotFoundError(
        f"configs/constants.yaml が見つかりません。\n"
        f"初回セットアップ: cp configs/constants.yaml.example configs/constants.yaml\n"
        f"Expected: {_YAML_PATH}"
    )

if not _GEN_PATH.exists():
    raise FileNotFoundError(
        f"_generated_constants.py が見つかりません。\n"
        f"実行してください: python scripts/codegen/generate_constants.py"
    )

# Auto-generated from YAML - the canonical runtime budget and tensor-layout
# values must be defined there, not re-declared here.
# Worker 04 Task 16: all runtime-budget constants are owned by
# configs/constants.yaml and must NOT be re-declared in this file.
from tmrvc_core._generated_constants import *  # noqa: F401, F403
