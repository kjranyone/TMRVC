"""Universal TMRVC UCLM Constants.

This module re-exports constants from _generated_constants.py (auto-generated from YAML).

Single source of truth: configs/constants.yaml
Generated automatically on import when YAML changes.
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

from tmrvc_core._codegen import ensure_generated

ensure_generated()

from tmrvc_core._generated_constants import *  # noqa: F401, F403
