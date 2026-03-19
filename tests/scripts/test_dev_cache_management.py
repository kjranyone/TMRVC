from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import dev

_V3_REMOVED = "v3 MFA/alignment/legacy API removed in v4 migration"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_find_latest_cache_for_enabled_datasets_uses_valid_candidates(
    monkeypatch, tmp_path: Path
):
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    invalid = experiments / f"{prefix}_old" / "cache"
    valid_old = experiments / f"{prefix}_mid" / "cache"
    valid_new = experiments / f"{prefix}_new" / "cache"
    invalid.mkdir(parents=True, exist_ok=True)
    _touch(valid_old / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(valid_old / "vctk" / "train" / "s" / "u" / "meta.json")
    _touch(valid_new / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(valid_new / "vctk" / "train" / "s" / "u" / "meta.json")

    os.utime(valid_old, (1, 1))
    os.utime(valid_new, (2, 2))

    picked = dev.find_latest_cache_for_enabled_datasets(enabled)
    assert picked == valid_new


@pytest.mark.skip(reason=_V3_REMOVED)
def test_clear_training_caches_for_enabled_datasets_is_scoped(
    monkeypatch, tmp_path: Path
):
    pass


def test_find_latest_uclm_checkpoint_for_enabled_datasets_picks_newest(
    monkeypatch, tmp_path: Path
):
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    old_exp = experiments / f"{prefix}_old"
    new_exp = experiments / f"{prefix}_new"
    other_exp = experiments / "other_aaa"

    old_ckpt = old_exp / "checkpoints" / "uclm_final.pt"
    new_ckpt = new_exp / "checkpoints" / "uclm_final.pt"
    _touch(old_ckpt)
    _touch(new_ckpt)
    _touch(other_exp / "checkpoints" / "uclm_final.pt")

    os.utime(old_ckpt, (1, 1))
    os.utime(new_ckpt, (2, 2))

    exp_dir, ckpt = dev.find_latest_uclm_checkpoint_for_enabled_datasets(enabled)
    assert exp_dir == new_exp
    assert ckpt == new_ckpt


def test_quality_gate_status_reads_json(tmp_path: Path):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir(parents=True)
    report = exp_dir / "quality_gate_report.json"
    report.write_text('{"status":"ok"}', encoding="utf-8")
    assert dev._quality_gate_status(exp_dir) == "ok"


def test_find_latest_experiment_for_enabled_datasets_uses_cache_validation(
    monkeypatch, tmp_path: Path
):
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    invalid = experiments / f"{prefix}_old"
    valid = experiments / f"{prefix}_new"
    other = experiments / "other_dataset_new"

    (invalid / "cache").mkdir(parents=True, exist_ok=True)
    _touch(valid / "cache" / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(valid / "cache" / "vctk" / "train" / "s" / "u" / "meta.json")
    _touch(other / "cache" / "jvs" / "train" / "s" / "u" / "meta.json")
    _touch(other / "cache" / "vctk" / "train" / "s" / "u" / "meta.json")

    os.utime(valid, (2, 2))
    os.utime(invalid, (1, 1))

    picked = dev._find_latest_experiment_for_enabled_datasets(enabled)
    assert picked == valid


@pytest.mark.skip(reason=_V3_REMOVED)
def test_finalize_training_outputs_blocks_missing_quality_gate(
    monkeypatch, tmp_path: Path, capsys
):
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_prepare_tts_alignment_noninteractive_uses_textgrid_override(
    monkeypatch, tmp_path: Path
):
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_mfa_align_and_inject_fails_when_mfa_command_missing(
    monkeypatch, tmp_path: Path
):
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_get_mfa_command_from_env(monkeypatch):
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_normalize_mfa_model_name_aliases():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_extract_env_name_from_run_command():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_suggest_mfa_install_cmd_with_python_pin():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_check_mfa_japanese_runtime_parses_ok(monkeypatch):
    pass


def test_dev_module_importable():
    """Verify that dev.py can be imported and has expected entry points."""
    assert hasattr(dev, "main")
    assert hasattr(dev, "print_menu")
    # v4: full training is private _cmd_full_training
    assert hasattr(dev, "_cmd_full_training")


@pytest.mark.skip(reason=_V3_REMOVED)
def test_dev_module_has_v3_entrypoints():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_menu_labels_contain_v3_pointer(capsys):
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_handlers_include_legacy_option():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_validate_v3_config_accepts_valid():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_validate_v3_config_rejects_missing_fields():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_validate_v3_config_rejects_bad_tts_mode():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_v3_optional_defaults_keys():
    pass


@pytest.mark.skip(reason=_V3_REMOVED)
def test_mfa_align_propagates_heuristic_fallback_flag(monkeypatch, tmp_path: Path):
    pass
