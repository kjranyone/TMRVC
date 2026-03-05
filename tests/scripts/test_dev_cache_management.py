from __future__ import annotations

import os
from pathlib import Path

import dev


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


def test_clear_training_caches_for_enabled_datasets_is_scoped(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(dev, "EXPERIMENTS_DIR", experiments)
    enabled = ["jvs", "vctk"]
    prefix = "_".join(sorted(enabled))

    # Legacy cache: only enabled datasets should be removed.
    (tmp_path / "data" / "cache" / "jvs").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "vctk").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "other").mkdir(parents=True)

    # Experiment caches: only matching prefix should be removed.
    (experiments / f"{prefix}_aaa" / "cache").mkdir(parents=True)
    (experiments / "other_dataset_aaa" / "cache").mkdir(parents=True)

    removed = dev.clear_training_caches_for_enabled_datasets(enabled)
    assert removed == 3
    assert not (tmp_path / "data" / "cache" / "jvs").exists()
    assert not (tmp_path / "data" / "cache" / "vctk").exists()
    assert (tmp_path / "data" / "cache" / "other").exists()
    assert not (experiments / f"{prefix}_aaa" / "cache").exists()
    assert (experiments / "other_dataset_aaa" / "cache").exists()


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
