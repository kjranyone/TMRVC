from __future__ import annotations

import os
from pathlib import Path

from tmrvc_train.cli.train_pipeline import (
    _cache_has_required_datasets,
    find_latest_cache,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def test_find_latest_cache_prefers_mtime_over_name(tmp_path: Path):
    out = tmp_path / "experiments"
    older = out / "jvs_z_oldname" / "cache"
    newer = out / "jvs_a_newname" / "cache"
    _touch(older / "jvs" / "train" / "spk" / "utt" / "meta.json")
    _touch(newer / "jvs" / "train" / "spk" / "utt" / "meta.json")

    # Force mtime ordering opposite to lexical name ordering.
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    picked = find_latest_cache(out, "jvs", required_datasets=["jvs"])
    assert picked == newer


def test_find_latest_cache_skips_candidates_missing_required_dataset(tmp_path: Path):
    out = tmp_path / "experiments"
    invalid = out / "jvs_run_1" / "cache"
    valid = out / "jvs_run_2" / "cache"
    invalid.mkdir(parents=True, exist_ok=True)
    _touch(valid / "jvs" / "train" / "spk" / "utt" / "meta.json")

    picked = find_latest_cache(out, "jvs", required_datasets=["jvs"])
    assert picked == valid


def test_cache_has_required_datasets_requires_dataset_train_dir(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    assert not _cache_has_required_datasets(cache_dir, ["jvs"])

    (cache_dir / "jvs" / "train").mkdir(parents=True, exist_ok=True)
    assert _cache_has_required_datasets(cache_dir, ["jvs"])
