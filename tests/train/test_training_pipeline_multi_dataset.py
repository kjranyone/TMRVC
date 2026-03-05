from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmrvc_train.pipeline import TrainingPipeline


def _write_valid_cache_entry(cache_dir: Path, dataset: str, speaker: str, utt: str) -> None:
    utt_dir = cache_dir / dataset / "train" / speaker / utt
    utt_dir.mkdir(parents=True, exist_ok=True)
    np.save(utt_dir / "codec_tokens.npy", np.zeros((8, 8), dtype=np.int64))
    np.save(utt_dir / "control_tokens.npy", np.zeros((4, 8), dtype=np.int64))
    np.save(utt_dir / "explicit_state.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(utt_dir / "ssl_state.npy", np.zeros((8, 128), dtype=np.float32))
    np.save(utt_dir / "spk_embed.npy", np.zeros((192,), dtype=np.float32))
    (utt_dir / "meta.json").write_text(
        json.dumps({"speaker_id": speaker, "language_id": 0}), encoding="utf-8"
    )


def test_run_training_passes_dataset_filter_arg(monkeypatch, tmp_path: Path):
    experiment_dir = tmp_path / "exp"
    cache_dir = tmp_path / "cache"
    _write_valid_cache_entry(cache_dir, "jvs", "s", "u")
    _write_valid_cache_entry(cache_dir, "vctk", "s", "u")

    called: list[list[str]] = []

    def fake_train_main(argv: list[str]) -> None:
        called.append(argv)

    monkeypatch.setattr("tmrvc_train.cli.train_uclm.main", fake_train_main)

    pipeline = TrainingPipeline(
        experiment_dir=experiment_dir,
        dataset="multi",
        raw_dir=tmp_path,
        cache_dir=cache_dir,
        config={"train_steps": 10, "train_batch_size": 2, "train_device": "cpu"},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs", "vctk"],
    )

    assert pipeline._run_training()
    args = called[0]
    assert "--datasets" in args
    assert args[args.index("--datasets") + 1] == "jvs,vctk"


def test_run_training_fails_when_any_required_dataset_cache_missing(tmp_path: Path):
    experiment_dir = tmp_path / "exp"
    cache_dir = tmp_path / "cache"
    _write_valid_cache_entry(cache_dir, "jvs", "s", "u")
    # vctk intentionally missing

    pipeline = TrainingPipeline(
        experiment_dir=experiment_dir,
        dataset="multi",
        raw_dir=tmp_path,
        cache_dir=cache_dir,
        config={"train_steps": 10},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs", "vctk"],
    )

    assert not pipeline._run_training()
