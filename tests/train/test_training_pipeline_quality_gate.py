from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmrvc_train.pipeline import TrainingPipeline


def _write_valid_cache_entry(cache_dir: Path, dataset: str, speaker: str, utt: str) -> Path:
    utt_dir = cache_dir / dataset / "train" / speaker / utt
    utt_dir.mkdir(parents=True, exist_ok=True)
    np.save(utt_dir / "codec_tokens.npy", np.zeros((8, 8), dtype=np.int64))
    np.save(utt_dir / "control_tokens.npy", np.zeros((4, 8), dtype=np.int64))
    np.save(utt_dir / "explicit_state.npy", np.zeros((8, 8), dtype=np.float32))
    np.save(utt_dir / "ssl_state.npy", np.zeros((8, 128), dtype=np.float32))
    np.save(utt_dir / "spk_embed.npy", np.zeros((192,), dtype=np.float32))
    (utt_dir / "meta.json").write_text(
        json.dumps({"speaker_id": speaker, "language_id": 0, "n_frames": 8}),
        encoding="utf-8",
    )
    return utt_dir


def test_quality_gate_passes_and_writes_report(monkeypatch, tmp_path: Path):
    exp = tmp_path / "exp"
    cache = tmp_path / "cache"
    _write_valid_cache_entry(cache, "jvs", "jvs_spk1", "utt1")

    called: list[list[str]] = []

    def fake_train_main(argv: list[str]) -> None:
        called.append(argv)

    monkeypatch.setattr("tmrvc_train.cli.train_uclm.main", fake_train_main)

    p = TrainingPipeline(
        experiment_dir=exp,
        dataset="jvs",
        raw_dir=tmp_path,
        cache_dir=cache,
        config={"train_steps": 1, "quality_gate_token_samples": 8},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs"],
    )
    assert p._run_training()
    assert called
    assert (exp / "quality_gate_report.json").exists()


def test_quality_gate_fails_on_missing_required_file(tmp_path: Path):
    exp = tmp_path / "exp"
    cache = tmp_path / "cache"
    utt = _write_valid_cache_entry(cache, "jvs", "jvs_spk1", "utt1")
    (utt / "ssl_state.npy").unlink()

    p = TrainingPipeline(
        experiment_dir=exp,
        dataset="jvs",
        raw_dir=tmp_path,
        cache_dir=cache,
        config={"train_steps": 1},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs"],
    )
    assert not p._run_training()


def test_quality_gate_fails_on_out_of_range_codec_tokens(tmp_path: Path):
    exp = tmp_path / "exp"
    cache = tmp_path / "cache"
    utt = _write_valid_cache_entry(cache, "jvs", "jvs_spk1", "utt1")
    np.save(utt / "codec_tokens.npy", np.full((8, 8), 1024, dtype=np.int64))

    p = TrainingPipeline(
        experiment_dir=exp,
        dataset="jvs",
        raw_dir=tmp_path,
        cache_dir=cache,
        config={"train_steps": 1, "quality_gate_token_samples": 8},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs"],
    )
    assert not p._run_training()


def test_quality_gate_fails_on_waveform_length_mismatch(tmp_path: Path):
    exp = tmp_path / "exp"
    cache = tmp_path / "cache"
    utt = _write_valid_cache_entry(cache, "jvs", "jvs_spk1", "utt1")
    # n_frames=8 in meta -> expected 8 * 240 = 1920 samples.
    # Severe mismatch (> 1 frame) should fail.
    np.save(utt / "waveform.npy", np.zeros((1, 2161), dtype=np.float32))

    p = TrainingPipeline(
        experiment_dir=exp,
        dataset="jvs",
        raw_dir=tmp_path,
        cache_dir=cache,
        config={"train_steps": 1, "quality_gate_token_samples": 8},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs"],
    )
    assert not p._run_training()


def test_quality_gate_allows_subframe_waveform_tail_remainder(tmp_path: Path):
    exp = tmp_path / "exp"
    cache = tmp_path / "cache"
    utt = _write_valid_cache_entry(cache, "jvs", "jvs_spk1", "utt1")
    # n_frames=8 in meta -> expected 1920. +1 sample tail remainder should pass.
    np.save(utt / "waveform.npy", np.zeros((1, 1921), dtype=np.float32))

    p = TrainingPipeline(
        experiment_dir=exp,
        dataset="jvs",
        raw_dir=tmp_path,
        cache_dir=cache,
        config={"train_steps": 1, "quality_gate_token_samples": 8},
        workers=1,
        seed=42,
        skip_preprocess=True,
        run_training=True,
        train_datasets=["jvs"],
    )
    assert p._run_training()
