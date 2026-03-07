from __future__ import annotations

from pathlib import Path

import pytest

from tmrvc_train.cli.train_uclm import (
    _build_sampler,
    _collect_tts_supervision_by_dataset,
    _compute_balanced_sample_weights,
    train_uclm,
)
from tmrvc_train.dataset import DisentangledUCLMDataset


def test_compute_balanced_sample_weights_balances_dataset_and_speaker():
    utterances = [
        {"dataset": "jvs", "speaker_id": "jvs_s1"},
        {"dataset": "jvs", "speaker_id": "jvs_s1"},
        {"dataset": "vctk", "speaker_id": "vctk_s1"},
    ]
    weights = _compute_balanced_sample_weights(utterances)

    # vctk has fewer dataset samples and same per-speaker count => larger weight.
    assert weights[2] > weights[0]
    # Same dataset/speaker bucket => same weight.
    assert weights[0] == weights[1]


def test_build_sampler_returns_none_for_shuffle(tmp_path: Path):
    ds = DisentangledUCLMDataset(tmp_path)
    assert _build_sampler(ds, sampling_strategy="shuffle", seed=42) is None


def test_train_uclm_raises_on_empty_cache(tmp_path: Path):
    with pytest.raises(ValueError, match="No training utterances found"):
        train_uclm(
            cache_dir=tmp_path,
            output_dir=tmp_path / "ckpt",
            batch_size=2,
            max_steps=1,
            device="cpu",
            lr=1e-4,
            datasets="jvs",
            seed=42,
            sampling_strategy="balanced",
        )


def test_collect_tts_supervision_by_dataset_reports_per_dataset(tmp_path: Path):
    d1_u1 = tmp_path / "d1_u1"
    d1_u2 = tmp_path / "d1_u2"
    d2_u1 = tmp_path / "d2_u1"
    for p in (d1_u1, d1_u2, d2_u1):
        p.mkdir(parents=True, exist_ok=True)

    # d1_u1 has full TTS supervision
    (d1_u1 / "phoneme_ids.npy").write_bytes(b"x")
    (d1_u1 / "durations.npy").write_bytes(b"x")
    # d1_u2 has partial (invalid for TTS supervision)
    (d1_u2 / "phoneme_ids.npy").write_bytes(b"x")
    # d2_u1 has none

    utterances = [
        {"dataset": "d1", "path": d1_u1},
        {"dataset": "d1", "path": d1_u2},
        {"dataset": "d2", "path": d2_u1},
    ]
    stats = _collect_tts_supervision_by_dataset(utterances)
    assert stats["d1"]["total"] == 2
    assert stats["d1"]["tts_supervised"] == 1
    assert stats["d2"]["total"] == 1
    assert stats["d2"]["tts_supervised"] == 0


def test_collect_tts_supervision_reports_text_and_legacy_separately(tmp_path: Path):
    """v3 update: _collect_tts_supervision_by_dataset now reports text_supervised
    and legacy_duration_supervised as separate counters."""
    d1_u1 = tmp_path / "d1_u1"
    d1_u2 = tmp_path / "d1_u2"
    d1_u3 = tmp_path / "d1_u3"
    for p in (d1_u1, d1_u2, d1_u3):
        p.mkdir(parents=True, exist_ok=True)

    # d1_u1: has phoneme_ids + durations (both text_supervised and legacy_duration)
    (d1_u1 / "phoneme_ids.npy").write_bytes(b"x")
    (d1_u1 / "durations.npy").write_bytes(b"x")
    # d1_u2: has phoneme_ids only (text_supervised but NOT legacy_duration)
    (d1_u2 / "phoneme_ids.npy").write_bytes(b"x")
    # d1_u3: has nothing

    utterances = [
        {"dataset": "d1", "path": d1_u1},
        {"dataset": "d1", "path": d1_u2},
        {"dataset": "d1", "path": d1_u3},
    ]
    stats = _collect_tts_supervision_by_dataset(utterances)

    assert stats["d1"]["total"] == 3
    # text_supervised counts utterances with phoneme_ids.npy
    assert stats["d1"]["text_supervised"] == 2
    # legacy_duration_supervised counts utterances with both phoneme_ids + durations
    assert stats["d1"]["legacy_duration_supervised"] == 1
    # tts_supervised is an alias for legacy_duration_supervised (backward compat)
    assert stats["d1"]["tts_supervised"] == stats["d1"]["legacy_duration_supervised"]
