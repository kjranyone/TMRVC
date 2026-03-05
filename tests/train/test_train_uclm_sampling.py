from __future__ import annotations

from pathlib import Path

import pytest

from tmrvc_train.cli.train_uclm import (
    _build_sampler,
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
