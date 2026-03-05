from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmrvc_train.dataset import DisentangledUCLMDataset


def _write_utt(cache_root: Path, dataset: str, speaker: str, utt: str) -> None:
    utt_dir = cache_root / dataset / "train" / speaker / utt
    utt_dir.mkdir(parents=True, exist_ok=True)

    np.save(utt_dir / "codec_tokens.npy", np.zeros((8, 4), dtype=np.int64))
    np.save(utt_dir / "explicit_state.npy", np.zeros((4, 8), dtype=np.float32))
    np.save(utt_dir / "ssl_state.npy", np.zeros((4, 128), dtype=np.float32))
    np.save(utt_dir / "spk_embed.npy", np.zeros((192,), dtype=np.float32))

    meta = {"speaker_id": speaker, "text": "x", "language_id": 0}
    (utt_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_dataset_include_datasets_filters_by_cache_top_dir(tmp_path: Path):
    _write_utt(tmp_path, "jvs", "jvs_spk1", "utt1")
    _write_utt(tmp_path, "vctk", "vctk_spk1", "utt1")

    ds_all = DisentangledUCLMDataset(tmp_path)
    ds_jvs = DisentangledUCLMDataset(tmp_path, include_datasets=["jvs"])
    ds_vctk = DisentangledUCLMDataset(tmp_path, include_datasets=["vctk"])

    assert len(ds_all) == 2
    assert len(ds_jvs) == 1
    assert len(ds_vctk) == 1
