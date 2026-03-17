
import pytest
import torch
import numpy as np
import json
from pathlib import Path
from tmrvc_train.dataset.uclm_dataset import DisentangledUCLMDataset

def test_require_tts_supervision_enforcement(tmp_path):
    # Setup dummy cache directory
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    
    # Utterance 1: Has phoneme_ids.npy (Supervised)
    utt1 = cache_dir / "ds1" / "train" / "utt1"
    utt1.mkdir(parents=True)
    with open(utt1 / "meta.json", "w") as f:
        json.dump({"speaker_id": "spk1"}, f)
    np.save(utt1 / "codec_tokens.npy", np.zeros((1, 100)))
    np.save(utt1 / "explicit_state.npy", np.zeros((100, 8)))
    np.save(utt1 / "ssl_state.npy", np.zeros((100, 768)))
    np.save(utt1 / "spk_embed.npy", np.zeros(192))
    np.save(utt1 / "phoneme_ids.npy", np.zeros(10, dtype=np.int64))
    
    # Utterance 2: Missing phoneme_ids.npy (Unsupervised/Noise)
    utt2 = cache_dir / "ds1" / "train" / "utt2"
    utt2.mkdir(parents=True)
    with open(utt2 / "meta.json", "w") as f:
        json.dump({"speaker_id": "spk1"}, f)
    np.save(utt2 / "codec_tokens.npy", np.zeros((1, 100)))
    np.save(utt2 / "explicit_state.npy", np.zeros((100, 8)))
    np.save(utt2 / "ssl_state.npy", np.zeros((100, 768)))
    np.save(utt2 / "spk_embed.npy", np.zeros(192))
    
    # Case 1: require_tts_supervision=False (Default, should load both)
    ds_all = DisentangledUCLMDataset(cache_dir, require_tts_supervision=False)
    assert len(ds_all) == 2
    
    # Case 2: require_tts_supervision=True (GEMINI.md Enforcement)
    ds_filtered = DisentangledUCLMDataset(cache_dir, require_tts_supervision=True)
    assert len(ds_filtered) == 1
    assert Path(ds_filtered.utterances[0]["path"]).name == "utt1"
