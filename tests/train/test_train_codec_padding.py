from __future__ import annotations

import pytest
import torch

from tmrvc_train.cli.train_codec import (
    _align_waveform_to_token_frames,
    collate_fn,
    _prepare_decoder_inputs,
    _validate_token_ranges,
)


def test_prepare_decoder_inputs_clamps_padding_to_zero():
    ta = torch.tensor([[[1, -1, 7]]], dtype=torch.long)
    tb = torch.tensor([[[2, -1, 3]]], dtype=torch.long)
    ta_in, tb_in = _prepare_decoder_inputs(ta, tb)
    assert int(ta_in.min().item()) >= 0
    assert int(tb_in.min().item()) >= 0
    assert ta_in.tolist() == [[[1, 0, 7]]]
    assert tb_in.tolist() == [[[2, 0, 3]]]


def test_validate_token_ranges_accepts_padding_and_valid_ids():
    ta = torch.tensor([[[0, 1, -1, 1023]]], dtype=torch.long)
    tb = torch.tensor([[[0, 1, -1, 63]]], dtype=torch.long)
    _validate_token_ranges(ta, tb)


def test_validate_token_ranges_rejects_invalid_ids():
    ta = torch.tensor([[[1024]]], dtype=torch.long)
    tb = torch.tensor([[[0]]], dtype=torch.long)
    with pytest.raises(ValueError, match="Invalid token range"):
        _validate_token_ranges(ta, tb)


def test_collate_fn_trims_waveform_tail_to_token_length():
    T = 44
    waveform = torch.arange(10650, dtype=torch.float32)
    batch = [
        {
            "waveform": waveform,
            "target_a": torch.zeros((8, T), dtype=torch.long),
            "target_b": torch.zeros((4, T), dtype=torch.long),
            "voice_state": torch.zeros((T, 8), dtype=torch.float32),
        }
    ]

    out = collate_fn(batch)
    assert out["waveform"].shape == (1, 1, T * 240)
    assert torch.equal(out["waveform"][0, 0], waveform[: T * 240])


def test_align_waveform_to_token_frames_pads_short_tail():
    T = 10
    waveform = torch.ones(2300, dtype=torch.float32)
    aligned = _align_waveform_to_token_frames(waveform, T)
    assert aligned.shape == (1, 2400)
    assert torch.all(aligned[0, :2300] == 1.0)
    assert torch.all(aligned[0, 2300:] == 0.0)
