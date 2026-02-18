"""Tests for constant integrity and internal consistency."""

import yaml

from tmrvc_core import constants


def test_sample_rate():
    assert constants.SAMPLE_RATE == 24000


def test_hop_length_is_10ms():
    # 10ms @ 24kHz = 240 samples
    assert constants.HOP_LENGTH == 240
    assert constants.HOP_LENGTH == constants.SAMPLE_RATE * 10 // 1000


def test_n_freq_bins():
    assert constants.N_FREQ_BINS == constants.N_FFT // 2 + 1


def test_d_vocoder_features_matches_freq_bins():
    assert constants.D_VOCODER_FEATURES == constants.N_FREQ_BINS


def test_lora_delta_size():
    # 4 layers × (384 × 4 + 4 × 384) × 2 (K+V) = 4 × 6144 = 24576
    expected = constants.N_LORA_LAYERS * 6144
    assert constants.LORA_DELTA_SIZE == expected


def test_ir_subband_edges_length():
    # n_ir_subbands + 1 edges
    assert len(constants.IR_SUBBAND_EDGES_HZ) == constants.N_IR_SUBBANDS + 1


def test_n_ir_params():
    # 8 subbands × 3 parameters
    assert constants.N_IR_PARAMS == constants.N_IR_SUBBANDS * 3


def test_yaml_matches_python():
    """Ensure the YAML file matches the Python constants."""
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "constants.yaml"
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    assert cfg["sample_rate"] == constants.SAMPLE_RATE
    assert cfg["hop_length"] == constants.HOP_LENGTH
    assert cfg["n_fft"] == constants.N_FFT
    assert cfg["n_mels"] == constants.N_MELS
    assert cfg["d_content"] == constants.D_CONTENT
    assert cfg["d_speaker"] == constants.D_SPEAKER
