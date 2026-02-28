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
    # Per-layer: (D_SPEAKER + N_ACOUSTIC_PARAMS) * LORA_RANK + LORA_RANK * (D_CONVERTER_HIDDEN * 2)
    # = 224 * 4 + 4 * 768 = 3968
    # 4 layers × 3968 = 15872
    d_in = constants.D_SPEAKER + constants.N_ACOUSTIC_PARAMS
    per_layer = d_in * constants.LORA_RANK + constants.LORA_RANK * (constants.D_CONVERTER_HIDDEN * 2)
    expected = constants.N_LORA_LAYERS * per_layer
    assert constants.LORA_DELTA_SIZE == expected


def test_ir_subband_edges_length():
    # n_ir_subbands + 1 edges
    assert len(constants.IR_SUBBAND_EDGES_HZ) == constants.N_IR_SUBBANDS + 1


def test_n_ir_params():
    # 8 subbands × 3 parameters
    assert constants.N_IR_PARAMS == constants.N_IR_SUBBANDS * 3


def test_n_voice_source_params():
    assert constants.N_VOICE_SOURCE_PARAMS == 8


def test_n_acoustic_params():
    # n_ir_params + n_voice_source_params
    assert constants.N_ACOUSTIC_PARAMS == constants.N_IR_PARAMS + constants.N_VOICE_SOURCE_PARAMS


def test_yaml_matches_python():
    """Ensure the YAML file matches the Python constants."""
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[2] / "configs" / "constants.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["sample_rate"] == constants.SAMPLE_RATE
    assert cfg["hop_length"] == constants.HOP_LENGTH
    assert cfg["n_fft"] == constants.N_FFT
    assert cfg["n_mels"] == constants.N_MELS
    assert cfg["d_content"] == constants.D_CONTENT
    assert cfg["d_speaker"] == constants.D_SPEAKER
