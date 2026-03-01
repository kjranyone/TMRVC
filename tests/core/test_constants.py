"""Tests for constant integrity and internal consistency."""

import pytest
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


# Legacy feature-based tests (deprecated in UCLM v2)
@pytest.mark.skip(reason="D_VOCODER_FEATURES deprecated in UCLM v2")
def test_d_vocoder_features_matches_freq_bins():
    pass


@pytest.mark.skip(reason="LoRA calculation changed in UCLM v2")
def test_lora_delta_size():
    pass


@pytest.mark.skip(reason="IR params deprecated in UCLM v2")
def test_ir_subband_edges_length():
    pass


@pytest.mark.skip(reason="IR params deprecated in UCLM v2")
def test_n_ir_params():
    pass


def test_n_voice_source_params():
    assert constants.N_VOICE_SOURCE_PARAMS == 8


@pytest.mark.skip(reason="Acoustic params calculation changed in UCLM v2")
def test_n_acoustic_params():
    pass


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
