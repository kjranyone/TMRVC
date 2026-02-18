"""Tests for scripts/generate_constants.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

# Import the generator functions directly
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_constants import (
    generate_cpp,
    generate_python,
    generate_rust,
    _write_if_changed,
    _RUST_NAME_MAP,
    _RUNTIME_KEYS,
    YAML_PATH,
)


@pytest.fixture
def sample_cfg() -> dict:
    """Minimal config for testing generators."""
    return {
        "sample_rate": 24000,
        "n_fft": 1024,
        "hop_length": 240,
        "n_mels": 80,
        "mel_fmin": 0.0,
        "log_floor": 1e-10,
        "d_content": 256,
        "ir_subband_edges_hz": [0, 375, 750, 1500, 3000],
    }


@pytest.fixture
def full_cfg() -> dict:
    """Load the real constants.yaml."""
    with open(YAML_PATH) as f:
        return yaml.safe_load(f)


class TestGeneratePython:
    def test_header(self, sample_cfg):
        out = generate_python(sample_cfg)
        assert "Auto-generated constants" in out
        assert "DO NOT EDIT MANUALLY" in out

    def test_integer_constant(self, sample_cfg):
        out = generate_python(sample_cfg)
        assert "SAMPLE_RATE = 24000" in out

    def test_float_constant(self, sample_cfg):
        out = generate_python(sample_cfg)
        assert "MEL_FMIN = 0.0" in out

    def test_scientific_float(self, sample_cfg):
        out = generate_python(sample_cfg)
        assert "LOG_FLOOR = 1e-10" in out

    def test_list_constant(self, sample_cfg):
        out = generate_python(sample_cfg)
        assert "IR_SUBBAND_EDGES_HZ = [0, 375, 750, 1500, 3000]" in out

    def test_trailing_newline(self, sample_cfg):
        out = generate_python(sample_cfg)
        assert out.endswith("\n")

    def test_all_keys_present(self, sample_cfg):
        out = generate_python(sample_cfg)
        for key in sample_cfg:
            assert key.upper() in out


class TestGenerateCpp:
    def test_pragma_once(self, sample_cfg):
        out = generate_cpp(sample_cfg)
        assert "#pragma once" in out

    def test_namespace(self, sample_cfg):
        out = generate_cpp(sample_cfg)
        assert "namespace tmrvc {" in out
        assert "}  // namespace tmrvc" in out

    def test_integer_constexpr(self, sample_cfg):
        out = generate_cpp(sample_cfg)
        assert "constexpr int SAMPLE_RATE = 24000;" in out

    def test_float_constexpr(self, sample_cfg):
        out = generate_cpp(sample_cfg)
        assert "constexpr float MEL_FMIN = 0.0f;" in out

    def test_array_constexpr(self, sample_cfg):
        out = generate_cpp(sample_cfg)
        assert "constexpr std::array<int, 5> IR_SUBBAND_EDGES_HZ" in out

    def test_includes_array_header(self, sample_cfg):
        out = generate_cpp(sample_cfg)
        assert "#include <array>" in out


class TestGenerateRust:
    def test_header(self, sample_cfg):
        out = generate_rust(sample_cfg)
        assert "Auto-generated from configs/constants.yaml" in out
        assert "DO NOT EDIT MANUALLY" in out

    def test_dead_code_allow(self, sample_cfg):
        out = generate_rust(sample_cfg)
        assert "#![allow(dead_code)]" in out

    def test_integer_const(self, sample_cfg):
        out = generate_rust(sample_cfg)
        assert "pub const SAMPLE_RATE: usize = 24000;" in out

    def test_float_const(self, sample_cfg):
        out = generate_rust(sample_cfg)
        assert "pub const MEL_FMIN: f32 = 0.0;" in out

    def test_name_mapping(self, full_cfg):
        """Rust name overrides should apply for backward compat."""
        out = generate_rust(full_cfg)
        assert "pub const CONTENT_ENC_STATE_FRAMES:" in out
        assert "pub const IR_EST_STATE_FRAMES:" in out
        # The original long name should NOT appear
        assert "CONTENT_ENCODER_STATE_FRAMES" not in out
        assert "IR_ESTIMATOR_STATE_FRAMES" not in out

    def test_section_headers(self, full_cfg):
        out = generate_rust(full_cfg)
        assert "// --- Audio parameters ---" in out
        assert "// --- Model dimensions ---" in out
        assert "// --- Inference parameters ---" in out
        assert "// --- State tensor context lengths ---" in out

    def test_derived_constants(self, full_cfg):
        out = generate_rust(full_cfg)
        assert "pub const RING_BUFFER_CAPACITY: usize = 4096;" in out
        assert "pub const PAST_CONTEXT: usize = WINDOW_LENGTH - HOP_LENGTH;" in out

    def test_only_runtime_keys(self, full_cfg):
        """Rust output should only contain runtime-relevant keys."""
        out = generate_rust(full_cfg)
        # Training-only keys should not appear
        assert "FLOW_MATCHING_STEPS" not in out
        assert "DEFAULT_BATCH_SIZE" not in out
        assert "D_CONTENT_VEC" not in out


class TestWriteIfChanged:
    def test_creates_new_file(self, tmp_path):
        p = tmp_path / "sub" / "file.txt"
        changed = _write_if_changed(p, "hello")
        assert changed is True
        assert p.read_text() == "hello"

    def test_no_change_when_same(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello")
        changed = _write_if_changed(p, "hello")
        assert changed is False

    def test_updates_when_different(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("old")
        changed = _write_if_changed(p, "new")
        assert changed is True
        assert p.read_text() == "new"

    def test_creates_parent_dirs(self, tmp_path):
        p = tmp_path / "a" / "b" / "c" / "file.txt"
        changed = _write_if_changed(p, "deep")
        assert changed is True
        assert p.read_text() == "deep"


class TestRealYaml:
    """Test that generate functions produce valid output from the real YAML."""

    def test_python_is_valid_syntax(self, full_cfg):
        out = generate_python(full_cfg)
        compile(out, "<generated>", "exec")

    def test_rust_has_all_runtime_keys(self, full_cfg):
        out = generate_rust(full_cfg)
        for key in _RUNTIME_KEYS:
            if key in full_cfg:
                rust_name = _RUST_NAME_MAP.get(key, key.upper())
                assert rust_name in out, f"Missing {rust_name} for key {key}"

    def test_cpp_has_all_keys(self, full_cfg):
        out = generate_cpp(full_cfg)
        for key in full_cfg:
            assert key.upper() in out

    def test_generate_constants_includes_hq(self, full_cfg):
        """Generated Python/Rust should include HQ mode constants."""
        py_out = generate_python(full_cfg)
        assert "MAX_LOOKAHEAD_HOPS = 6" in py_out
        assert "CONVERTER_HQ_STATE_FRAMES = 46" in py_out
        assert "HQ_THRESHOLD_Q = 0.3" in py_out
        assert "CROSSFADE_FRAMES = 10" in py_out

        rust_out = generate_rust(full_cfg)
        assert "MAX_LOOKAHEAD_HOPS" in rust_out
        assert "CONVERTER_HQ_STATE_FRAMES" in rust_out
        assert "HQ_THRESHOLD_Q" in rust_out
        assert "CROSSFADE_FRAMES" in rust_out
        assert "// --- Lookahead / HQ mode ---" in rust_out
