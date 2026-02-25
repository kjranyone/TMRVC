"""Tests for examples/batch_script_generation/generate.py parser."""

import sys
from pathlib import Path

import pytest

# Add examples to path so we can import the standalone script
_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "batch_script_generation"


def _import_generate():
    """Import generate.py from examples directory."""
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    import generate
    return generate


class TestBuildParser:
    def test_parser_creation(self):
        gen = _import_generate()
        parser = gen.build_parser()
        assert parser is not None

    def test_required_script_arg(self):
        gen = _import_generate()
        parser = gen.build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # script is required

    def test_default_args(self):
        gen = _import_generate()
        parser = gen.build_parser()
        args = parser.parse_args(["test.yaml"])
        assert str(args.script) == "test.yaml"
        assert args.output_dir is None
        assert args.speed == 1.0
        assert args.format == "wav"
        assert args.device == "cpu"

    def test_all_args(self):
        gen = _import_generate()
        parser = gen.build_parser()
        args = parser.parse_args([
            "script.yaml",
            "--output-dir", "out/",
            "--tts-checkpoint", "ckpt/tts.pt",
            "--vc-checkpoint", "ckpt/vc.pt",
            "--device", "xpu",
            "--speed", "1.5",
            "--format", "flac",
            "--sample-rate", "48000",
            "-v",
        ])
        assert args.script == Path("script.yaml")
        assert args.output_dir == Path("out")
        assert args.tts_checkpoint == Path("ckpt/tts.pt")
        assert args.speed == 1.5
        assert args.format == "flac"
        assert args.sample_rate == 48000
        assert args.verbose is True
