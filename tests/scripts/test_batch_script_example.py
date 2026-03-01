"""Tests for the batch script generation example CLI (UCLM v2)."""

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _PROJECT_ROOT / "examples" / "batch_script_generation"

def _import_generate():
    if str(_EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(_EXAMPLES_DIR))
    import generate
    return generate

class TestBuildParser:
    def test_default_args(self):
        gen = _import_generate()
        parser = gen.build_parser()
        args = parser.parse_args(["test.yaml"])
        assert str(args.script) == "test.yaml"
        assert args.output_dir is None
        assert str(args.uclm_checkpoint) == "checkpoints/uclm/uclm_latest.pt"

    def test_all_args(self):
        gen = _import_generate()
        parser = gen.build_parser()
        args = parser.parse_args([
            "script.yaml",
            "--output-dir", "out/",
            "--uclm-checkpoint", "ckpt/uclm.pt",
            "--codec-checkpoint", "ckpt/codec.pt",
            "--device", "cuda",
            "--format", "flac",
            "-v",
        ])
        assert str(args.script) == "script.yaml"
        assert str(args.output_dir) == "out"
        assert str(args.uclm_checkpoint) == "ckpt/uclm.pt"
        assert str(args.codec_checkpoint) == "ckpt/codec.pt"
        assert args.device == "cuda"
        assert args.format == "flac"
        assert args.verbose is True
