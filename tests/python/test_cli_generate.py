"""Tests for tmrvc-generate-script CLI."""

import pytest


class TestBuildParser:
    def test_parser_creation(self):
        from tmrvc_serve.cli_generate import build_parser
        parser = build_parser()
        assert parser is not None

    def test_required_script_arg(self):
        from tmrvc_serve.cli_generate import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])  # script is required

    def test_default_args(self):
        from tmrvc_serve.cli_generate import build_parser
        parser = build_parser()
        args = parser.parse_args(["test.yaml"])
        assert str(args.script) == "test.yaml"
        assert args.output_dir is None
        assert args.speed == 1.0
        assert args.format == "wav"
        assert args.device == "cpu"

    def test_all_args(self):
        from pathlib import Path
        from tmrvc_serve.cli_generate import build_parser
        parser = build_parser()
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


class TestGenerateConstants:
    def test_tts_section_in_rust(self):
        """Verify TTS constants appear in generated Rust file."""
        from pathlib import Path
        rust_path = Path(__file__).resolve().parents[2] / "tmrvc-engine-rs" / "src" / "constants.rs"
        if not rust_path.exists():
            pytest.skip("Rust constants file not found")
        content = rust_path.read_text(encoding="utf-8")
        assert "D_STYLE" in content
        assert "N_STYLE_PARAMS" in content
        assert "D_TEXT_ENCODER" in content
        assert "PHONEME_VOCAB_SIZE" in content
        assert "N_EMOTION_CATEGORIES" in content
        assert "TTS extension" in content

    def test_tts_section_in_python(self):
        """Verify TTS constants appear in generated Python file."""
        from tmrvc_core._generated_constants import (
            D_STYLE,
            N_STYLE_PARAMS,
            D_TEXT_ENCODER,
            PHONEME_VOCAB_SIZE,
            N_EMOTION_CATEGORIES,
            D_F0_PREDICTOR,
            D_CONTENT_SYNTHESIZER,
            N_LANGUAGES,
        )
        assert D_STYLE == 32
        assert N_STYLE_PARAMS == 64
        assert D_TEXT_ENCODER == 256
        assert PHONEME_VOCAB_SIZE == 200
        assert N_EMOTION_CATEGORIES == 12
        assert D_F0_PREDICTOR == 128
        assert D_CONTENT_SYNTHESIZER == 256
        assert N_LANGUAGES == 4

    def test_python_string_list_quoted(self):
        """Verify string lists are properly quoted in generated Python."""
        from tmrvc_core._generated_constants import VOICE_SOURCE_PARAM_NAMES
        assert isinstance(VOICE_SOURCE_PARAM_NAMES, list)
        assert all(isinstance(s, str) for s in VOICE_SOURCE_PARAM_NAMES)
        assert "breathiness_low" in VOICE_SOURCE_PARAM_NAMES
