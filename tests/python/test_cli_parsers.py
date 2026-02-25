"""Tests for CLI argument parsers across all packages.

Tests build_parser() for each CLI module to verify:
- Required arguments are enforced
- Default values are correct
- Argument types are properly converted
"""

from __future__ import annotations

import pytest


class TestExtractFeaturesCLI:
    def test_build_parser(self):
        from tmrvc_data.cli.extract_features import build_parser

        parser = build_parser()
        assert parser.prog == "tmrvc-extract-features"

    def test_required_args(self):
        from tmrvc_data.cli.extract_features import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_args(self):
        from tmrvc_data.cli.extract_features import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--audio-dir", "/data/audio",
            "--cache-dir", "/data/cache",
            "--dataset", "test",
            "--f0-method", "rmvpe",
            "--device", "xpu",
            "--skip-existing",
            "-v",
        ])
        assert args.dataset == "test"
        assert args.f0_method == "rmvpe"
        assert args.device == "xpu"
        assert args.skip_existing is True
        assert args.verbose is True

    def test_defaults(self):
        from tmrvc_data.cli.extract_features import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--audio-dir", "/a", "--cache-dir", "/c", "--dataset", "x",
        ])
        assert args.split == "train"
        assert args.f0_method == "torchcrepe"
        assert args.device == "cpu"
        assert args.skip_existing is False


class TestPreprocessCLI:
    def test_build_parser(self):
        from tmrvc_data.cli.preprocess import build_parser

        parser = build_parser()
        assert parser.prog == "tmrvc-preprocess"

    def test_required_args(self):
        from tmrvc_data.cli.preprocess import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_valid_datasets(self):
        from tmrvc_data.cli.preprocess import build_parser

        parser = build_parser()
        for ds in ["vctk", "jvs", "libritts_r", "tsukuyomi"]:
            args = parser.parse_args([
                "--dataset", ds, "--raw-dir", "/r", "--cache-dir", "/c",
            ])
            assert args.dataset == ds

    def test_invalid_dataset_rejected(self):
        """Unknown dataset names are accepted by argparse but rejected by get_adapter()."""
        from tmrvc_data.cli.preprocess import build_parser
        from tmrvc_data.dataset_adapters import get_adapter

        parser = build_parser()
        # argparse no longer rejects unknown dataset names (choices removed)
        args = parser.parse_args(["--dataset", "invalid", "--raw-dir", "/r", "--cache-dir", "/c"])
        assert args.dataset == "invalid"
        # But get_adapter raises ValueError for unknown datasets
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_adapter(args.dataset)

    def test_defaults(self):
        from tmrvc_data.cli.preprocess import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--dataset", "vctk", "--raw-dir", "/r", "--cache-dir", "/c",
        ])
        assert args.content_teacher == "contentvec"
        assert args.max_utterances == 0
        assert args.subset == 1.0


class TestVerifyCacheCLI:
    def test_build_parser(self):
        from tmrvc_data.cli.verify_cache import build_parser

        parser = build_parser()
        assert parser.prog == "tmrvc-verify-cache"

    def test_required_args(self):
        from tmrvc_data.cli.verify_cache import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_args(self):
        from tmrvc_data.cli.verify_cache import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "/data/cache",
            "--dataset", "vctk",
            "--split", "dev",
            "-v",
        ])
        assert args.split == "dev"
        assert args.verbose is True


class TestExportCLI:
    def test_build_parser(self):
        from tmrvc_export.cli.export import build_parser

        parser = build_parser()
        assert parser.prog == "tmrvc-export"

    def test_required_args(self):
        from tmrvc_export.cli.export import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_args(self):
        from tmrvc_export.cli.export import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--checkpoint", "/ckpt.pt",
            "--output-dir", "/out",
            "--quantize",
            "--verify",
            "-v",
        ])
        assert args.quantize is True
        assert args.verify is True
        assert args.verbose is True

    def test_defaults(self):
        from tmrvc_export.cli.export import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--checkpoint", "/ckpt.pt", "--output-dir", "/out",
        ])
        assert args.quantize is False
        assert args.verify is False
        assert args.speaker_encoder_ckpt is None


class TestServeCLI:
    def test_build_parser(self):
        from tmrvc_serve.cli import build_parser

        parser = build_parser()
        assert parser.prog == "tmrvc-serve"

    def test_no_required_args(self):
        from tmrvc_serve.cli import build_parser

        parser = build_parser()
        # serve has no required args (all optional)
        args = parser.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_all_args(self):
        from tmrvc_serve.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--host", "0.0.0.0",
            "--port", "9000",
            "--device", "xpu",
            "--text-frontend", "tokenizer",
            "--reload",
            "--api-key", "test-key",
        ])
        assert args.host == "0.0.0.0"
        assert args.port == 9000
        assert args.device == "xpu"
        assert args.text_frontend == "tokenizer"
        assert args.reload is True
        assert args.api_key == "test-key"


class TestTrainTTSCLI:
    def test_build_parser(self):
        from tmrvc_train.cli.train_tts import build_parser

        parser = build_parser()
        assert parser.prog == "tmrvc-train-tts"

    def test_required_args(self):
        from tmrvc_train.cli.train_tts import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_args(self):
        from tmrvc_train.cli.train_tts import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "/data/cache",
            "--dataset", "jsut,ljspeech",
            "--lr", "5e-5",
            "--max-steps", "100000",
            "--batch-size", "16",
            "--device", "xpu",
            "--max-frames", "500",
            "--text-frontend", "tokenizer",
            "--wandb",
        ])
        assert args.lr == 5e-5
        assert args.max_steps == 100000
        assert args.batch_size == 16
        assert args.max_frames == 500
        assert args.text_frontend == "tokenizer"
        assert args.wandb is True

    def test_load_config_none(self):
        from tmrvc_train.cli.train_tts import _load_config

        assert _load_config(None) == {}

    def test_load_config_yaml(self, tmp_path):
        from tmrvc_train.cli.train_tts import _load_config

        config_file = tmp_path / "test.yaml"
        config_file.write_text("lr: 0.001\nmax_steps: 50000\n", encoding="utf-8")
        cfg = _load_config(config_file)
        assert cfg["lr"] == 0.001
        assert cfg["max_steps"] == 50000
