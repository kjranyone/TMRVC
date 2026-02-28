"""Tests for tmrvc-train-style and tmrvc-create-character CLIs."""

import pytest
from pathlib import Path


class TestTrainStyleParser:
    def test_parser_creation(self):
        from tmrvc_train.cli.train_style import build_parser
        parser = build_parser()
        assert parser is not None

    def test_required_cache_dir(self):
        from tmrvc_train.cli.train_style import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_args(self):
        from tmrvc_train.cli.train_style import build_parser
        parser = build_parser()
        args = parser.parse_args(["--cache-dir", "data/cache", "--dataset", "expresso"])
        assert args.cache_dir == Path("data/cache")
        assert args.dataset == "expresso"
        assert args.device == "cpu"
        assert args.lr is None
        assert args.wandb is False

    def test_all_args(self):
        from tmrvc_train.cli.train_style import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "data/cache",
            "--dataset", "expresso,jvnv",
            "--lr", "1e-3",
            "--max-steps", "10000",
            "--batch-size", "32",
            "--device", "xpu",
            "--wandb",
            "-v",
        ])
        assert args.lr == 1e-3
        assert args.max_steps == 10000
        assert args.batch_size == 32
        assert args.device == "xpu"
        assert args.wandb is True
        assert args.verbose is True


class TestCreateCharacterParser:
    def test_parser_creation(self):
        from tmrvc_export.cli.create_character import build_parser
        parser = build_parser()
        assert parser is not None

    def test_required_speaker_file(self):
        from tmrvc_export.cli.create_character import build_parser
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_args(self):
        from tmrvc_export.cli.create_character import build_parser
        parser = build_parser()
        args = parser.parse_args(["models/test.tmrvc_speaker"])
        assert args.speaker_file == Path("models/test.tmrvc_speaker")
        assert args.output is None
        assert args.name == ""
        assert args.language == "ja"

    def test_full_args(self):
        from tmrvc_export.cli.create_character import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "models/test.tmrvc_speaker",
            "-o", "models/test.tmrvc_character",
            "--name", "Test",
            "--personality", "Calm",
            "--voice-description", "Low voice",
            "--language", "en",
        ])
        assert args.output == Path("models/test.tmrvc_character")
        assert args.name == "Test"
        assert args.personality == "Calm"
        assert args.language == "en"


class TestEmotionDataset:
    def test_import(self):
        from tmrvc_data.emotion_dataset import EmotionDataset, create_emotion_dataloader
        assert EmotionDataset is not None
        assert create_emotion_dataloader is not None


class TestPreprocessEmotionParser:
    def test_parser(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "preprocess_emotion",
            str(Path(__file__).resolve().parents[2] / "scripts" / "preprocess_emotion.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        parser = mod.build_parser()
        args = parser.parse_args(["--dataset", "expresso", "--raw-dir", "/tmp", "--cache-dir", "/tmp/cache"])
        assert args.dataset == "expresso"
