"""Tests for pseudo-labeling pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tmrvc_core.constants import N_MELS
from tmrvc_data.pseudo_label import PseudoLabeler, PseudoLabelStats


class TestPseudoLabelStats:
    def test_defaults(self):
        s = PseudoLabelStats()
        assert s.total == 0
        assert s.labeled == 0
        assert s.label_rate == 0.0

    def test_label_rate(self):
        s = PseudoLabelStats(total=100, labeled=80)
        assert s.label_rate == pytest.approx(0.8)

    def test_label_rate_zero_total(self):
        s = PseudoLabelStats(total=0, labeled=0)
        assert s.label_rate == 0.0


class TestNormalizeFrames:
    @pytest.fixture
    def labeler(self):
        """Create a PseudoLabeler with mocked classifier."""
        with patch(
            "tmrvc_data.pseudo_label.PseudoLabeler.__init__",
            lambda self, **kw: None,
        ):
            obj = PseudoLabeler.__new__(PseudoLabeler)
            obj.max_frames = 200
            obj.confidence_threshold = 0.8
            obj.device = torch.device("cpu")
            return obj

    def test_crop_long(self, labeler):
        mel = np.random.randn(N_MELS, 500).astype(np.float32)
        result = labeler._normalize_frames(mel)
        assert result.shape == (N_MELS, 200)

    def test_pad_short(self, labeler):
        mel = np.random.randn(N_MELS, 50).astype(np.float32)
        result = labeler._normalize_frames(mel)
        assert result.shape == (N_MELS, 200)
        # Padded region should be zeros
        np.testing.assert_array_equal(result[:, 50:], 0.0)

    def test_exact_length(self, labeler):
        mel = np.random.randn(N_MELS, 200).astype(np.float32)
        result = labeler._normalize_frames(mel)
        assert result.shape == (N_MELS, 200)
        np.testing.assert_array_equal(result, mel)


class TestProcessBatch:
    @pytest.fixture
    def labeler_with_mock_classifier(self):
        """Create PseudoLabeler with a real EmotionClassifier."""
        from tmrvc_train.models.emotion_classifier import EmotionClassifier

        with patch(
            "tmrvc_data.pseudo_label.PseudoLabeler.__init__",
            lambda self, **kw: None,
        ):
            obj = PseudoLabeler.__new__(PseudoLabeler)
            obj.max_frames = 50
            obj.confidence_threshold = 0.5
            obj.device = torch.device("cpu")
            obj.classifier = EmotionClassifier()
            obj.classifier.eval()
            return obj

    def test_writes_json(self, tmp_path, labeler_with_mock_classifier):
        labeler = labeler_with_mock_classifier
        # Create a mock utterance directory with mel
        utt_dir = tmp_path / "spk001" / "utt001"
        utt_dir.mkdir(parents=True)
        mel = np.random.randn(N_MELS, 100).astype(np.float32)
        np.save(utt_dir / "mel.npy", mel)

        stats = PseudoLabelStats(total=1)
        labeler._process_batch([utt_dir], stats, overwrite=False)

        # Either labeled or skipped_low_confidence
        assert stats.labeled + stats.skipped_low_confidence == 1
        if stats.labeled == 1:
            pseudo_path = utt_dir / "pseudo_emotion.json"
            assert pseudo_path.exists()
            with open(pseudo_path, encoding="utf-8") as f:
                data = json.load(f)
            assert "emotion_id" in data
            assert "emotion" in data
            assert "confidence" in data
            assert "vad" in data
            assert data["source"] == "pseudo_label"

    def test_skip_existing(self, tmp_path, labeler_with_mock_classifier):
        labeler = labeler_with_mock_classifier
        utt_dir = tmp_path / "spk001" / "utt002"
        utt_dir.mkdir(parents=True)
        mel = np.random.randn(N_MELS, 100).astype(np.float32)
        np.save(utt_dir / "mel.npy", mel)
        # Create existing pseudo_emotion.json
        (utt_dir / "pseudo_emotion.json").write_text("{}", encoding="utf-8")

        stats = PseudoLabelStats(total=1)
        labeler._process_batch([utt_dir], stats, overwrite=False)
        assert stats.skipped_existing == 1

    def test_overwrite_existing(self, tmp_path, labeler_with_mock_classifier):
        labeler = labeler_with_mock_classifier
        utt_dir = tmp_path / "spk001" / "utt003"
        utt_dir.mkdir(parents=True)
        mel = np.random.randn(N_MELS, 100).astype(np.float32)
        np.save(utt_dir / "mel.npy", mel)
        (utt_dir / "pseudo_emotion.json").write_text("{}", encoding="utf-8")

        stats = PseudoLabelStats(total=1)
        labeler._process_batch([utt_dir], stats, overwrite=True)
        # Should attempt to process (not skip)
        assert stats.skipped_existing == 0


class TestLabelDataset:
    @pytest.fixture
    def labeler(self):
        from tmrvc_train.models.emotion_classifier import EmotionClassifier

        with patch(
            "tmrvc_data.pseudo_label.PseudoLabeler.__init__",
            lambda self, **kw: None,
        ):
            obj = PseudoLabeler.__new__(PseudoLabeler)
            obj.max_frames = 50
            obj.confidence_threshold = 0.3  # low threshold to get labels
            obj.device = torch.device("cpu")
            obj.classifier = EmotionClassifier()
            obj.classifier.eval()
            return obj

    def test_missing_dir(self, tmp_path, labeler):
        stats = labeler.label_dataset(
            cache_dir=tmp_path,
            dataset="nonexistent",
            split="train",
        )
        assert stats.total == 0

    def test_label_full_dataset(self, tmp_path, labeler):
        # Create cache structure: cache_dir/dataset/split/spk/utt/mel.npy
        ds_dir = tmp_path / "test_ds" / "train"
        for spk_idx in range(2):
            for utt_idx in range(3):
                utt_dir = ds_dir / f"spk{spk_idx:03d}" / f"utt{utt_idx:03d}"
                utt_dir.mkdir(parents=True)
                mel = np.random.randn(N_MELS, 80).astype(np.float32)
                np.save(utt_dir / "mel.npy", mel)

        stats = labeler.label_dataset(
            cache_dir=tmp_path,
            dataset="test_ds",
            split="train",
            batch_size=4,
        )
        assert stats.total == 6
        assert stats.labeled + stats.skipped_low_confidence == 6

    def test_empty_dir(self, tmp_path, labeler):
        ds_dir = tmp_path / "empty_ds" / "train"
        ds_dir.mkdir(parents=True)
        stats = labeler.label_dataset(
            cache_dir=tmp_path,
            dataset="empty_ds",
            split="train",
        )
        assert stats.total == 0


class TestCLIParser:
    def test_train_subcommand(self):
        from scripts.apply_pseudo_labels import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--cache-dir", "/data/cache",
            "--datasets", "expresso,jvnv",
            "--output", "/ckpt/emotion_cls.pt",
            "--device", "xpu",
        ])
        assert args.command == "train"
        assert args.cache_dir == Path("/data/cache")
        assert args.datasets == "expresso,jvnv"
        assert args.device == "xpu"
        assert args.max_steps == 10000
        assert args.batch_size == 64
        assert args.lr == 1e-3

    def test_label_subcommand(self):
        from scripts.apply_pseudo_labels import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "label",
            "--cache-dir", "/data/cache",
            "--classifier", "/ckpt/emotion_cls.pt",
            "--datasets", "vctk,jvs",
            "--confidence", "0.9",
            "--overwrite",
        ])
        assert args.command == "label"
        assert args.confidence == 0.9
        assert args.overwrite is True
        assert args.batch_size == 32

    def test_train_required_args(self):
        from scripts.apply_pseudo_labels import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])

    def test_label_required_args(self):
        from scripts.apply_pseudo_labels import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["label"])

    def test_no_subcommand(self):
        from scripts.apply_pseudo_labels import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])
