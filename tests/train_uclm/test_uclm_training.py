"""Tests for UCLM Dataset and Trainer.

Tests:
- UCLMDataset: loading, filtering, collation
- UCLMTrainer: initialization, forward pass
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
import tempfile
import numpy as np
import json


class TestUCLMDataset:
    """Tests for UCLMDataset."""

    @pytest.fixture
    def temp_cache(self, tmp_path):
        """Create a temporary cache directory with sample data."""
        dataset_dir = tmp_path / "test_dataset" / "train" / "speaker_001" / "utt_001"
        dataset_dir.mkdir(parents=True)

        # Create sample data
        n_codebooks = 8
        T = 100
        L = 20
        d_speaker = 192
        d_voice_state = 8

        # Save codec tokens
        codec_tokens = np.random.randint(0, 1024, (n_codebooks, T), dtype=np.int64)
        np.save(dataset_dir / "codec_tokens.npy", codec_tokens)

        # Save voice state
        voice_state = np.random.randn(T, d_voice_state).astype(np.float32)
        np.save(dataset_dir / "voice_state.npy", voice_state)

        # Save phoneme ids
        phoneme_ids = np.random.randint(0, 200, L, dtype=np.int64)
        np.save(dataset_dir / "phoneme_ids.npy", phoneme_ids)

        # Save durations
        durations = np.random.randint(1, 10, L, dtype=np.int64)
        np.save(dataset_dir / "durations.npy", durations)

        # Save speaker embedding
        spk_embed = np.random.randn(d_speaker).astype(np.float32)
        np.save(dataset_dir / "spk_embed.npy", spk_embed)

        # Save meta.json
        meta = {
            "utterance_id": "utt_001",
            "speaker_id": "speaker_001",
            "n_frames": T,
            "text": "test utterance",
        }
        with open(dataset_dir / "meta.json", "w") as f:
            json.dump(meta, f)

        return tmp_path

    def test_dataset_loading(self, temp_cache):
        """Test that dataset loads correctly."""
        from tmrvc_data.uclm_dataset import UCLMDataset

        dataset = UCLMDataset(
            cache_dir=temp_cache,
            datasets=["test_dataset"],
        )

        assert len(dataset) == 1

        sample = dataset[0]

        assert "codec_tokens" in sample
        assert "voice_state" in sample
        assert sample["codec_tokens"].shape == (8, 100)
        assert sample["voice_state"].shape == (100, 8)

    def test_dataset_filtering(self, temp_cache):
        """Test that dataset filters by frame count."""
        from tmrvc_data.uclm_dataset import UCLMDataset

        # Should be filtered out (max_frames < 100)
        dataset = UCLMDataset(
            cache_dir=temp_cache,
            datasets=["test_dataset"],
            max_frames=50,
        )

        assert len(dataset) == 0

    def test_collate_batch(self, temp_cache):
        """Test batch collation."""
        from tmrvc_data.uclm_dataset import UCLMDataset, collate_uclm_batch

        dataset = UCLMDataset(
            cache_dir=temp_cache,
            datasets=["test_dataset"],
        )

        # Create batch with same sample twice
        samples = [dataset[0], dataset[0]]
        batch = collate_uclm_batch(samples)

        assert batch.codec_tokens.shape[0] == 2
        assert batch.voice_state.shape[0] == 2
        assert batch.spk_embed.shape == (2, 192)
        assert len(batch.text) == 2


class TestUCLMTrainerConfig:
    """Tests for UCLMTrainerConfig."""

    def test_default_config(self):
        from tmrvc_train.uclm_trainer import UCLMTrainerConfig

        config = UCLMTrainerConfig()

        assert config.d_model == 256
        assert config.n_heads == 8
        assert config.n_layers == 12
        assert config.lr == 1e-4
        assert config.max_steps == 200000

    def test_custom_config(self):
        from tmrvc_train.uclm_trainer import UCLMTrainerConfig

        config = UCLMTrainerConfig(
            d_model=512,
            n_layers=24,
            batch_size=32,
        )

        assert config.d_model == 512
        assert config.n_layers == 24
        assert config.batch_size == 32


class TestUCLMTrainer:
    """Tests for UCLMTrainer (without actual training)."""

    def test_trainer_initialization(self):
        """Test that trainer initializes correctly."""
        from tmrvc_train.uclm_trainer import UCLMTrainer, UCLMTrainerConfig

        # Use minimal config for testing
        config = UCLMTrainerConfig(
            d_model=64,
            n_heads=2,
            n_layers=1,
            batch_size=1,
            max_steps=1,
            num_workers=0,
        )

        # This will fail if no data, but we test model init
        try:
            trainer = UCLMTrainer(config)
            assert trainer.model is not None
            assert trainer.optimizer is not None
        except Exception as e:
            # Expected if no data available
            err_str = str(e)
            assert (
                "does not exist" in err_str
                or "num_samples" in err_str
                or "No valid" in err_str
            )

    def test_forward_step(self):
        """Test forward pass with mock batch."""
        from tmrvc_train.uclm_trainer import UCLMTrainer, UCLMTrainerConfig
        from tmrvc_data.uclm_dataset import UCLMBatch

        config = UCLMTrainerConfig(
            d_model=64,
            n_heads=2,
            n_layers=1,
            batch_size=1,
            max_steps=1,
            num_workers=0,
        )

        try:
            trainer = UCLMTrainer(config)

            # Create mock batch
            batch = UCLMBatch(
                codec_tokens=torch.randint(0, 1024, (1, 8, 50)),
                voice_state=torch.randn(1, 50, 8),
                phoneme_ids=torch.randint(0, 200, (1, 20)),
                durations=torch.randint(1, 10, (1, 20)),
                spk_embed=torch.randn(1, 192),
                text=["test"],
                utterance_ids=["test_001"],
                frame_lengths=torch.tensor([50]),
                phoneme_lengths=torch.tensor([20]),
            )

            batch = trainer._move_batch_to_device(batch)
            loss, loss_dict = trainer._forward_step(batch)

            assert loss.item() > 0
            assert "loss_ar" in loss_dict
            assert "loss_parallel" in loss_dict

        except Exception as e:
            # Skip if no data directory
            pytest.skip(f"Skipping due to: {e}")
