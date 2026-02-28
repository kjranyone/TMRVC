"""Tests for StyleTrainer."""

import torch
import pytest

from tmrvc_train.models.style_encoder import StyleEncoder
from tmrvc_train.style_trainer import StyleTrainer, StyleTrainerConfig


@pytest.fixture
def mock_batch():
    """Create a minimal batch for style training."""
    B = 4
    T = 50
    return {
        "mel": torch.randn(B, 80, T),
        "emotion_id": torch.randint(0, 12, (B,)),
        "vad": torch.randn(B, 3).clamp(-1, 1),
        "prosody": torch.randn(B, 3).clamp(-1, 1),
    }


@pytest.fixture
def trainer_components():
    model = StyleEncoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


class TestStyleTrainer:
    def test_train_step(self, trainer_components, mock_batch):
        model, optimizer = trainer_components
        config = StyleTrainerConfig(max_steps=10, log_every=5)
        trainer = StyleTrainer(
            model=model,
            optimizer=optimizer,
            dataloader=[],
            config=config,
        )
        losses = trainer.train_step(mock_batch)
        assert "total" in losses
        assert "emotion" in losses
        assert "vad" in losses
        assert "prosody" in losses
        assert "accuracy" in losses
        assert trainer.global_step == 1

    def test_train_step_without_vad(self, trainer_components):
        model, optimizer = trainer_components
        config = StyleTrainerConfig()
        trainer = StyleTrainer(model=model, optimizer=optimizer, dataloader=[], config=config)
        batch = {
            "mel": torch.randn(2, 80, 30),
            "emotion_id": torch.randint(0, 12, (2,)),
        }
        losses = trainer.train_step(batch)
        assert "emotion" in losses
        assert "vad" not in losses

    def test_save_load_checkpoint(self, tmp_path, trainer_components, mock_batch):
        model, optimizer = trainer_components
        config = StyleTrainerConfig(checkpoint_dir=str(tmp_path))
        trainer = StyleTrainer(model=model, optimizer=optimizer, dataloader=[], config=config)

        # Do a step
        trainer.train_step(mock_batch)
        assert trainer.global_step == 1

        # Save
        path = trainer.save_checkpoint()
        assert path.exists()

        # Load into new trainer
        model2 = StyleEncoder()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        trainer2 = StyleTrainer(model=model2, optimizer=optimizer2, dataloader=[], config=config)
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == 1

    def test_train_iter(self, trainer_components, mock_batch):
        model, optimizer = trainer_components
        config = StyleTrainerConfig(max_steps=3, log_every=1, save_every=100)

        dataloader = [mock_batch]  # single-batch "dataloader"
        trainer = StyleTrainer(
            model=model, optimizer=optimizer, dataloader=dataloader, config=config,
        )

        steps = list(trainer.train_iter())
        assert len(steps) == 3
        assert steps[-1][0] == 3  # global_step
        assert all("total" in s[1] for s in steps)

    def test_accuracy_range(self, trainer_components, mock_batch):
        model, optimizer = trainer_components
        config = StyleTrainerConfig()
        trainer = StyleTrainer(model=model, optimizer=optimizer, dataloader=[], config=config)
        losses = trainer.train_step(mock_batch)
        assert 0.0 <= losses["accuracy"] <= 1.0

    def test_grad_clip(self, trainer_components, mock_batch):
        model, optimizer = trainer_components
        config = StyleTrainerConfig(grad_clip=0.5)
        trainer = StyleTrainer(model=model, optimizer=optimizer, dataloader=[], config=config)
        trainer.train_step(mock_batch)
        # Should not raise — grad clipping applied
        assert trainer.global_step == 1
