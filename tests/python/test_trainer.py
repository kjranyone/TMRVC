"""Tests for TeacherTrainer and train_iter() generator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
import pytest

from tmrvc_core.constants import D_CONTENT_VEC, D_SPEAKER, N_MELS
from tmrvc_core.types import TrainingBatch
from tmrvc_train.diffusion import FlowMatchingScheduler
from tmrvc_train.models.teacher_unet import TeacherUNet
from tmrvc_train.trainer import ReflowTrainer, TeacherTrainer, TrainerConfig


def _make_batch(batch_size: int = 2, n_frames: int = 64) -> TrainingBatch:
    """Create a synthetic TrainingBatch."""
    return TrainingBatch(
        content=torch.randn(batch_size, D_CONTENT_VEC, n_frames),
        f0=torch.randn(batch_size, 1, n_frames),
        spk_embed=torch.randn(batch_size, D_SPEAKER),
        mel_target=torch.randn(batch_size, N_MELS, n_frames),
        lengths=torch.full((batch_size,), n_frames, dtype=torch.long),
    )


class TestTrainerConfig:
    def test_defaults(self):
        cfg = TrainerConfig()
        assert cfg.phase == "0"
        assert cfg.lr == 2e-4
        assert cfg.max_steps == 100_000
        assert cfg.grad_clip == 1.0

    def test_custom(self):
        cfg = TrainerConfig(phase="1b", lr=5e-5, max_steps=200_000)
        assert cfg.phase == "1b"
        assert cfg.lr == 5e-5


class TestTeacherTrainer:
    @pytest.fixture
    def trainer(self, tmp_path):
        teacher = TeacherUNet()
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)

        batches = [_make_batch() for _ in range(3)]
        dataloader = batches  # Simple iterable

        config = TrainerConfig(
            max_steps=5,
            save_every=3,
            log_every=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        return TeacherTrainer(teacher, scheduler, optimizer, dataloader, config)

    def test_train_step_returns_losses(self, trainer):
        batch = _make_batch()
        losses = trainer.train_step(batch)
        assert "flow" in losses
        assert "total" in losses
        assert isinstance(losses["total"], float)

    def test_train_step_phase_1b_extra_losses(self, tmp_path):
        teacher = TeacherUNet()
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=5e-5)
        config = TrainerConfig(
            phase="1b",
            max_steps=5,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        trainer = TeacherTrainer(
            teacher, scheduler, optimizer, [_make_batch()], config,
        )
        losses = trainer.train_step(_make_batch())
        assert "stft" in losses
        assert "spk" in losses

    def test_train_iter_yields_step_and_losses(self, trainer):
        results = list(trainer.train_iter())
        # max_steps=5, dataloader has 3 batches → exhausts, reloops, stops at 5
        assert len(results) == 5
        for step, losses in results:
            assert isinstance(step, int)
            assert "total" in losses

    def test_train_iter_increments_global_step(self, trainer):
        results = list(trainer.train_iter())
        steps = [s for s, _ in results]
        assert steps == [1, 2, 3, 4, 5]
        assert trainer.global_step == 5

    def test_train_iter_can_be_interrupted(self, trainer):
        """Caller can stop iteration early (simulating GUI cancel)."""
        count = 0
        for step, losses in trainer.train_iter():
            count += 1
            if count >= 2:
                break
        assert trainer.global_step == 2
        assert count == 2

    def test_train_iter_saves_checkpoint(self, trainer, tmp_path):
        """Checkpoint should be saved at save_every intervals."""
        list(trainer.train_iter())
        ckpt_dir = tmp_path / "ckpt"
        ckpts = list(ckpt_dir.glob("teacher_step*.pt"))
        # save_every=3, so we should have step 3
        assert any("step3" in str(p) for p in ckpts)

    def test_train_epoch_respects_max_steps(self, trainer):
        trainer.train_epoch()
        assert trainer.global_step == 5

    def test_save_and_load_checkpoint(self, trainer, tmp_path):
        # Train a few steps
        for step, _ in trainer.train_iter():
            if step >= 3:
                break

        path = trainer.save_checkpoint()
        assert path.exists()

        # Create new trainer and load
        teacher2 = TeacherUNet()
        scheduler2 = FlowMatchingScheduler()
        optimizer2 = torch.optim.AdamW(teacher2.parameters(), lr=2e-4)
        config2 = TrainerConfig(
            checkpoint_dir=str(tmp_path / "ckpt2"),
        )
        trainer2 = TeacherTrainer(
            teacher2, scheduler2, optimizer2, [], config2,
        )
        trainer2.load_checkpoint(path)
        assert trainer2.global_step == 3


class TestTrainTeacherCLI:
    """Tests for CLI argument parsing and config loading."""

    def test_build_parser_defaults(self):
        from tmrvc_train.cli.train_teacher import build_parser

        parser = build_parser()
        args = parser.parse_args(["--cache-dir", "/tmp/cache", "--dataset", "vctk"])
        assert args.cache_dir == Path("/tmp/cache")
        assert args.dataset == "vctk"
        assert args.phase == "0"
        assert args.lr is None
        assert args.config is None

    def test_build_parser_all_args(self):
        from tmrvc_train.cli.train_teacher import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "/data",
            "--dataset", "vctk",
            "--phase", "1b",
            "--lr", "1e-5",
            "--max-steps", "50000",
            "--batch-size", "32",
            "--config", "configs/train_teacher.yaml",
        ])
        assert args.dataset == "vctk"
        assert args.phase == "1b"
        assert args.lr == 1e-5
        assert args.max_steps == 50000
        assert args.batch_size == 32
        assert args.config == Path("configs/train_teacher.yaml")

    def test_load_config_none(self):
        from tmrvc_train.cli.train_teacher import _load_config

        result = _load_config(None, "0")
        assert result == {}

    def test_load_config_yaml(self, tmp_path):
        from tmrvc_train.cli.train_teacher import _load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(
            "batch_size: 32\n"
            "phases:\n"
            "  '0':\n"
            "    lr: 0.001\n"
            "    description: test phase\n"
        )
        result = _load_config(cfg_file, "0")
        assert result["lr"] == 0.001
        assert result["batch_size"] == 32
        # description should be popped
        assert "description" not in result
        # phases should be popped
        assert "phases" not in result

    def test_load_config_missing_phase(self, tmp_path):
        from tmrvc_train.cli.train_teacher import _load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text("batch_size: 16\n")
        result = _load_config(cfg_file, "1a")
        assert result["batch_size"] == 16

    def test_default_lr(self):
        from tmrvc_train.cli.train_teacher import _default_lr

        assert _default_lr("0") == 2e-4
        assert _default_lr("1a") == 1e-4
        assert _default_lr("1b") == 5e-5
        assert _default_lr("2") == 5e-5


class TestDistillCLI:
    """Tests for distill CLI argument parsing and config loading."""

    def test_build_parser_defaults(self):
        from tmrvc_train.cli.distill import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "/tmp/cache",
            "--dataset", "vctk",
            "--teacher-ckpt", "teacher.pt",
        ])
        assert args.cache_dir == Path("/tmp/cache")
        assert args.dataset == "vctk"
        assert args.teacher_ckpt == Path("teacher.pt")
        assert args.phase == "A"
        assert args.lr is None
        assert args.config is None

    def test_build_parser_all_args(self):
        from tmrvc_train.cli.distill import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "/data",
            "--dataset", "vctk",
            "--teacher-ckpt", "t.pt",
            "--phase", "B",
            "--lr", "5e-5",
            "--max-steps", "100000",
            "--batch-size", "32",
            "--config", "configs/train_student.yaml",
        ])
        assert args.dataset == "vctk"
        assert args.phase == "B"
        assert args.lr == 5e-5
        assert args.max_steps == 100000
        assert args.config == Path("configs/train_student.yaml")

    def test_load_config_none(self):
        from tmrvc_train.cli.distill import _load_config

        result = _load_config(None, "A")
        assert result == {}

    def test_load_config_yaml(self, tmp_path):
        from tmrvc_train.cli.distill import _load_config

        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(
            "batch_size: 32\n"
            "phases:\n"
            "  'A':\n"
            "    lr: 0.0001\n"
            "    teacher_steps: 20\n"
            "    description: test\n"
        )
        result = _load_config(cfg_file, "A")
        assert result["lr"] == 0.0001
        assert result["teacher_steps"] == 20
        assert result["batch_size"] == 32
        assert "description" not in result

    def test_default_lr(self):
        from tmrvc_train.cli.distill import _default_lr

        assert _default_lr("A") == 1e-4
        assert _default_lr("B") == 5e-5
        assert _default_lr("B2") == 5e-5
        assert _default_lr("C") == 2e-5

    def test_build_parser_phase_b2(self):
        from tmrvc_train.cli.distill import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "--cache-dir", "/tmp/cache",
            "--dataset", "vctk",
            "--teacher-ckpt", "teacher.pt",
            "--phase", "B2",
        ])
        assert args.phase == "B2"


class TestTrainerOTCFM:
    """Tests for OT-CFM integration in TeacherTrainer."""

    def test_train_step_with_ot_cfm(self, tmp_path):
        teacher = TeacherUNet()
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)
        config = TrainerConfig(
            max_steps=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
            use_ot_cfm=True,
        )
        trainer = TeacherTrainer(
            teacher, scheduler, optimizer, [_make_batch()], config,
        )
        losses = trainer.train_step(_make_batch())
        assert "flow" in losses
        assert "total" in losses


class TestTrainerCFGFree:
    """Tests for CFG-free conditioning dropout."""

    def test_train_step_with_p_uncond(self, tmp_path):
        """p_uncond > 0 should allow training to proceed."""
        teacher = TeacherUNet()
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)
        config = TrainerConfig(
            max_steps=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
            p_uncond=0.5,
        )
        trainer = TeacherTrainer(
            teacher, scheduler, optimizer, [_make_batch()], config,
        )
        losses = trainer.train_step(_make_batch())
        assert "flow" in losses
        assert "total" in losses

    def test_train_step_p_uncond_zero_unchanged(self, tmp_path):
        """p_uncond=0 should produce same behavior as default."""
        teacher = TeacherUNet()
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)
        config = TrainerConfig(
            max_steps=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
            p_uncond=0.0,
        )
        trainer = TeacherTrainer(
            teacher, scheduler, optimizer, [_make_batch()], config,
        )
        losses = trainer.train_step(_make_batch())
        assert "flow" in losses
        assert "total" in losses


class TestReflowTrainer:
    """Tests for ReflowTrainer."""

    def test_reflow_trainer_train_step(self, tmp_path):
        """Reflow trainer should complete a single training step with pre-generated noise."""
        teacher = TeacherUNet()
        scheduler = FlowMatchingScheduler()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=2e-4)

        batch = _make_batch()
        # Pre-generated noise endpoint (same shape as mel_target)
        x_1_noise = torch.randn(2, N_MELS, 64)

        config = TrainerConfig(
            phase="reflow",
            max_steps=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        trainer = ReflowTrainer(
            teacher, scheduler, optimizer, [batch], config,
        )
        losses = trainer.train_step(batch, x_1_noise=x_1_noise)
        assert "flow" in losses
        assert "total" in losses
        assert losses["total"] > 0


class TestDistillationPhaseB2:
    """Tests for DMD2 distillation (Phase B2)."""

    def test_distillation_phase_b2_step(self, tmp_path):
        """Phase B2 with discriminator should complete a single training step."""
        from tmrvc_train.distillation import DistillationConfig, DistillationTrainer
        from tmrvc_train.models.content_encoder import ContentEncoderStudent
        from tmrvc_train.models.converter import ConverterStudent
        from tmrvc_train.models.discriminator import MelDiscriminator
        from tmrvc_train.models.ir_estimator import IREstimator
        from tmrvc_train.models.vocoder import VocoderStudent

        teacher = TeacherUNet()
        content_encoder = ContentEncoderStudent()
        converter = ConverterStudent()
        vocoder = VocoderStudent()
        ir_estimator = IREstimator()
        discriminator = MelDiscriminator()
        scheduler = FlowMatchingScheduler()

        student_params = (
            list(content_encoder.parameters())
            + list(converter.parameters())
            + list(vocoder.parameters())
            + list(ir_estimator.parameters())
        )
        optimizer = torch.optim.AdamW(student_params, lr=5e-5)
        disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-4)

        config = DistillationConfig(
            phase="B2",
            max_steps=2,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        trainer = DistillationTrainer(
            teacher, content_encoder, converter, vocoder, ir_estimator,
            scheduler, optimizer, [_make_batch()], config,
            discriminator=discriminator,
            disc_optimizer=disc_optimizer,
        )
        losses = trainer.train_step(_make_batch())
        assert "gen" in losses
        assert "disc" in losses
        assert "total" in losses
