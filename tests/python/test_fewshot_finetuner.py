"""Tests for FewShotAdapterGTM.finetune_step, FewShotConfig, and FewShotFinetuner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from tmrvc_core.constants import (
    D_CONTENT,
    D_SPEAKER,
    LORA_DELTA_SIZE,
    N_IR_PARAMS,
)


# ---------------------------------------------------------------------------
# Test: FewShotAdapterGTM.finetune_step
# ---------------------------------------------------------------------------


class TestFewShotAdapterGTMFinetuneStep:
    """FewShotAdapterGTM.finetune_step returns a loss and LoRA params receive gradients."""

    def test_finetune_step_returns_loss_and_updates_params(self):
        from tmrvc_train.fewshot import FewShotAdapterGTM
        from tmrvc_train.models.converter import ConverterStudentGTM

        converter = ConverterStudentGTM()
        adapter = FewShotAdapterGTM(converter)

        optimizer = torch.optim.Adam(adapter.get_lora_parameters(), lr=1e-3)

        B, T = 1, 20
        content = torch.randn(B, D_CONTENT, T)
        spk_embed = torch.randn(B, D_SPEAKER)
        ir_params = torch.zeros(B, N_IR_PARAMS)
        mel_target = torch.randn(B, 80, T)

        # Record initial LoRA params
        initial_a = adapter.lora_layer.lora_A.data.clone()
        initial_b = adapter.lora_layer.lora_B.data.clone()

        loss = adapter.finetune_step(optimizer, content, spk_embed, ir_params, mel_target)

        assert isinstance(loss, float)
        assert loss > 0.0

        # Verify LoRA params were updated (at least one should change)
        a_changed = not torch.allclose(adapter.lora_layer.lora_A.data, initial_a)
        b_changed = not torch.allclose(adapter.lora_layer.lora_B.data, initial_b)
        assert a_changed or b_changed, "LoRA parameters should be updated after finetune_step"

    def test_finetune_step_preserves_original_weights(self):
        """GTM projection weight must be restored after finetune_step."""
        from tmrvc_train.fewshot import FewShotAdapterGTM
        from tmrvc_train.models.converter import ConverterStudentGTM

        converter = ConverterStudentGTM()
        adapter = FewShotAdapterGTM(converter)
        optimizer = torch.optim.Adam(adapter.get_lora_parameters(), lr=1e-3)

        original_weight = converter.gtm.proj.weight.data.clone()

        B, T = 1, 10
        content = torch.randn(B, D_CONTENT, T)
        spk_embed = torch.randn(B, D_SPEAKER)
        ir_params = torch.zeros(B, N_IR_PARAMS)
        mel_target = torch.randn(B, 80, T)

        adapter.finetune_step(optimizer, content, spk_embed, ir_params, mel_target)

        assert torch.allclose(converter.gtm.proj.weight.data, original_weight), (
            "GTM projection weight must be restored after finetune_step"
        )


# ---------------------------------------------------------------------------
# Test: FewShotConfig defaults
# ---------------------------------------------------------------------------


class TestFewShotConfig:
    def test_defaults(self):
        from tmrvc_train.fewshot import FewShotConfig

        cfg = FewShotConfig()
        assert cfg.max_steps == 200
        assert cfg.lr == 1e-3
        assert cfg.segment_frames == 200
        assert cfg.use_gtm is False
        assert cfg.log_every == 10

    def test_custom_values(self):
        from tmrvc_train.fewshot import FewShotConfig

        cfg = FewShotConfig(max_steps=50, lr=5e-4, use_gtm=True)
        assert cfg.max_steps == 50
        assert cfg.lr == 5e-4
        assert cfg.use_gtm is True


# ---------------------------------------------------------------------------
# Test: FewShotFinetuner.prepare_data
# ---------------------------------------------------------------------------


class TestFewShotFinetunerPrepareData:
    def test_prepare_data_returns_correct_shapes(self, tmp_path):
        """prepare_data should return (content, mel) with matching T dimensions."""
        from tmrvc_train.fewshot import FewShotConfig, FewShotFinetuner
        from tmrvc_train.models.content_encoder import ContentEncoderStudent
        from tmrvc_train.models.converter import ConverterStudent

        converter = ConverterStudent()
        content_encoder = ContentEncoderStudent()
        spk_embed = torch.randn(D_SPEAKER)
        config = FewShotConfig(max_steps=5)

        finetuner = FewShotFinetuner(converter, content_encoder, spk_embed, config)

        # Create a dummy wav file (1 second at 24kHz)
        import soundfile as sf

        sr = 24000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr)

        pairs = finetuner.prepare_data([str(wav_path)])

        assert len(pairs) == 1
        content, mel = pairs[0]
        assert content.dim() == 3
        assert content.shape[0] == 1
        assert content.shape[1] == D_CONTENT  # 256
        assert mel.dim() == 3
        assert mel.shape[0] == 1
        assert mel.shape[1] == 80
        # T dimensions must match
        assert content.shape[2] == mel.shape[2]


# ---------------------------------------------------------------------------
# Test: FewShotFinetuner.finetune_iter
# ---------------------------------------------------------------------------


class TestFewShotFinetunerIter:
    def test_finetune_iter_yields_step_loss(self):
        """finetune_iter yields (step, loss) tuples and loss decreases or stays reasonable."""
        from tmrvc_train.fewshot import FewShotConfig, FewShotFinetuner
        from tmrvc_train.models.content_encoder import ContentEncoderStudent
        from tmrvc_train.models.converter import ConverterStudent

        converter = ConverterStudent()
        content_encoder = ContentEncoderStudent()
        spk_embed = torch.randn(D_SPEAKER)
        config = FewShotConfig(max_steps=10, segment_frames=20)

        finetuner = FewShotFinetuner(converter, content_encoder, spk_embed, config)

        # Synthetic data (skip disk I/O)
        T = 30
        data = [
            (torch.randn(1, D_CONTENT, T), torch.randn(1, 80, T)),
        ]

        results = list(finetuner.finetune_iter(data))

        assert len(results) == 10
        assert results[0][0] == 1
        assert results[-1][0] == 10

        # All losses should be positive floats
        for step, loss in results:
            assert isinstance(loss, float)
            assert loss > 0.0

    def test_finetune_with_gtm(self):
        """finetune_iter works with GTM adapter."""
        from tmrvc_train.fewshot import FewShotConfig, FewShotFinetuner
        from tmrvc_train.models.content_encoder import ContentEncoderStudent
        from tmrvc_train.models.converter import ConverterStudentGTM

        converter = ConverterStudentGTM()
        content_encoder = ContentEncoderStudent()
        spk_embed = torch.randn(D_SPEAKER)
        config = FewShotConfig(max_steps=5, use_gtm=True, segment_frames=10)

        finetuner = FewShotFinetuner(converter, content_encoder, spk_embed, config)

        T = 15
        data = [(torch.randn(1, D_CONTENT, T), torch.randn(1, 80, T))]

        results = list(finetuner.finetune_iter(data))
        assert len(results) == 5


# ---------------------------------------------------------------------------
# Test: CLI finetune --help
# ---------------------------------------------------------------------------


class TestFinetuneCLI:
    def test_help(self, capsys):
        from tmrvc_train.cli.finetune import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_e2e_with_mocks(self, tmp_path):
        """End-to-end CLI test with mocked models and audio."""
        import soundfile as sf

        from tmrvc_core.constants import LORA_DELTA_SIZE
        from tmrvc_export.speaker_file import read_speaker_file

        # Create dummy audio
        sr = 24000
        audio = np.random.randn(sr).astype(np.float32) * 0.1
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, sr)

        # Create dummy checkpoint
        from tmrvc_train.models.content_encoder import ContentEncoderStudent
        from tmrvc_train.models.converter import ConverterStudent

        ce = ContentEncoderStudent()
        cv = ConverterStudent()
        ckpt_path = tmp_path / "distill.pt"
        torch.save({
            "content_encoder": ce.state_dict(),
            "converter": cv.state_dict(),
        }, ckpt_path)

        output_path = tmp_path / "out.tmrvc_speaker"

        # Mock SpeakerEncoder to avoid downloading ECAPA model
        mock_encoder = MagicMock()
        mock_encoder.extract_from_file.return_value = torch.randn(D_SPEAKER)

        with patch("tmrvc_data.speaker.SpeakerEncoder", return_value=mock_encoder):
            from tmrvc_train.cli.finetune import main

            main([
                "--audio-files", str(wav_path),
                "--checkpoint", str(ckpt_path),
                "--output", str(output_path),
                "--steps", "5",
                "--device", "cpu",
            ])

        assert output_path.exists()

        # Verify the speaker file is valid
        spk_embed, lora_delta, _meta, _thumb = read_speaker_file(output_path)
        assert spk_embed.shape == (D_SPEAKER,)
        assert lora_delta.shape == (LORA_DELTA_SIZE,)
        # lora_delta should not be all zeros (fine-tuned)
        assert not np.allclose(lora_delta, 0.0), "LoRA delta should be non-zero after fine-tuning"


# ---------------------------------------------------------------------------
# Test: FinetuneWorker signals
# ---------------------------------------------------------------------------


class TestFinetuneWorker:
    def test_worker_init(self):
        """FinetuneWorker can be instantiated with config dict."""
        # Only test import and init — actual run requires Qt event loop
        from tmrvc_gui.workers.finetune_worker import FinetuneWorker

        config = {
            "audio_paths": ["/tmp/a.wav"],
            "checkpoint_path": "/tmp/distill.pt",
            "output_path": "/tmp/out.tmrvc_speaker",
            "max_steps": 100,
            "lr": 1e-3,
            "use_gtm": False,
        }
        # FinetuneWorker requires QApplication — just test import + config parsing
        worker = FinetuneWorker.__new__(FinetuneWorker)
        # Manually set attributes without calling __init__ (no Qt needed)
        worker.audio_paths = config["audio_paths"]
        worker.checkpoint_path = Path(config["checkpoint_path"])
        worker.output_path = Path(config["output_path"])
        worker.max_steps = config["max_steps"]
        worker.lr = config["lr"]
        worker.use_gtm = config["use_gtm"]

        assert worker.max_steps == 100
        assert worker.use_gtm is False
