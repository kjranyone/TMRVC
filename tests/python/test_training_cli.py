"""Tests for training CLI modules."""

import pytest
import torch
from pathlib import Path

from tmrvc_train.models.streaming_codec import (
    CodecConfig,
    StreamingCodec,
)
from tmrvc_train.models.token_model import (
    TokenModelConfig,
    TokenModel,
)


class TestCodecTrainingSetup:
    def test_codec_config_default(self):
        config = CodecConfig()
        assert config.frame_size == 480
        assert config.n_codebooks == 4
        assert config.codebook_size == 1024

    def test_codec_forward_pass(self):
        config = CodecConfig()
        model = StreamingCodec(config)
        model.eval()

        audio = torch.randn(2, 1, 480 * 50)

        with torch.no_grad():
            audio_out, indices, commit_loss, enc_state, dec_state = model(
                audio, None, None
            )

        assert audio_out.shape == audio.shape
        assert indices.shape[0] == 2
        assert indices.shape[1] == config.n_codebooks

    def test_codec_loss_computation(self):
        config = CodecConfig()
        model = StreamingCodec(config)

        audio = torch.randn(2, 1, 480 * 50)

        audio_out, indices, commit_loss, _, _ = model(audio, None, None)

        rec_loss = torch.nn.functional.l1_loss(audio_out, audio)

        assert rec_loss.item() > 0
        assert commit_loss.item() >= 0


class TestTokenModelTrainingSetup:
    def test_token_config_default(self):
        config = TokenModelConfig()
        assert config.n_codebooks == 4
        assert config.codebook_size == 1024
        assert config.context_length == 10

    def test_token_model_forward_pass(self):
        config = TokenModelConfig()
        model = TokenModel(config)
        model.eval()

        B, K, L = 2, config.n_codebooks, config.context_length
        tokens = torch.randint(0, config.codebook_size, (B, K, L))
        spk_embed = torch.randn(B, config.d_spk)

        with torch.no_grad():
            logits, kv_caches = model(tokens, spk_embed, None)

        assert logits.shape == (B, K, config.codebook_size)
        assert len(kv_caches) == config.n_layers

    def test_token_loss_computation(self):
        config = TokenModelConfig()
        model = TokenModel(config)

        B, K, L = 2, config.n_codebooks, config.context_length
        tokens = torch.randint(0, config.codebook_size, (B, K, L))
        target = tokens[:, :, -1]
        spk_embed = torch.randn(B, config.d_spk)

        logits, _ = model(tokens, spk_embed, None)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.codebook_size), target.view(-1)
        )

        assert loss.item() > 0

    def test_training_step_simulation(self):
        config = TokenModelConfig()
        model = TokenModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        B, K, L = 2, config.n_codebooks, config.context_length
        tokens = torch.randint(0, config.codebook_size, (B, K, L))
        target = tokens[:, :, -1]
        spk_embed = torch.randn(B, config.d_spk)

        optimizer.zero_grad()
        logits, _ = model(tokens, spk_embed, None)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, config.codebook_size), target.view(-1)
        )
        loss.backward()
        optimizer.step()

        assert loss.item() > 0


class TestCodecDatasetMock:
    def test_segment_extraction_logic(self):
        segment_frames = 50
        frame_size = 480
        segment_samples = segment_frames * frame_size

        audio = torch.randn(1, segment_samples * 2)

        start = torch.randint(0, audio.shape[1] - segment_samples, (1,))
        segment = audio[:, start : start + segment_samples]

        assert segment.shape == (1, segment_samples)
