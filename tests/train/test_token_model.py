"""Tests for token_model module."""

import pytest
import torch

from tmrvc_train.models.token_model import (
    TokenModelConfig,
    TokenModel,
    MambaBlock,
    CausalMambaBlock,
    FiLMConditioner,
)


class TestTokenModelConfig:
    def test_default_config(self):
        config = TokenModelConfig()
        assert config.n_codebooks == 4
        assert config.codebook_size == 1024
        assert config.d_model == 256
        assert config.d_state == 16
        assert config.n_layers == 6
        assert config.context_length == 10
        assert config.d_spk == 192

    def test_d_inner(self):
        config = TokenModelConfig()
        assert config.d_inner == 512


class TestMambaBlock:
    def test_forward_shape(self):
        config = TokenModelConfig()
        block = MambaBlock(config)
        x = torch.randn(1, 10, config.d_model)
        out, state = block(x, None)
        assert out.shape == x.shape

    def test_state_shape(self):
        config = TokenModelConfig()
        block = MambaBlock(config)
        x = torch.randn(1, 10, config.d_model)
        _, state = block(x, None)
        assert state.shape == (1, config.d_inner, config.d_state)


class TestFiLMConditioner:
    def test_forward_shape(self):
        film = FiLMConditioner(d_cond=192, d_model=256)
        x = torch.randn(1, 10, 256)
        cond = torch.randn(1, 192)
        out = film(x, cond)
        assert out.shape == x.shape


class TestTokenModel:
    def test_forward_shape(self):
        config = TokenModelConfig()
        model = TokenModel(config)

        tokens = torch.randint(0, 1024, (1, 10, 4))
        spk_embed = torch.randn(1, 192)

        logits, states = model(tokens, spk_embed, None)

        assert logits.shape == (1, 4, 1024)
        assert len(states) == config.n_layers

    def test_forward_with_state(self):
        config = TokenModelConfig()
        model = TokenModel(config)

        tokens = torch.randint(0, 1024, (1, 10, 4))
        spk_embed = torch.randn(1, 192)

        logits1, states1 = model(tokens, spk_embed, None)
        logits2, states2 = model(tokens, spk_embed, states1)

        assert logits1.shape == logits2.shape

    def test_generate_next_tokens(self):
        config = TokenModelConfig()
        model = TokenModel(config)
        model.eval()

        tokens = torch.randint(0, 1024, (1, 10, 4))
        spk_embed = torch.randn(1, 192)

        with torch.no_grad():
            next_tokens, states = model.generate_next_tokens(
                tokens, spk_embed, temperature=1.0, top_k=50
            )

        assert next_tokens.shape == (1, 4)
        assert (next_tokens >= 0).all() and (next_tokens < 1024).all()
