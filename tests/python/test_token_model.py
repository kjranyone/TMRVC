"""Tests for token_model module."""

import pytest
import torch

from tmrvc_train.models.token_model import (
    TokenModelConfig,
    TokenModel,
    CausalSelfAttention,
    TransformerBlock,
    FiLMConditioner,
)


class TestTokenModelConfig:
    def test_default_config(self):
        config = TokenModelConfig()
        assert config.n_codebooks == 4
        assert config.codebook_size == 1024
        assert config.d_model == 256
        assert config.n_heads == 4
        assert config.n_layers == 6
        assert config.context_length == 10
        assert config.d_spk == 192


class TestCausalSelfAttention:
    def test_forward_shape(self):
        config = TokenModelConfig()
        attn = CausalSelfAttention(config)
        x = torch.randn(1, 10, config.d_model)
        out, kv_cache = attn(x, None)
        assert out.shape == x.shape
        assert kv_cache[0].shape == (
            1,
            config.n_heads,
            10,
            config.d_model // config.n_heads,
        )

    def test_kv_cache_accumulation(self):
        config = TokenModelConfig()
        attn = CausalSelfAttention(config)

        x1 = torch.randn(1, 5, config.d_model)
        x2 = torch.randn(1, 5, config.d_model)

        out1, kv1 = attn(x1, None)
        out2, kv2 = attn(x2, kv1)

        assert kv2[0].shape[2] == 10


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

        tokens = torch.randint(0, 1024, (1, config.n_codebooks, config.context_length))
        spk_embed = torch.randn(1, 192)

        logits, kv_caches = model(tokens, spk_embed, None)

        assert logits.shape == (1, 4, 1024)
        assert len(kv_caches) == config.n_layers

    def test_forward_with_kv_cache(self):
        config = TokenModelConfig()
        model = TokenModel(config)

        tokens = torch.randint(0, 1024, (1, config.n_codebooks, config.context_length))
        spk_embed = torch.randn(1, 192)

        logits1, kv1 = model(tokens, spk_embed, None)
        logits2, kv2 = model(tokens, spk_embed, kv1)

        assert logits1.shape == logits2.shape

    def test_generate_next_tokens(self):
        config = TokenModelConfig()
        model = TokenModel(config)
        model.eval()

        tokens = torch.randint(0, 1024, (1, config.n_codebooks, config.context_length))
        spk_embed = torch.randn(1, 192)

        with torch.no_grad():
            next_tokens, kv = model.generate_next_tokens(
                tokens, spk_embed, temperature=1.0, top_k=50
            )

        assert next_tokens.shape == (1, 4)
        assert (next_tokens >= 0).all() and (next_tokens < 1024).all()

    def test_init_kv_cache(self):
        config = TokenModelConfig()
        model = TokenModel(config)

        kv_cache = model.init_kv_cache(2, torch.device("cpu"))
        assert len(kv_cache) == config.n_layers
        assert all(kv is None for kv in kv_cache)
