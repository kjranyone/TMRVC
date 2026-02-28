"""Tests for LoRA fine-tuning module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tmrvc_train.lora import (
    LoRAConfig,
    LoRALinear,
    TokenModelLoRAWrapper,
    extract_lora_deltas,
    flatten_lora_deltas,
    finetune_token_model_lora,
)
from tmrvc_train.models.token_model import TokenModel, TokenModelConfig


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestLoRALinear:
    def test_init(self):
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=4, alpha=1.0)

        assert lora.rank == 4
        assert lora.scaling == 0.25
        assert lora.lora_A.shape == (4, 64)
        assert lora.lora_B.shape == (128, 4)
        assert not lora.base.weight.requires_grad

    def test_forward_shape(self):
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=4)

        x = torch.randn(2, 64)
        out = lora(x)

        assert out.shape == (2, 128)

    def test_lora_contribution(self):
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=4)

        x = torch.randn(2, 64)

        base_out = base(x)
        lora_out = lora(x)

        assert not torch.allclose(base_out, lora_out)

    def test_get_delta(self):
        base = nn.Linear(64, 128)
        lora = LoRALinear(base, rank=4, alpha=1.0)

        delta = lora.get_delta()

        assert delta.shape == (128, 64)


class TestTokenModelLoRAWrapper:
    def test_setup_applies_lora(self):
        model = TokenModel(TokenModelConfig())
        wrapper = TokenModelLoRAWrapper(model)

        wrapper.setup()

        assert len(wrapper.lora_modules) > 0

        trainable_params = wrapper.get_trainable_parameters()
        assert len(trainable_params) > 0

        for p in trainable_params:
            assert p.requires_grad

    def test_extract_delta_flat(self):
        model = TokenModel(TokenModelConfig())
        wrapper = TokenModelLoRAWrapper(model)
        wrapper.setup()

        delta = wrapper.extract_delta_flat(target_size=15872)

        assert delta.shape == (15872,)


class TestFinetuneTokenModelLoRA:
    def test_finetune_basic(self):
        config = TokenModelConfig(n_layers=2, n_heads=2, d_model=64)
        model = TokenModel(config)

        ref_tokens = torch.randint(0, 1024, (20, 4))
        spk_embed = torch.randn(192)

        delta = finetune_token_model_lora(
            model=model,
            reference_tokens=ref_tokens,
            spk_embed=spk_embed,
            n_steps=5,
            lr=1e-3,
            device="cpu",
        )

        assert delta.shape[0] == 15872
        assert not torch.allclose(delta, torch.zeros_like(delta))
