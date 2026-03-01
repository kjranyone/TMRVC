"""LoRA (Low-Rank Adaptation) for UCLM v2 fine-tuning.

Implements LoRA for speaker adaptation in the UCLM architecture.
Applied to Transformer attention projections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import LORA_DELTA_SIZE


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj")


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 8.0, dropout: float = 0.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base_linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._init_weights()
        for param in self.base.parameters(): param.requires_grad = False

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def get_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


class UCLMLoRAWrapper:
    """Wrapper for fine-tuning DisentangledUCLM with LoRA."""

    def __init__(self, model: nn.Module, config: Optional[LoRAConfig] = None):
        self.model = model
        self.config = config or LoRAConfig()
        self.lora_modules: dict[str, LoRALinear] = {}

    def setup(self):
        """Apply LoRA to UCLM core layers."""
        for param in self.model.parameters(): param.requires_grad = False
        self._apply_recursive(self.model.uclm_core, "uclm_core")

    def _apply_recursive(self, module: nn.Module, prefix: str):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}"
            if isinstance(child, nn.Linear) and name in self.config.target_modules:
                lora = LoRALinear(child, self.config.rank, self.config.alpha, self.config.dropout)
                setattr(module, name, lora)
                self.lora_modules[full_name] = lora
            else:
                self._apply_recursive(child, full_name)

    def parameters(self):
        params = []
        for lora in self.lora_modules.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params

    def extract_delta_flat(self) -> torch.Tensor:
        flat_parts = [self.lora_modules[n].get_delta().flatten() for n in sorted(self.lora_modules.keys())]
        flat = torch.cat(flat_parts)
        # Pad or trim to LORA_DELTA_SIZE
        if len(flat) > LORA_DELTA_SIZE: flat = flat[:LORA_DELTA_SIZE]
        else: flat = F.pad(flat, (0, LORA_DELTA_SIZE - len(flat)))
        return flat


def finetune_uclm_lora(
    model: nn.Module,
    ref_audio_tokens: torch.Tensor, # [B, 8, T]
    ref_control_tokens: torch.Tensor, # [B, 4, T]
    speaker_embed: torch.Tensor,
    voice_state: torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-4,
    device: str = "cpu",
) -> torch.Tensor:
    """Few-shot fine-tune UCLM core using LoRA."""
    model = model.to(device)
    wrapper = UCLMLoRAWrapper(model)
    wrapper.setup()
    
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=lr)
    
    # Simple Reconstruction Loss on tokens
    for step in range(n_steps):
        optimizer.zero_grad()
        # Mocking content features from source tokens for VC-style reconstruction
        content, _ = model.vc_encoder(ref_audio_tokens.to(device))
        
        state_cond = model.voice_state_enc(voice_state.to(device), torch.zeros_like(content)) # SSL zero
        
        out = model.forward_streaming(
            content, state_cond, speaker_embed.to(device)
        )
        
        # Loss against ground truth tokens
        from .uclm_loss import uclm_loss
        losses = uclm_loss(
            out["logits_a"], out["logits_b"], 
            ref_audio_tokens.to(device), ref_control_tokens.to(device)
        )
        
        losses["loss"].backward()
        optimizer.step()
        
    return wrapper.extract_delta_flat()
