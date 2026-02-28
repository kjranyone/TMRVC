"""LoRA (Low-Rank Adaptation) for Token Model fine-tuning.

Implements LoRA for speaker adaptation in the Codec-Latent paradigm.
Applied to attention projections and FFN layers.

Reference: LoRA: Low-Rank Adaptation of Large Language Models (arXiv:2106.09685)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
    )


class LoRALinear(nn.Module):
    """LoRA-adapted Linear layer.

    Wraps a base linear layer with low-rank adaptation:
        output = W @ x + (alpha / r) * B @ A @ x

    Where:
        W: [out_features, in_features] - frozen base weight
        A: [rank, in_features] - down projection
        B: [out_features, rank] - up projection
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

        for param in self.base.parameters():
            param.requires_grad = False

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def merge_weights(self) -> None:
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.base.weight.data += delta
            self.lora_A.zero_()
            self.lora_B.zero_()

    def get_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


def apply_lora_to_model(
    model: nn.Module,
    config: LoRAConfig,
) -> dict[str, LoRALinear]:
    """Apply LoRA to target modules in the model.

    Args:
        model: TokenModel or similar
        config: LoRA configuration

    Returns:
        Dict mapping parameter names to LoRALinear instances
    """
    lora_modules = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module_name = name.split(".")[-1]
            if module_name in config.target_modules:
                lora = LoRALinear(
                    module,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                )
                lora_modules[name] = lora

    return lora_modules


def get_lora_parameters(lora_modules: dict[str, LoRALinear]) -> list[nn.Parameter]:
    """Get all trainable LoRA parameters."""
    params = []
    for lora in lora_modules.values():
        params.extend([lora.lora_A, lora.lora_B])
    return params


def extract_lora_deltas(lora_modules: dict[str, LoRALinear]) -> dict[str, torch.Tensor]:
    """Extract weight deltas from all LoRA modules.

    Returns:
        Dict mapping module names to weight deltas [out_features, in_features]
    """
    deltas = {}
    for name, lora in lora_modules.items():
        deltas[name] = lora.get_delta().detach().clone()
    return deltas


def flatten_lora_deltas(
    deltas: dict[str, torch.Tensor], target_size: int
) -> torch.Tensor:
    """Flatten all deltas into a single vector.

    Args:
        deltas: Dict of weight deltas
        target_size: Target size for the flat vector (LORA_DELTA_SIZE)

    Returns:
        Flattened delta vector [target_size]
    """
    flat_parts = []
    for name in sorted(deltas.keys()):
        flat_parts.append(deltas[name].flatten())

    flat = torch.cat(flat_parts)

    if len(flat) > target_size:
        flat = flat[:target_size]
    elif len(flat) < target_size:
        flat = F.pad(flat, (0, target_size - len(flat)))

    return flat


class TokenModelLoRAWrapper:
    """Wrapper for fine-tuning TokenModel with LoRA.

    Handles the full fine-tuning pipeline:
    1. Freeze base model, apply LoRA
    2. Train on speaker's reference audio
    3. Extract and flatten LoRA deltas
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[LoRAConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or LoRAConfig()
        self.lora_modules: dict[str, LoRALinear] = {}

    def setup(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        self._apply_lora_recursive(self.model, "")

    def _apply_lora_recursive(self, module: nn.Module, prefix: str) -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            if isinstance(child, nn.Linear):
                module_name = name
                if module_name in self.config.target_modules:
                    lora = LoRALinear(
                        child,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                    )
                    setattr(module, name, lora)
                    self.lora_modules[full_name] = lora
            else:
                self._apply_lora_recursive(child, full_name)

    def get_trainable_parameters(self) -> list[nn.Parameter]:
        params = []
        for lora in self.lora_modules.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params

    def parameters(self) -> list[nn.Parameter]:
        return self.get_trainable_parameters()

    def extract_delta_flat(self, target_size: int = 15872) -> torch.Tensor:
        deltas = extract_lora_deltas(self.lora_modules)
        return flatten_lora_deltas(deltas, target_size)


def finetune_token_model_lora(
    model: nn.Module,
    reference_tokens: torch.Tensor,
    spk_embed: torch.Tensor,
    n_steps: int = 200,
    lr: float = 1e-4,
    device: str = "cpu",
) -> torch.Tensor:
    """Fine-tune TokenModel with LoRA on reference tokens.

    Args:
        model: TokenModel instance
        reference_tokens: Reference codec tokens [T, 4] or [1, 4, T]
        spk_embed: Speaker embedding [192]
        n_steps: Number of fine-tuning steps
        lr: Learning rate
        device: Device

    Returns:
        Flattened LoRA delta vector [15872]
    """
    model = model.to(device)
    model.eval()

    wrapper = TokenModelLoRAWrapper(model)
    wrapper.setup()

    optimizer = torch.optim.Adam(wrapper.parameters(), lr=lr)

    if reference_tokens.dim() == 2:
        reference_tokens = reference_tokens.unsqueeze(0).transpose(1, 2)

    reference_tokens = reference_tokens.to(device)
    spk_embed = spk_embed.to(device).unsqueeze(0)

    B, K, T = reference_tokens.shape

    for step in range(n_steps):
        total_loss = 0.0
        n_batches = 0

        for t in range(T - 1):
            input_tokens = reference_tokens[:, :, t : t + 1]
            target_token = reference_tokens[:, :, t + 1]

            optimizer.zero_grad()

            logits, _ = model(input_tokens, spk_embed)

            loss = F.cross_entropy(
                logits.squeeze(2),
                target_token,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if step % 50 == 0:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"Step {step}/{n_steps}, loss: {avg_loss:.4f}")

    delta_flat = wrapper.extract_delta_flat()
    return delta_flat
