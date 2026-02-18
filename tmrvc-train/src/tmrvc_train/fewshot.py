"""FewShotAdapter: LoRA-based few-shot speaker adaptation."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import (
    LORA_ALPHA,
    LORA_DELTA_SIZE,
    LORA_RANK,
    N_IR_PARAMS,
    N_LORA_LAYERS,
)
from tmrvc_train.models.converter import ConverterStudent, ConverterStudentGTM

logger = logging.getLogger(__name__)


class LoRALayer(nn.Module):
    """Single LoRA adapter for a FiLM projection layer."""

    def __init__(self, d_in: int, d_out: int, rank: int = LORA_RANK) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, d_out))
        self.rank = rank

    def forward(self) -> torch.Tensor:
        """Compute LoRA delta weight: B^T @ A^T → [d_out, d_in]."""
        return self.lora_B.T @ self.lora_A.T  # [d_out, d_in]

    @property
    def delta_flat(self) -> torch.Tensor:
        """Flattened delta for storage."""
        return torch.cat([self.lora_A.flatten(), self.lora_B.flatten()])


class FewShotAdapter:
    """LoRA-based few-shot adaptation for the converter.

    Steps:
    1. Target speaker audio → SpeakerEncoder → spk_embed + lora_delta
    2. Apply LoRA delta to converter FiLM layers
    3. Fine-tune LoRA weights on few samples (10-50 steps)
    4. Merge: W_merged = W_original + (alpha/rank) * B^T @ A^T
    """

    def __init__(
        self,
        converter: ConverterStudent,
        n_lora_layers: int = N_LORA_LAYERS,
        rank: int = LORA_RANK,
        alpha: int = LORA_ALPHA,
    ) -> None:
        self.converter = converter
        self.n_lora_layers = n_lora_layers
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Create LoRA layers for the first n_lora_layers FiLM projections
        d_cond = converter.blocks[0].film.proj.in_features
        d_model_x2 = converter.blocks[0].film.proj.out_features
        self.lora_layers = nn.ModuleList([
            LoRALayer(d_cond, d_model_x2, rank=rank)
            for _ in range(min(n_lora_layers, len(converter.blocks)))
        ])

    def apply_lora_delta(self, lora_delta: torch.Tensor) -> None:
        """Load LoRA delta from a flat tensor into LoRA layer parameters.

        Args:
            lora_delta: ``[lora_delta_size]`` flat tensor.
        """
        offset = 0
        for lora in self.lora_layers:
            a_size = lora.lora_A.numel()
            b_size = lora.lora_B.numel()
            lora.lora_A.data = lora_delta[offset:offset + a_size].reshape_as(lora.lora_A)
            offset += a_size
            lora.lora_B.data = lora_delta[offset:offset + b_size].reshape_as(lora.lora_B)
            offset += b_size

    def merge_weights(self) -> None:
        """Merge LoRA weights into converter FiLM projections.

        After merging, the converter runs as vanilla inference without LoRA overhead.
        """
        for i, lora in enumerate(self.lora_layers):
            if i >= len(self.converter.blocks):
                break
            film_proj = self.converter.blocks[i].film.proj
            delta_w = lora()  # [d_out, d_in]
            film_proj.weight.data += self.scale * delta_w
        logger.info("Merged LoRA weights into %d FiLM layers", len(self.lora_layers))

    def get_lora_parameters(self) -> list[nn.Parameter]:
        """Get LoRA parameters for optimizer."""
        return list(self.lora_layers.parameters())

    def get_lora_delta_flat(self) -> torch.Tensor:
        """Get all LoRA deltas as a flat tensor."""
        return torch.cat([lora.delta_flat for lora in self.lora_layers])

    def finetune_step(
        self,
        optimizer: torch.optim.Optimizer,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        ir_params: torch.Tensor,
        mel_target: torch.Tensor,
    ) -> float:
        """Single fine-tuning step.

        Uses forward hooks to inject LoRA deltas so gradients flow
        back to LoRA parameters without modifying original weights.

        Returns:
            Loss value.
        """
        self.converter.train()

        # Register hooks to add LoRA delta to FiLM projection outputs
        handles = []
        for i, lora in enumerate(self.lora_layers):
            if i >= len(self.converter.blocks):
                break
            film_proj = self.converter.blocks[i].film.proj
            scale = self.scale

            def make_hook(lora_layer, s):
                def hook(module, inp, output):
                    delta_w = s * lora_layer()  # [d_out, d_in]
                    return output + F.linear(inp[0], delta_w)
                return hook

            h = film_proj.register_forward_hook(make_hook(lora, scale))
            handles.append(h)

        # Forward pass
        pred_features, _ = self.converter(content, spk_embed, ir_params)

        # L1 loss on mel portion
        loss = F.l1_loss(
            pred_features[:, :mel_target.shape[1], :],
            mel_target,
        )

        # Backward (only LoRA params have grad)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Remove hooks
        for h in handles:
            h.remove()

        return loss.item()


class FewShotAdapterGTM:
    """LoRA-based few-shot adaptation for GTM converter.

    When GTM is enabled, LoRA targets the GTM projection layer
    (``gtm.proj``: ``Linear(d_speaker, n_entries * d_entry)``)
    instead of the per-block FiLM layers.
    """

    def __init__(
        self,
        converter: ConverterStudentGTM,
        rank: int = LORA_RANK,
        alpha: int = LORA_ALPHA,
    ) -> None:
        self.converter = converter
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Single LoRA layer targeting the GTM projection
        d_in = converter.gtm.proj.in_features
        d_out = converter.gtm.proj.out_features
        self.lora_layer = LoRALayer(d_in, d_out, rank=rank)

    def apply_lora_delta(self, lora_delta: torch.Tensor) -> None:
        """Load LoRA delta into parameters."""
        lora = self.lora_layer
        a_size = lora.lora_A.numel()
        lora.lora_A.data = lora_delta[:a_size].reshape_as(lora.lora_A)
        lora.lora_B.data = lora_delta[a_size:].reshape_as(lora.lora_B)

    def merge_weights(self) -> None:
        """Merge LoRA weights into GTM projection."""
        delta_w = self.lora_layer()  # [d_out, d_in]
        self.converter.gtm.proj.weight.data += self.scale * delta_w
        logger.info("Merged LoRA weights into GTM projection")

    def get_lora_parameters(self) -> list[nn.Parameter]:
        return list(self.lora_layer.parameters())

    def get_lora_delta_flat(self) -> torch.Tensor:
        return self.lora_layer.delta_flat

    def finetune_step(
        self,
        optimizer: torch.optim.Optimizer,
        content: torch.Tensor,
        spk_embed: torch.Tensor,
        ir_params: torch.Tensor,
        mel_target: torch.Tensor,
    ) -> float:
        """Single fine-tuning step for GTM adapter.

        Uses a forward hook to inject LoRA delta so gradients flow
        back to LoRA parameters without modifying original weights.

        Returns:
            Loss value.
        """
        self.converter.train()

        # Register hook to add LoRA delta to GTM projection output
        lora = self.lora_layer
        scale = self.scale

        def lora_hook(module, inp, output):
            delta_w = scale * lora()  # [d_out, d_in]
            return output + F.linear(inp[0], delta_w)

        handle = self.converter.gtm.proj.register_forward_hook(lora_hook)

        # Forward pass
        pred_features, _ = self.converter(content, spk_embed, ir_params)

        # L1 loss on mel portion of output
        loss = F.l1_loss(
            pred_features[:, :mel_target.shape[1], :],
            mel_target,
        )

        # Backward (only LoRA params have grad)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        handle.remove()

        return loss.item()


@dataclass
class FewShotConfig:
    """Configuration for few-shot LoRA fine-tuning."""

    max_steps: int = 200
    lr: float = 1e-3
    segment_frames: int = 200  # 2 sec random crop
    use_gtm: bool = False
    log_every: int = 10


class FewShotFinetuner:
    """End-to-end few-shot fine-tuning pipeline.

    Orchestrates data preparation, LoRA training loop, and delta extraction.
    """

    def __init__(
        self,
        converter: ConverterStudent | ConverterStudentGTM,
        content_encoder: nn.Module,
        spk_embed: torch.Tensor,
        config: FewShotConfig,
    ) -> None:
        if config.use_gtm:
            if not isinstance(converter, ConverterStudentGTM):
                raise TypeError(
                    f"use_gtm=True requires ConverterStudentGTM, got {type(converter).__name__}"
                )
            self.adapter = FewShotAdapterGTM(converter)
        else:
            self.adapter = FewShotAdapter(converter)

        self.content_encoder = content_encoder
        self.content_encoder.eval()
        for p in self.content_encoder.parameters():
            p.requires_grad = False

        for p in converter.parameters():
            p.requires_grad = False

        self.spk_embed = spk_embed
        self.config = config
        self.optimizer = torch.optim.Adam(
            self.adapter.get_lora_parameters(), lr=config.lr,
        )

    def prepare_data(
        self,
        audio_paths: list[str | Path],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Convert audio files to (content, mel) pairs.

        Returns:
            List of ``(content[1, 256, T], mel[1, 80, T])`` tuples.
        """
        from tmrvc_core.audio import compute_mel
        from tmrvc_data.preprocessing import load_and_resample

        pairs = []
        for path in audio_paths:
            waveform, _sr = load_and_resample(str(path))
            mel = compute_mel(waveform.unsqueeze(0))  # [1, 80, T]
            f0 = torch.zeros(1, 1, mel.shape[-1])     # [1, 1, T]

            with torch.no_grad():
                content, _ = self.content_encoder(mel, f0)  # [1, 256, T]

            pairs.append((content, mel))
        return pairs

    def finetune_iter(
        self,
        data: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> Iterator[tuple[int, float]]:
        """Generator yielding ``(step, loss)`` for each training step."""
        ir_params = torch.zeros(1, N_IR_PARAMS)
        spk = self.spk_embed.unsqueeze(0)

        for step in range(1, self.config.max_steps + 1):
            content, mel = random.choice(data)
            T = content.shape[-1]
            seg = self.config.segment_frames

            if T > seg:
                start = random.randint(0, T - seg)
                content_seg = content[:, :, start:start + seg]
                mel_seg = mel[:, :, start:start + seg]
            else:
                content_seg = content
                mel_seg = mel

            loss = self.adapter.finetune_step(
                self.optimizer, content_seg, spk,
                ir_params, mel_seg,
            )
            yield step, loss

    def get_lora_delta(self) -> torch.Tensor:
        """Return trained LoRA delta padded to LORA_DELTA_SIZE."""
        delta = self.adapter.get_lora_delta_flat()
        if delta.numel() < LORA_DELTA_SIZE:
            delta = F.pad(delta, (0, LORA_DELTA_SIZE - delta.numel()))
        return delta
