"""Dual-stream UCLM Transformer with KV Cache support for streaming inference."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking and KV cache."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (k, v)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        T_q = q.shape[2]
        T_k = k.shape[2]
        causal_mask = torch.triu(
            torch.ones(T_q, T_k, device=x.device, dtype=torch.bool),
            diagonal=T_k - T_q + 1,
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out, new_kv_cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h, new_kv = self.attn(self.norm1(x), kv_cache)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, new_kv


class CodecTransformer(nn.Module):
    """Dual-stream UCLM Transformer with KV cache for streaming inference.

    Architecture:
        - N_layers of TransformerBlock with pre-norm
        - Dual output heads: acoustic (A_t) and control (B_t)
        - Optional KV cache for efficient streaming

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        rvq_vocab_size: Vocabulary size for acoustic tokens (default: 1024).
        n_codebooks: Number of RVQ codebooks (default: 8).
        control_vocab_size: Vocabulary size for control tokens (default: 64).
        d_speaker: Speaker embedding dimension.
        d_ff: Feed-forward dimension (default: d_model * 4).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        rvq_vocab_size: int = 1024,
        n_codebooks: int = 8,
        control_vocab_size: int = 64,
        d_speaker: int = 192,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_codebooks = n_codebooks
        self.rvq_vocab_size = rvq_vocab_size
        self.control_vocab_size = control_vocab_size

        d_ff = d_ff or d_model * 4

        self.speaker_proj = nn.Linear(d_speaker, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

        self.acoustic_heads = nn.ModuleList(
            [nn.Linear(d_model, rvq_vocab_size) for _ in range(n_codebooks)]
        )

        self.control_heads = nn.ModuleList(
            [nn.Linear(d_model, control_vocab_size) for _ in range(4)]
        )

    @property
    def kv_cache_size(self) -> int:
        """Total size of flattened KV cache per frame."""
        return 2 * self.n_layers * self.n_heads * self.d_model // self.n_heads * 200

    def init_kv_cache(
        self, batch_size: int, max_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """Initialize empty KV cache tensor.

        Returns:
            Flattened KV cache tensor of shape [batch_size, kv_cache_size]
        """
        head_dim = self.d_model // self.n_heads
        cache_size = 2 * self.n_layers * self.n_heads * head_dim * max_seq_len
        return torch.zeros(batch_size, cache_size, device=device)

    def _split_kv_cache(
        self, kv_cache: torch.Tensor, max_seq_len: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split flattened KV cache into per-layer tuples.

        Always returns valid caches (zeros if empty).
        """
        head_dim = self.d_model // self.n_heads
        layer_cache_size = 2 * self.n_heads * head_dim * max_seq_len

        caches = []
        for i in range(self.n_layers):
            start = i * layer_cache_size
            end = start + layer_cache_size
            layer_cache = kv_cache[:, start:end]

            k_cache = layer_cache[:, : layer_cache_size // 2].view(
                -1, self.n_heads, max_seq_len, head_dim
            )
            v_cache = layer_cache[:, layer_cache_size // 2 :].view(
                -1, self.n_heads, max_seq_len, head_dim
            )
            caches.append((k_cache, v_cache))

        return caches

    def _merge_kv_cache(
        self,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        max_seq_len: int,
    ) -> torch.Tensor:
        """Merge per-layer KV caches into flattened tensor."""
        parts = []
        for k, v in caches:
            parts.append(k.reshape(k.shape[0], -1))
            parts.append(v.reshape(v.shape[0], -1))
        return torch.cat(parts, dim=1)

    def forward(
        self,
        content_features: torch.Tensor,
        state_cond: torch.Tensor,
        speaker_embed: torch.Tensor,
        cfg_scale: float = 1.0,
        kv_cache_in: Optional[torch.Tensor] = None,
        max_seq_len: int = 200,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with optional KV cache and CFG.

        Args:
            content_features: [B, T, d_model] - content from VC or text encoder
            state_cond: [B, T, d_model] - voice state condition
            speaker_embed: [B, 192] - speaker embedding
            cfg_scale: CFG amplification scale (1.0 = no guidance, >1.0 = amplify conditioning)
                       Computes: output = uncond + cfg_scale * (cond - uncond)
                       When cfg_scale=1.0, this reduces to: uncond + (cond - uncond) = cond
            kv_cache_in: [B, kv_cache_size] - flattened KV cache from previous frames
                         If None, creates empty cache (for training).
            max_seq_len: Maximum sequence length for KV cache

        Returns:
            logits_a: [B, 8, T, 1024] - acoustic token logits
            logits_b: [B, 4, T, 64] - control token logits
            kv_cache_out: [B, kv_cache_size] - updated flattened KV cache
        """
        B, T, _ = content_features.shape

        if kv_cache_in is None:
            kv_cache_in = self.init_kv_cache(B, max_seq_len, content_features.device)

        layer_caches = self._split_kv_cache(kv_cache_in, max_seq_len)

        spk_cond = self.speaker_proj(speaker_embed).unsqueeze(1)
        x_cond = content_features + state_cond + spk_cond

        new_caches = []
        for i, layer in enumerate(self.layers):
            x_cond, new_kv = layer(x_cond, layer_caches[i])
            new_caches.append(new_kv)

        x_cond = self.norm(x_cond)

        logits_a_cond = torch.stack(
            [head(x_cond) for head in self.acoustic_heads], dim=1
        )
        logits_b_cond = torch.stack(
            [head(x_cond) for head in self.control_heads], dim=1
        )

        zero_spk = torch.zeros_like(speaker_embed)
        zero_state = torch.zeros_like(state_cond)
        spk_uncond = self.speaker_proj(zero_spk).unsqueeze(1)
        x_uncond = content_features + zero_state + spk_uncond

        for i, layer in enumerate(self.layers):
            x_uncond, _ = layer(x_uncond, layer_caches[i])

        x_uncond = self.norm(x_uncond)

        logits_a_uncond = torch.stack(
            [head(x_uncond) for head in self.acoustic_heads], dim=1
        )
        logits_b_uncond = torch.stack(
            [head(x_uncond) for head in self.control_heads], dim=1
        )

        logits_a = logits_a_uncond + cfg_scale * (logits_a_cond - logits_a_uncond)
        logits_b = logits_b_uncond + cfg_scale * (logits_b_cond - logits_b_uncond)

        kv_cache_out = self._merge_kv_cache(new_caches, max_seq_len)

        return logits_a, logits_b, kv_cache_out

    def forward_no_cache(
        self,
        content_features: torch.Tensor,
        state_cond: torch.Tensor,
        speaker_embed: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass without KV cache (for training/batch inference).

        Args:
            content_features: [B, T, d_model]
            state_cond: [B, T, d_model]
            speaker_embed: [B, 192]
            cfg_scale: CFG scale

        Returns:
            logits_a: [B, 8, T, 1024]
            logits_b: [B, 4, T, 64]
        """
        B, T, _ = content_features.shape

        spk_cond = self.speaker_proj(speaker_embed).unsqueeze(1)

        x = content_features + state_cond + spk_cond

        for layer in self.layers:
            x, _ = layer(x, None)

        x = self.norm(x)

        logits_a = torch.stack([head(x) for head in self.acoustic_heads], dim=1)
        logits_b = torch.stack([head(x) for head in self.control_heads], dim=1)

        return logits_a, logits_b
