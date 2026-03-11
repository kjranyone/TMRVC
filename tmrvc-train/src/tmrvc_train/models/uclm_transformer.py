"""Dual-stream UCLM Transformer with Contract-compliant I/O.

Modern LLM-style backbone: RMSNorm, RoPE, GQA, SwiGLU, pre-norm, Cross-Attention.
Supports FlashAttention2 when available for faster training/serving on CUDA GPUs.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# FlashAttention2 conditional import
# ---------------------------------------------------------------------------
try:
    from flash_attn import flash_attn_func  # type: ignore[import-untyped]

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes rotary sin/cos tables up to *max_seq_len* and expands on
    demand so the buffer never needs re-allocation during training."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self._max_cached = 0
        # inv_freq is not a learned parameter but we register it for device tracking
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Eagerly build cache for the requested length
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        if seq_len <= self._max_cached:
            return
        self._max_cached = seq_len
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim/2]
        # Duplicate for (cos, sin) pairing across the full head_dim
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) each of shape ``[seq_len, head_dim]``."""
        if seq_len > self._max_cached:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
    position_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply rotary embeddings to *x* ``[B, n_heads, T, head_dim]``.

    If *position_indices* [B, T] is provided, it uses those as indices into
    the RoPE table instead of a simple linear offset. This prevents attention
    collapse during pointer stalls (Progress-aware RoPE).
    """
    B, n_h, T, hd = x.shape
    if position_indices is not None:
        # SOTA fix: Combine phoneme-based position with temporal progression (offset).
        # Even during a pointer stall, the RoPE index must increment frame-by-frame
        # to allow the Transformer to distinguish between temporal steps.
        t_seq = torch.arange(T, device=x.device).unsqueeze(0) + offset
        indices = (position_indices.long() + t_seq.long()).clamp(max=cos.shape[0] - 1)
        
        cos_b = cos[indices].unsqueeze(1)  # [B, 1, T, hd]
        sin_b = sin[indices].unsqueeze(1)  # [B, 1, T, hd]
        return x * cos_b + _rotate_half(x) * sin_b
    else:
        # Legacy linear offset mode
        cos = cos[offset : offset + T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, hd]
        sin = sin[offset : offset + T].unsqueeze(0).unsqueeze(0)
        return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Prompt Resampler (condenses T_prompt into fixed N_summary tokens)
# ---------------------------------------------------------------------------

class PromptResampler(nn.Module):
    """Condenses a variable-length prompt sequence into fixed summary tokens.

    Uses a learned query-based attention (Perceiver style) to aggregate
    information from the prompt tokens.
    """

    def __init__(self, d_model: int, n_summary: int = 32, n_heads: int = 8):
        super().__init__()
        self.n_summary = n_summary
        self.d_model = d_model
        # Learned summary queries
        self.queries = nn.Parameter(torch.randn(n_summary, d_model) * 0.02)
        self.attn = CrossAttention(d_model, n_heads)
        self.norm = RMSNorm(d_model)
        self.ff = SwiGLUFFN(d_model, d_model * 4)

    def forward(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            prompt_tokens: [B, T_prompt, d_model]
        Returns:
            summary_tokens: [B, n_summary, d_model]
        """
        B = prompt_tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, n_summary, d_model]

        # Attend to prompt tokens
        h = self.attn(q, prompt_tokens)
        h = self.norm(q + h)
        h = h + self.ff(h)
        return h


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward (Shazeer 2020).

    Projects to ``d_ff * 2`` so that the gated and linear paths each get
    ``d_ff`` dimensions, then projects back to ``d_model``.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        # Gate and up projections are fused into a single linear for clarity
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


# ---------------------------------------------------------------------------
# Grouped-Query Attention (GQA) with RoPE & Flash Attention support
# ---------------------------------------------------------------------------

class CausalGQAttention(nn.Module):
    """Causal Grouped-Query Attention.

    * *n_heads* query heads are grouped into *n_kv_heads* key/value groups.
    * RoPE is applied to queries and keys.
    * Uses ``F.scaled_dot_product_attention`` (Flash Attention 2 path) when
      the PyTorch version supports it; falls back to manual attention otherwise.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
        max_seq_len: int = 4096,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.n_rep = n_heads // self.n_kv_heads  # repetitions per KV group
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=max_seq_len)

        # Check for SDPA availability (PyTorch >= 2.0)
        self._use_sdpa = hasattr(F, "scaled_dot_product_attention")

        # FlashAttention2: only usable on CUDA with fp16/bf16, not during ONNX export
        self._use_flash_attn = use_flash_attn and HAS_FLASH_ATTN

    @staticmethod
    def _expand_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat KV heads to match the number of query heads.

        Input:  [B, n_kv_heads, T, hd]
        Output: [B, n_heads, T, hd]
        """
        if n_rep == 1:
            return kv
        B, H, T, D = kv.shape
        return kv[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)

    def forward(
        self,
        x: torch.Tensor,
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
        position_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Determine the RoPE offset from the KV cache length
        rope_offset = k_cache.shape[2] if k_cache is not None else 0
        cos, sin = self.rope(rope_offset + T)

        q = apply_rotary_emb(q, cos, sin, offset=rope_offset, position_indices=position_indices)
        k = apply_rotary_emb(k, cos, sin, offset=rope_offset, position_indices=position_indices)

        # Concatenate KV cache
        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Expand KV heads for GQA
        k_exp = self._expand_kv(k, self.n_rep)
        v_exp = self._expand_kv(v, self.n_rep)

        # Compute attention -- choose the fastest available backend
        _is_onnx_export = torch.onnx.is_in_onnx_export() if hasattr(torch.onnx, "is_in_onnx_export") else False
        _on_cuda = x.is_cuda
        _dtype_ok = x.dtype in (torch.float16, torch.bfloat16)

        if (
            self._use_flash_attn
            and _on_cuda
            and _dtype_ok
            and not _is_onnx_export
            and T > 1
            and k_cache is None
        ):
            # FlashAttention2 path: expects (B, T, n_heads, head_dim) layout
            # q is [B, n_heads, T, hd], k_exp/v_exp are [B, n_heads, S, hd]
            q_fa = q.transpose(1, 2)  # [B, T, n_heads, hd]
            k_fa = k_exp.transpose(1, 2)  # [B, S, n_heads, hd]
            v_fa = v_exp.transpose(1, 2)  # [B, S, n_heads, hd]
            out = flash_attn_func(
                q_fa, k_fa, v_fa,
                dropout_p=self.dropout_p if self.training else 0.0,
                causal=True,
            )  # [B, T, n_heads, hd]
            out = out.reshape(B, T, C)
        elif self._use_sdpa and T > 1 and k_cache is None and not _is_onnx_export:
            # Use PyTorch SDPA (Flash / Memory-efficient) for training (no cache)
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        elif self._use_sdpa and T == 1 and not _is_onnx_export:
            # Single-step decoding: no causal mask needed
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).contiguous().view(B, T, C)
        else:
            # Manual fallback (covers ONNX export, cached multi-step, old PyTorch, CPU fp32)
            S = k_exp.shape[2]
            attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
            if T > 1 and k_cache is None:
                causal_mask = torch.triu(
                    torch.ones(T, S, device=x.device, dtype=torch.bool), diagonal=1,
                )
                attn = attn.masked_fill(causal_mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v_exp)
            out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out), k, v


class CrossAttention(nn.Module):
    """Grouped-Query Cross Attention against an external memory sequence."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        assert n_heads % self.n_kv_heads == 0, (
            f"n_heads ({n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.n_rep = n_heads // self.n_kv_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] query sequence.
            memory: [B, S, d_model] key/value sequence.
        """
        B, T, C = x.shape
        _, S, _ = memory.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(memory).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(memory).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k_exp = CausalGQAttention._expand_kv(k, self.n_rep)
        v_exp = CausalGQAttention._expand_kv(v, self.n_rep)

        # Cross-attention uses SDPA without causal mask
        out = F.scaled_dot_product_attention(
            q, k_exp, v_exp,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


# Backward-compatible aliases
CausalSelfAttention = CausalGQAttention
GQAAttention = CausalGQAttention


# ---------------------------------------------------------------------------
# Modern Transformer Block (pre-norm with RMSNorm + SwiGLU + GQA + RoPE)
# ---------------------------------------------------------------------------

class ModernTransformerBlock(nn.Module):
    """Modern LLM-style transformer block using RMSNorm, RoPE, GQA, SwiGLU.

    This is the recommended block for new training runs (use_modern_backbone=True).
    Identical to TransformerBlock in interface, but uses named modern primitives
    explicitly for clarity and auditability.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        # Pre-norm with RMSNorm (not LayerNorm)
        self.norm1 = RMSNorm(d_model)
        # GQA self-attention with RoPE (via CausalGQAttention)
        self.attn = GQAAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, max_seq_len=max_seq_len,
        )
        # Cross-attention layer
        self.norm_cross = RMSNorm(d_model)
        self.cross_attn = CrossAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout
        )
        # SwiGLU FFN (not standard GELU FFN)
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLUFFN(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # SOTA: Speaker FiLM
        self.speaker_film = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x, memory=None, k_cache=None, v_cache=None, position_indices=None, speaker_embed_mapped=None):
        # Apply speaker FiLM if provided
        if speaker_embed_mapped is not None:
            film_params = self.speaker_film(speaker_embed_mapped)
            gamma, beta = film_params.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        # Self-attention with pre-norm
        h, nk, nv = self.attn(self.norm1(x), k_cache, v_cache, position_indices=position_indices)
        x = x + self.dropout(h)

        # Cross-attention against phoneme memory
        if memory is not None:
            h_cross = self.cross_attn(self.norm_cross(x), memory)
            x = x + self.dropout(h_cross)

        # SwiGLU FFN with pre-norm
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, nk, nv


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm with RMSNorm + SwiGLU + Cross-Attention)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        n_kv_heads: Optional[int] = None,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout, max_seq_len=max_seq_len,
        )
        self.norm_cross = RMSNorm(d_model)
        self.cross_attn = CrossAttention(
            d_model, n_heads, n_kv_heads=n_kv_heads, dropout=dropout
        )
        self.norm2 = RMSNorm(d_model)
        self.ff = SwiGLUFFN(d_model, d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
        # SOTA: Speaker FiLM
        self.speaker_film = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(
        self,
        x,
        memory=None,
        k_cache=None,
        v_cache=None,
        position_indices=None,
        speaker_embed_mapped=None,
    ):
        # Apply speaker FiLM if provided (per-layer speaker conditioning)
        if speaker_embed_mapped is not None:
            film_params = self.speaker_film(speaker_embed_mapped)
            gamma, beta = film_params.chunk(2, dim=-1)
            x = x * (1 + gamma) + beta

        # Self-attention
        h, nk, nv = self.attn(self.norm1(x), k_cache, v_cache, position_indices=position_indices)
        x = x + self.dropout(h)

        # Cross-attention (against full phoneme sequence/memory)
        if memory is not None:
            h_cross = self.cross_attn(self.norm_cross(x), memory)
            x = x + self.dropout(h_cross)

        # FFN
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, nk, nv


# ---------------------------------------------------------------------------
# CodecTransformer
# ---------------------------------------------------------------------------

class CodecTransformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        n_heads=8,
        n_layers=12,
        rvq_vocab_size=1024,
        n_codebooks=8,
        control_vocab_size=64,
        d_speaker=192,
        n_kv_heads=None,
        max_seq_len=4096,
        n_prompt_summary_tokens=32,
        use_modern_backbone: bool = False,
    ):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.n_heads = n_heads
        self.n_codebooks = n_codebooks
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
        self.head_dim = d_model // n_heads
        self.use_modern_backbone = use_modern_backbone

        self.speaker_proj = nn.Linear(d_speaker, d_model)
        self.f0_proj = nn.Linear(2, d_model)  # F0 fusion layer (UCLM v3)

        # Prompt Resampler (condenses T_prompt into fixed N_summary tokens)
        self.prompt_resampler = PromptResampler(d_model, n_summary=n_prompt_summary_tokens, n_heads=n_heads)

        # Acoustic context embeds (Stream A) - past A_{t-1}
        self.a_ctx_embeds = nn.ModuleList(
            [nn.Embedding(rvq_vocab_size, d_model // n_codebooks) for _ in range(n_codebooks)]
        )
        self.a_ctx_fusion = nn.Linear(d_model, d_model)

        # Control context embeds (Stream B) - past B_{t-1}
        self.b_ctx_embeds = nn.ModuleList(
            [nn.Embedding(control_vocab_size, d_model // 4) for _ in range(4)]
        )
        self.b_ctx_fusion = nn.Linear(d_model, d_model)

        # Select block class based on use_modern_backbone flag
        BlockClass = ModernTransformerBlock if use_modern_backbone else TransformerBlock
        self.layers = nn.ModuleList(
            [
                BlockClass(
                    d_model,
                    n_heads,
                    d_model * 4,
                    n_kv_heads=self.n_kv_heads,
                    max_seq_len=max_seq_len,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.acoustic_heads = nn.ModuleList(
            [nn.Linear(d_model, rvq_vocab_size) for _ in range(n_codebooks)]
        )
        self.control_heads = nn.ModuleList(
            [nn.Linear(d_model, control_vocab_size) for _ in range(4)]
        )

    def forward(
        self,
        queries,  # [B, T_q, d_model] current frames
        memory,   # [B, L_mem, d_model] phoneme sequence for cross-attn
        a_ctx,
        b_ctx,
        speaker_embed,
        state_cond,
        prompt_tokens=None,  # [B, T_prompt, d_model] raw prompt
        prompt_summary_tokens=None,  # [B, n_summary, d_model] pre-condensed
        cfg_scale=1.0,
        kv_caches=None,
        max_seq_len=200,
        f0_condition=None,
        position_indices=None,
    ):
        # x starts as the queries (temporal frame base)
        x = queries
        B, T, _ = x.shape

        # 1. Handle Prompt Conditioning (Few-shot)
        # Use pre-condensed summary if available, otherwise run resampler
        if prompt_summary_tokens is None and prompt_tokens is not None:
            prompt_summary_tokens = self.prompt_resampler(prompt_tokens)

        if prompt_summary_tokens is not None:
            # Concatenate summary tokens to the phoneme memory for cross-attention
            memory = torch.cat([prompt_summary_tokens, memory], dim=1)

        # Add F0 conditioning if provided
        if f0_condition is not None:
            # Ensure f0_condition matches sequence length T
            if f0_condition.shape[1] > T:
                f0_condition = f0_condition[:, :T, :]
            elif f0_condition.shape[1] < T:
                # Pad if necessary (unlikely in streaming but safe)
                f0_condition = F.pad(f0_condition, (0, 0, 0, T - f0_condition.shape[1]))
            x = x + self.f0_proj(f0_condition)

        # a_ctx: [B, 8, T] - past acoustic tokens A_{t-1}
        # Slice to match current frame sequence length T
        a_ctx_curr = a_ctx[:, :, :T]
        a_embeds = torch.stack(
            [self.a_ctx_embeds[i](a_ctx_curr[:, i, :]) for i in range(self.n_codebooks)], dim=1,
        )  # [B, 8, T, d_model//8]
        a_ctx_fused = self.a_ctx_fusion(
            a_embeds.permute(0, 2, 1, 3).reshape(B, T, -1),
        )  # [B, T, d_model]
        x = x + a_ctx_fused

        # b_ctx: [B, 4, T] - past control tokens B_{t-1}
        b_ctx_curr = b_ctx[:, :, :T]
        b_embeds = torch.stack(
            [self.b_ctx_embeds[i](b_ctx_curr[:, i, :]) for i in range(4)], dim=1,
        )  # [B, 4, T, d_model//4]
        b_ctx_fused = self.b_ctx_fusion(
            b_embeds.permute(0, 2, 1, 3).reshape(B, T, -1),
        )  # [B, T, d_model]
        x = x + b_ctx_fused

        # SOTA: Map speaker embedding once for FiLM conditioning across all layers
        spk_mapped = self.speaker_proj(speaker_embed).unsqueeze(1)  # [B, 1, d_model]
        
        # Add speaker embedding as a base bias (optional, but kept for stability)
        x = x + spk_mapped

        # Add voice state condition (temporal)
        if state_cond.dim() == 2:
            sc = state_cond.unsqueeze(1).expand(-1, T, -1)
        else:
            sc = state_cond

        # Ensure state_cond matches T (in case of streaming/KV cache)
        if sc.shape[1] > T:
            sc = sc[:, -T:, :]

        x = x + sc

        # Handle KV cache
        kv_list = None
        if kv_caches is not None:
            if isinstance(kv_caches, torch.Tensor):
                kv_list = []
                step = self.n_kv_heads * self.head_dim * max_seq_len
                offset = 0
                for _ in range(self.n_layers):
                    k = kv_caches[:, offset : offset + step].view(
                        B, self.n_kv_heads, max_seq_len, self.head_dim,
                    )
                    v = kv_caches[:, offset + step : offset + 2 * step].view(
                        B, self.n_kv_heads, max_seq_len, self.head_dim,
                    )
                    kv_list.append((k, v))
                    offset += 2 * step
            else:
                kv_list = kv_caches

        new_kv_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            k_in = kv_list[i][0] if kv_list else None
            v_in = kv_list[i][1] if kv_list else None
            # layer now takes 'memory' (full phonemes) for cross-attention
            # and speaker_embed_mapped for FiLM
            x, nk, nv = layer(
                x, 
                memory=memory, 
                k_cache=k_in, 
                v_cache=v_in,
                position_indices=position_indices,
                speaker_embed_mapped=spk_mapped,
            )

            if nk.shape[2] > max_seq_len:
                nk = nk[:, :, -max_seq_len:, :]
                nv = nv[:, :, -max_seq_len:, :]
            new_kv_list.append((nk, nv))

        x = self.norm(x)

        # Dual-stream heads
        logits_a = torch.stack(
            [head(x) for head in self.acoustic_heads], dim=1,
        )  # [B, 8, T, 1024]
        logits_b = torch.stack(
            [head(x) for head in self.control_heads], dim=1,
        )  # [B, 4, T, 64]

        if kv_caches is not None and isinstance(kv_caches, torch.Tensor):
            out_kv = torch.cat(
                [
                    torch.cat([k.reshape(B, -1), v.reshape(B, -1)], dim=1)
                    for k, v in new_kv_list
                ],
                dim=1,
            )
            return logits_a, logits_b, out_kv, x

        return logits_a, logits_b, new_kv_list, x

    def forward_no_cache(
        self,
        queries,
        memory,
        a_ctx,
        b_ctx,
        state_cond,
        speaker_embed,
        cfg_scale=1.0,
        f0_condition=None,
        prompt_tokens=None,
        prompt_summary_tokens=None,
    ):
        """Non-streaming forward for training. History must be shifted."""
        la, lb, _, x = self.forward(
            queries, memory, a_ctx, b_ctx, speaker_embed, state_cond,
            prompt_tokens=prompt_tokens,
            prompt_summary_tokens=prompt_summary_tokens,
            cfg_scale=cfg_scale,
            f0_condition=f0_condition,
        )
        return la, lb, x
