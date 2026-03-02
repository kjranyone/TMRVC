"""Dual-stream UCLM Transformer with Contract-compliant I/O."""

from __future__ import annotations

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
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
        k_cache: Optional[torch.Tensor] = None,
        v_cache: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if k_cache is not None:
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if k.shape[2] > T: # If we have cache, T is current frame
            # Masking not needed if T=1 and we only attend to past
            pass
        elif T > 1:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(self.dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out), k, v


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, k_cache=None, v_cache=None):
        h, nk, nv = self.attn(self.norm1(x), k_cache, v_cache)
        x = x + self.dropout(h)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x, nk, nv


class CodecTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=12, rvq_vocab_size=1024, n_codebooks=8, control_vocab_size=64, d_speaker=192):
        super().__init__()
        self.d_model, self.n_layers = d_model, n_layers
        self.speaker_proj = nn.Linear(d_speaker, d_model)
        self.b_ctx_embeds = nn.ModuleList([nn.Embedding(control_vocab_size, d_model // 4) for _ in range(4)])
        self.b_ctx_fusion = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads, d_model * 4) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.acoustic_heads = nn.ModuleList([nn.Linear(d_model, rvq_vocab_size) for _ in range(n_codebooks)])
        self.control_heads = nn.ModuleList([nn.Linear(d_model, control_vocab_size) for _ in range(4)])

    def forward(self, content_features, b_ctx, speaker_embed, state_cond, cfg_scale=1.0, kv_caches=None, max_seq_len=200):
        # content_features: [B, T, d_model]
        x = content_features
        B, T, _ = x.shape
        
        # b_ctx: [B, 4, T_full] - These should be tokens B_{t-1} for predicting A_t, B_t
        # Slice b_ctx to match content length T
        b_ctx_curr = b_ctx[:, :, :T]
        
        # Embed control context and add to x
        b_embeds = torch.stack([self.b_ctx_embeds[i](b_ctx_curr[:, i, :]) for i in range(4)], dim=1) # [B, 4, T, d_model//4]
        b_ctx_fused = self.b_ctx_fusion(b_embeds.permute(0, 2, 1, 3).reshape(B, T, -1)) # [B, T, d_model]
        x = x + b_ctx_fused
        
        # Add speaker embedding (global)
        x = x + self.speaker_proj(speaker_embed).unsqueeze(1)
        
        # Add voice state condition (temporal)
        if state_cond.dim() == 2:
            sc = state_cond.unsqueeze(1).expand(-1, T, -1)
        else:
            sc = state_cond
        
        # Ensure state_cond matches T (in case of streaming/KV cache)
        if sc.shape[1] > T:
            sc = sc[:, -T:, :]
            
        x = x + sc

        # Handle KV cache (optional flattening for ONNX)
        kv_list = None
        if kv_caches is not None:
            if isinstance(kv_caches, torch.Tensor):
                # Unflatten: [B, kv_cache_size] -> list of (k, v)
                # Size: 2 * n_layers * n_heads * head_dim * max_seq_len
                head_dim = self.d_model // self.n_heads
                kv_list = []
                offset = 0
                step = self.n_heads * head_dim * max_seq_len
                for _ in range(self.n_layers):
                    k = kv_caches[:, offset : offset + step].view(B, self.n_heads, max_seq_len, head_dim)
                    v = kv_caches[:, offset + step : offset + 2 * step].view(B, self.n_heads, max_seq_len, head_dim)
                    kv_list.append((k, v))
                    offset += 2 * step
            else:
                kv_list = kv_caches

        new_kv_list = []
        for i, layer in enumerate(self.layers):
            k_in = kv_list[i][0] if kv_list else None
            v_in = kv_list[i][1] if kv_list else None
            x, nk, nv = layer(x, k_in, v_in)
            
            # If we used cache, we only want to keep the last max_seq_len frames
            if nk.shape[2] > max_seq_len:
                nk = nk[:, :, -max_seq_len:, :]
                nv = nv[:, :, -max_seq_len:, :]
            new_kv_list.append((nk, nv))

        x = self.norm(x)
        
        # Dual-stream heads
        logits_a = torch.stack([head(x) for head in self.acoustic_heads], dim=1) # [B, 8, T, 1024]
        logits_b = torch.stack([head(x) for head in self.control_heads], dim=1)  # [B, 4, T, 64]
        
        # Flatten KV cache if output requested as tensor
        if kv_caches is not None and isinstance(kv_caches, torch.Tensor):
            out_kv = torch.cat([torch.cat([k.reshape(B, -1), v.reshape(B, -1)], dim=1) for k, v in new_kv_list], dim=1)
            return logits_a, logits_b, out_kv
            
        return logits_a, logits_b, new_kv_list

    def forward_no_cache(self, content_features, b_ctx, state_cond, speaker_embed, cfg_scale=1.0):
        """Non-streaming forward for training. b_ctx must be shifted B_{t-1}."""
        la, lb, _ = self.forward(content_features, b_ctx, speaker_embed, state_cond, cfg_scale)
        return la, lb
