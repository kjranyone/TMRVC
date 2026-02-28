"""
Transformer-based Token Model for TMRVC Codec-Latent Pipeline

Streaming token prediction using causal Transformer with KV-cache.
Input: Context tokens [B, K, L] + speaker embedding [B, 192]
Output: Next tokens [B, K]

References:
- VALL-E: Neural Codec Language Models (arXiv:2301.02111)
- MusicGen: Transformer-based music generation
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TokenModelConfig:
    n_codebooks: int = 4
    codebook_size: int = 1024
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 6
    context_length: int = 10
    d_spk: int = 192
    d_f0: int = 2
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TokenModelConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape

        residual = x
        x = self.norm(x)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_kv_cache = (
            k[:, :, -self.config.context_length :],
            v[:, :, -self.config.context_length :],
        )

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        L_q = q.shape[2]
        L_k = k.shape[2]
        causal_mask = torch.triu(
            torch.full((L_q, L_k), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=L_k - L_q + 1,
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.config.dropout, training=self.training
        )

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)

        return attn_output + residual, new_kv_cache


class FeedForward(nn.Module):
    def __init__(self, config: TokenModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_model * 4)
        self.fc2 = nn.Linear(config.d_model * 4, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(self, config: TokenModelConfig):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ff = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, new_kv_cache = self.attn(x, kv_cache)
        x = self.ff(x)
        return x, new_kv_cache


class FiLMConditioner(nn.Module):
    def __init__(self, d_cond: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, d_model * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(cond)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(1) * x + beta.unsqueeze(1)


class TokenModel(nn.Module):
    def __init__(self, config: Optional[TokenModelConfig] = None):
        super().__init__()
        self.config = config or TokenModelConfig()

        self.token_embeddings = nn.ModuleList(
            [
                nn.Embedding(self.config.codebook_size, self.config.d_model)
                for _ in range(self.config.n_codebooks)
            ]
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.config.context_length, self.config.d_model) * 0.02
        )

        self.spk_proj = nn.Linear(self.config.d_spk, self.config.d_model)
        self.f0_proj = nn.Linear(self.config.d_f0, self.config.d_model)

        self.film = FiLMConditioner(self.config.d_model, self.config.d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        self.output_heads = nn.ModuleList(
            [
                nn.Linear(self.config.d_model, self.config.codebook_size)
                for _ in range(self.config.n_codebooks)
            ]
        )

        self._init_weights()

    def _init_weights(self):
        for emb in self.token_embeddings:
            nn.init.normal_(emb.weight, std=0.02)

        for head in self.output_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        spk_embed: torch.Tensor,
        f0_condition: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        B, K, L = tokens.shape
        assert K == self.config.n_codebooks, (
            f"Expected {self.config.n_codebooks} codebooks, got {K}"
        )

        x = torch.zeros(
            B, L, self.config.d_model, device=tokens.device, dtype=tokens.dtype
        )
        for i, emb in enumerate(self.token_embeddings):
            x = x + emb(tokens[:, i, :])

        x = x + self.pos_embedding[:, :L, :]

        spk_cond = self.spk_proj(spk_embed)
        x = self.film(x, spk_cond)

        if f0_condition is not None:
            f0_emb = self.f0_proj(f0_condition)
            x = x + f0_emb

        if kv_caches is None:
            kv_caches = [None] * self.config.n_layers

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, new_kv = layer(x, kv_caches[i])
            new_kv_caches.append(new_kv)

        x_last = x[:, -1, :]

        logits = []
        for head in self.output_heads:
            logits.append(head(x_last))
        logits = torch.stack(logits, dim=1)

        return logits, new_kv_caches

    @torch.no_grad()
    def generate_next_tokens(
        self,
        tokens: torch.Tensor,
        spk_embed: torch.Tensor,
        f0_condition: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        logits, new_kv_caches = self.forward(tokens, spk_embed, f0_condition, kv_caches)

        logits = logits / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, :, -1:]] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        next_tokens = torch.zeros(
            logits.shape[0], logits.shape[1], dtype=torch.long, device=logits.device
        )
        for i in range(logits.shape[1]):
            next_tokens[:, i] = torch.multinomial(probs[:, i, :], 1).squeeze(-1)

        return next_tokens, new_kv_caches

    def init_kv_cache(
        self, batch_size: int, device: torch.device
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        return [None] * self.config.n_layers

    def kv_cache_to_flat(
        self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        cache_tensors = []
        for kv in kv_caches:
            if kv is not None:
                k, v = kv
                cache_tensors.append(k)
                cache_tensors.append(v)
        if cache_tensors:
            return torch.stack([c.flatten(1) for c in cache_tensors], dim=0)
        return torch.zeros(
            self.config.n_layers * 2,
            1,
            self.config.n_heads
            * self.config.context_length
            * (self.config.d_model // self.config.n_heads),
            device=kv_caches[0][0].device if kv_caches[0] is not None else "cpu",
        )

    def flat_to_kv_cache(
        self, flat: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        B = flat.shape[1]
        head_dim = self.config.d_model // self.config.n_heads
        kv_caches = []
        idx = 0
        for _ in range(self.config.n_layers):
            if idx + 2 <= flat.shape[0]:
                k = flat[idx].view(B, self.config.n_heads, -1, head_dim)
                v = flat[idx + 1].view(B, self.config.n_heads, -1, head_dim)
                kv_caches.append((k, v))
            else:
                kv_caches.append(None)
            idx += 2
        return kv_caches


def create_token_model(config: Optional[TokenModelConfig] = None) -> TokenModel:
    return TokenModel(config)
