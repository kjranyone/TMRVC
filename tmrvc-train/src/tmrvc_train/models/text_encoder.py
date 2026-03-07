"""TextEncoder: phoneme-based Transformer encoder for TTS."""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_MODEL,
    N_LANGUAGES,
    PHONEME_VOCAB_SIZE,
    UCLM_N_HEADS,
    UCLM_N_LAYERS,
)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = PHONEME_VOCAB_SIZE,
        d_model: int = D_MODEL,
        n_layers: int = 4,
        n_heads: int = 8,
        ff_dim: int = 1024,
        n_languages: int = N_LANGUAGES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.phoneme_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_embed = nn.Embedding(n_languages, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, phoneme_ids, language_ids, phoneme_lengths=None):
        B, L = phoneme_ids.shape
        x = self.phoneme_embed(phoneme_ids) + self.lang_embed(language_ids).unsqueeze(1)
        x = self.embed_dropout(self.pos_enc(x))
        
        mask = None
        if phoneme_lengths is not None:
            mask = (torch.arange(L, device=x.device).unsqueeze(0) >= phoneme_lengths.unsqueeze(1))
            
        x = self.encoder(x, src_key_padding_mask=mask)
        return self.output_norm(x).transpose(1, 2)
