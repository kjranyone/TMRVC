"""TextEncoder: phoneme-based Transformer encoder for TTS.

Supports optional inline acting tag embeddings (v4). When acting tags are
present in the input (IDs >= phoneme_vocab_size), they are routed through
a separate acting tag embedding layer. This keeps the phoneme embedding
table frozen/unchanged for backward compatibility.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_MODEL,
    D_SUPRASEGMENTAL,
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
        n_layers: int = 6,
        n_heads: int = 4,
        ff_dim: int = 1024,
        n_languages: int = N_LANGUAGES,
        d_supra: int = D_SUPRASEGMENTAL,
        dropout: float = 0.1,
        acting_tag_vocab_size: int = 0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.phoneme_vocab_size = vocab_size
        self.phoneme_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.lang_embed = nn.Embedding(n_languages, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Suprasegmental projection: [B, L, d_supra] -> [B, L, d_model]
        self.suprasegmental_proj = nn.Linear(d_supra, d_model)

        # Optional acting tag embedding (v4 inline acting tags)
        # Acting tag IDs in the input are offset by phoneme_vocab_size.
        # This layer maps tag-local indices (0..acting_tag_vocab_size-1) to d_model.
        self.acting_tag_embedding: nn.Embedding | None = None
        if acting_tag_vocab_size > 0:
            self.acting_tag_embedding = nn.Embedding(acting_tag_vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, phoneme_ids, language_ids, phoneme_lengths=None, text_suprasegmentals=None):
        B, L = phoneme_ids.shape
        lang = self.lang_embed(language_ids)
        if lang.dim() == 2:
            # language_ids was (B,) scalar per sample -- broadcast over sequence
            lang = lang.unsqueeze(1)

        # Detect acting tag positions (IDs >= phoneme_vocab_size)
        if self.acting_tag_embedding is not None:
            is_acting_tag = phoneme_ids >= self.phoneme_vocab_size

            if is_acting_tag.any():
                # Split into phoneme and acting tag embeddings
                # Clamp phoneme IDs so acting tag positions don't OOB the phoneme table
                phoneme_ids_clamped = phoneme_ids.clamp(max=self.phoneme_vocab_size - 1)
                phoneme_emb = self.phoneme_embed(phoneme_ids_clamped)

                # Compute acting tag local indices (offset by phoneme_vocab_size)
                tag_local_ids = (phoneme_ids - self.phoneme_vocab_size).clamp(min=0)
                tag_emb = self.acting_tag_embedding(tag_local_ids)

                # Merge: use tag embedding where is_acting_tag, phoneme embedding elsewhere
                x = torch.where(is_acting_tag.unsqueeze(-1), tag_emb, phoneme_emb) + lang
            else:
                # No acting tags in this batch -- standard path
                x = self.phoneme_embed(phoneme_ids) + lang
        else:
            # No acting tag layer configured -- pure phoneme path
            x = self.phoneme_embed(phoneme_ids) + lang

        # Add suprasegmental features if provided
        if text_suprasegmentals is not None:
            x = x + self.suprasegmental_proj(text_suprasegmentals)

        x = self.embed_dropout(self.pos_enc(x))

        mask = None
        if phoneme_lengths is not None:
            mask = (torch.arange(L, device=x.device).unsqueeze(0) >= phoneme_lengths.unsqueeze(1))

        x = self.encoder(x, src_key_padding_mask=mask)
        return self.output_norm(x).transpose(1, 2)
