"""TextEncoder: phoneme-based Transformer encoder for TTS.

Converts phoneme ID sequences to continuous text features that can be
used by DurationPredictor, F0Predictor, and ContentSynthesizer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tmrvc_core.constants import (
    D_TEXT_ENCODER,
    N_LANGUAGES,
    N_TEXT_ENCODER_HEADS,
    N_TEXT_ENCODER_LAYERS,
    PHONEME_VOCAB_SIZE,
    TEXT_ENCODER_FF_DIM,
)


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: ``[B, L, d_model]`` input tensor.

        Returns:
            ``[B, L, d_model]`` with positional encoding added.
        """
        return x + self.pe[:, :x.size(1)]


class TextEncoder(nn.Module):
    """Phoneme-based Transformer encoder.

    Architecture::

        phoneme_ids[B, L] → Embedding(vocab, d) + LangEmbedding(n_lang, d)
            + PositionalEncoding
            → TransformerEncoder (n_layers, d, n_heads, ff_dim)
            → text_features[B, d, L]

    Args:
        vocab_size: Phoneme vocabulary size.
        d_model: Hidden dimension (default: D_TEXT_ENCODER=256).
        n_layers: Number of Transformer encoder layers.
        n_heads: Number of attention heads.
        ff_dim: Feed-forward dimension.
        n_languages: Number of language IDs.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = PHONEME_VOCAB_SIZE,
        d_model: int = D_TEXT_ENCODER,
        n_layers: int = N_TEXT_ENCODER_LAYERS,
        n_heads: int = N_TEXT_ENCODER_HEADS,
        ff_dim: int = TEXT_ENCODER_FF_DIM,
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
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        phoneme_ids: torch.Tensor,
        language_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode phoneme sequence.

        Args:
            phoneme_ids: ``[B, L]`` phoneme index tensor.
            language_ids: ``[B]`` language index tensor.
            phoneme_lengths: ``[B]`` unpadded phoneme counts for masking.

        Returns:
            ``[B, d_model, L]`` text features (channel-first for conv layers).
        """
        B, L = phoneme_ids.shape

        # Embeddings
        x = self.phoneme_embed(phoneme_ids)  # [B, L, d]
        x = x + self.lang_embed(language_ids).unsqueeze(1)  # broadcast to [B, L, d]
        x = self.pos_enc(x)
        x = self.embed_dropout(x)

        # Padding mask for Transformer
        src_key_padding_mask = None
        if phoneme_lengths is not None:
            src_key_padding_mask = (
                torch.arange(L, device=phoneme_ids.device).unsqueeze(0)
                >= phoneme_lengths.unsqueeze(1)
            )  # [B, L] True = masked

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, d]
        x = self.output_norm(x)

        # Transpose to channel-first: [B, d, L]
        return x.transpose(1, 2)
