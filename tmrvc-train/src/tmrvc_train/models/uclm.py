"""Unified Codec Language Model (UCLM).

A single model that performs both TTS and VC through conditioned codec token
transformation. Treats both tasks as instances of the same fundamental problem:
generating acoustic tokens conditioned on input modality, speaker identity,
and voice state parameters.

Usage:
    # TTS mode
    output = model(text_features, voice_state, speaker_embed, mode='tts')

    # VC mode
    output = model(source_tokens, voice_state, speaker_embed, mode='vc')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import (
    D_MODEL,
    D_SPEAKER,
    N_CODEBOOKS,
    VOCAB_SIZE,
)


@dataclass
class UCLMConfig:
    """Configuration for UCLM model."""

    vocab_size: int = VOCAB_SIZE  # 1024
    n_codebooks: int = N_CODEBOOKS  # 8
    d_model: int = D_MODEL  # 256
    n_heads: int = 8
    n_layers: int = 12
    d_speaker: int = D_SPEAKER  # 192
    d_voice_state: int = 8
    d_text: int | None = None  # If None, uses d_model (Identity projection)
    dropout: float = 0.1
    max_seq_len: int = 2000

    def __post_init__(self):
        if self.d_text is None:
            self.d_text = self.d_model


class UCLM(nn.Module):
    """Unified Codec Language Model for TTS and VC.

    Architecture::

        Inputs:
            - text_features: [B, L, d_text] (TTS mode) or None (VC mode)
            - source_tokens: [B, n_codebooks, T] (VC mode) or None (TTS mode)
            - voice_state: [B, T, d_voice_state]
            - speaker_embed: [B, d_speaker]
            - past_tokens: [B, n_codebooks, k] (context from previous blocks)

        Processing:
            1. Encode conditions (text, voice_state, speaker, mode)
            2. Fuse conditions with cross-attention
            3. Predict codec tokens with AR + parallel decoding

        Outputs:
            - logits_ar: [B, T, vocab_size] for first codebook
            - logits_parallel: [B, n_codebooks-1, T, vocab_size] for rest

    Args:
        config: UCLMConfig with model hyperparameters.
    """

    def __init__(self, config: UCLMConfig | None = None) -> None:
        super().__init__()
        self.config = config or UCLMConfig()
        c = self.config

        # Mode embedding (TTS=0, VC=1)
        self.mode_embed = nn.Embedding(2, c.d_model)

        # Speaker conditioning
        self.speaker_proj = nn.Linear(c.d_speaker, c.d_model)

        # Voice state encoder
        from tmrvc_train.models.voice_state_encoder import VoiceStateEncoder

        self.voice_state_encoder = VoiceStateEncoder(
            d_state=c.d_voice_state,
            d_model=c.d_model,
        )

        # Text projection (if text features are different dimension)
        if c.d_text != c.d_model:
            self.text_proj = nn.Linear(c.d_text, c.d_model)
        else:
            self.text_proj = nn.Identity()

        # Source token embedding (for VC mode)
        self.codebook_embed = nn.ModuleList(
            [
                nn.Embedding(c.vocab_size, c.d_model // c.n_codebooks)
                for _ in range(c.n_codebooks)
            ]
        )

        # Context encoder (for past tokens)
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=c.d_model,
                nhead=c.n_heads,
                dim_feedforward=c.d_model * 4,
                dropout=c.dropout,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Main transformer decoder (causal)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=c.d_model * 4,
            dropout=c.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=c.n_layers,
        )

        # Target token embedding (for teacher forcing)
        self.target_codebook_embed = nn.ModuleList(
            [
                nn.Embedding(c.vocab_size, c.d_model // c.n_codebooks)
                for _ in range(c.n_codebooks)
            ]
        )

        # Output heads
        self.ar_head = nn.Linear(c.d_model, c.vocab_size)  # First codebook (AR)
        self.parallel_heads = nn.ModuleList(
            [nn.Linear(c.d_model, c.vocab_size) for _ in range(c.n_codebooks - 1)]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        text_features: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        voice_state: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
        past_tokens: torch.Tensor | None = None,
        target_tokens: torch.Tensor | None = None,
        mode: str = "tts",
    ) -> dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            text_features: [B, L, d_text] text features (TTS mode).
            source_tokens: [B, n_codebooks, T] source audio tokens (VC mode).
            voice_state: [B, T, d_voice_state] voice state parameters.
            speaker_embed: [B, d_speaker] speaker embedding.
            past_tokens: [B, n_codebooks, k] context from previous blocks.
            target_tokens: [B, n_codebooks, T] target tokens for training.
            mode: "tts" or "vc".

        Returns:
            Dict with:
                - logits_ar: [B, T, vocab_size]
                - logits_parallel: [B, n_codebooks-1, T, vocab_size]
        """
        c = self.config

        # Determine batch size and sequence length
        if voice_state is not None:
            B, T, _ = voice_state.shape
            device = voice_state.device
        elif target_tokens is not None:
            B, n_cb, T = target_tokens.shape
            device = target_tokens.device
        else:
            raise ValueError("Must provide voice_state or target_tokens")

        # Get mode embedding
        mode_id = 0 if mode == "tts" else 1
        mode_cond = (
            self.mode_embed(torch.tensor([mode_id], device=device))
            .unsqueeze(0)
            .expand(B, -1, -1)
        )  # [B, 1, d_model]

        # Encode voice state
        if voice_state is not None:
            state_cond = self.voice_state_encoder(voice_state)  # [B, T, d_model]
        else:
            state_cond = torch.zeros(B, T, c.d_model, device=device)

        # Encode speaker
        if speaker_embed is not None:
            spk_cond = self.speaker_proj(speaker_embed).unsqueeze(1)  # [B, 1, d_model]
        else:
            spk_cond = torch.zeros(B, 1, c.d_model, device=device)

        # Encode text (TTS mode)
        if mode == "tts" and text_features is not None:
            text_cond = self.text_proj(text_features)  # [B, L, d_model]
            memory = text_cond  # [B, L, d_model] for batch_first=True
        else:
            # No text conditioning for VC
            memory = state_cond  # [B, T, d_model] for batch_first=True

        # Encode source tokens (VC mode)
        if mode == "vc" and source_tokens is not None:
            source_cond = self._encode_source_tokens(source_tokens)  # [B, T, d_model]
            state_cond = state_cond + source_cond

        # Combine conditions
        cond = state_cond + spk_cond + mode_cond  # [B, T, d_model]

        # Encode past context
        if past_tokens is not None:
            past_cond = self._encode_past_tokens(past_tokens)  # [B, k, d_model]
            # Prepend past to cond
            cond = torch.cat([past_cond, cond], dim=1)  # [B, k+T, d_model]

        # Target embedding (teacher forcing)
        if target_tokens is None:
            tgt = cond[:, -T:]
        else:
            tgt_emb = self._embed_target_tokens(target_tokens)
            tgt = tgt_emb + cond[:, -T:]

        # Causal mask for autoregressive generation
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)

        # Transformer decoder (batch_first=True)
        out = self.transformer(
            tgt,
            memory,  # [B, S, d_model] for batch_first=True
            tgt_mask=tgt_mask,
        )  # [B, T, d_model]

        # Predict
        logits_ar = self.ar_head(out)  # [B, T, vocab_size]

        logits_parallel = torch.stack(
            [head(out) for head in self.parallel_heads], dim=1
        )  # [B, n_codebooks-1, T, vocab_size]

        return {
            "logits_ar": logits_ar,
            "logits_parallel": logits_parallel,
        }

    def _encode_source_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode source tokens for VC mode.

        Args:
            tokens: [B, n_codebooks, T]

        Returns:
            [B, T, d_model]
        """
        embeddings = [
            embed(tokens[:, i, :]) for i, embed in enumerate(self.codebook_embed)
        ]
        # Concatenate and project
        concat = torch.cat(embeddings, dim=-1)  # [B, T, d_model]
        return concat

    def _encode_past_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode past tokens for context.

        Args:
            tokens: [B, n_codebooks, k]

        Returns:
            [B, k, d_model]
        """
        embeddings = [
            embed(tokens[:, i, :]) for i, embed in enumerate(self.codebook_embed)
        ]
        concat = torch.cat(embeddings, dim=-1)  # [B, k, d_model]
        return self.context_encoder(concat)

    def _embed_target_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed target tokens for teacher forcing.

        Args:
            tokens: [B, n_codebooks, T]

        Returns:
            [B, T, d_model]
        """
        embeddings = [
            embed(tokens[:, i, :]) for i, embed in enumerate(self.target_codebook_embed)
        ]
        # Concatenate across codebooks
        # Each codebook contributes d_model // n_codebooks dimensions
        return torch.cat(embeddings, dim=-1)  # [B, T, d_model]

    @torch.no_grad()
    def generate(
        self,
        voice_state: torch.Tensor,
        speaker_embed: torch.Tensor,
        text_features: torch.Tensor | None = None,
        source_tokens: torch.Tensor | None = None,
        past_tokens: torch.Tensor | None = None,
        mode: str = "tts",
        max_length: int | None = None,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            text_features: [B, L, d_text] text features (TTS mode).
            source_tokens: [B, n_codebooks, T] source tokens (VC mode).
            voice_state: [B, T, d_voice_state] voice state parameters.
            speaker_embed: [B, d_speaker] speaker embedding.
            past_tokens: [B, n_codebooks, k] context from previous blocks.
            mode: "tts" or "vc".
            max_length: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            tokens: [B, n_codebooks, T] generated tokens.
        """
        c = self.config
        B, T, _ = voice_state.shape

        if max_length is None:
            max_length = T
        assert max_length is not None

        # Initialize with padding
        generated = torch.zeros(
            B, c.n_codebooks, max_length, dtype=torch.long, device=voice_state.device
        )

        # Generate first codebook autoregressively
        for t in range(max_length):
            vs_t = voice_state[:, : t + 1, :]
            tgt_so_far = generated[:, :, : t + 1] if t > 0 else None

            output = self.forward(
                text_features=text_features,
                source_tokens=source_tokens[:, :, : t + 1]
                if source_tokens is not None
                else None,
                voice_state=vs_t,
                speaker_embed=speaker_embed,
                past_tokens=past_tokens,
                target_tokens=tgt_so_far,
                mode=mode,
            )

            logits = output["logits_ar"][:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled = torch.multinomial(probs, 1)
            token = top_k_indices.gather(-1, sampled).squeeze(-1)
            generated[:, 0, t] = token

        # Generate remaining codebooks in parallel (non-AR)
        # Pad target tokens for parallel decoding
        target_for_parallel = torch.zeros(
            B, c.n_codebooks, max_length, dtype=torch.long, device=voice_state.device
        )
        target_for_parallel[:, 0, :] = generated[:, 0, :]

        output = self.forward(
            text_features=text_features,
            source_tokens=source_tokens,
            voice_state=voice_state,
            speaker_embed=speaker_embed,
            past_tokens=past_tokens,
            target_tokens=target_for_parallel,
            mode=mode,
        )

        for i in range(c.n_codebooks - 1):
            logits = output["logits_parallel"][:, i, :, :] / temperature
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled = torch.multinomial(probs.view(B * max_length, -1), 1).view(
                B, max_length, 1
            )
            tokens = top_k_indices.gather(-1, sampled).squeeze(-1)
            generated[:, i + 1, :] = tokens

        return generated


def uclm_loss(
    logits_ar: torch.Tensor,
    logits_parallel: torch.Tensor,
    target_tokens: torch.Tensor,
    pad_id: int = 0,
) -> dict[str, torch.Tensor]:
    """Compute UCLM training loss.

    Args:
        logits_ar: [B, T, vocab] first codebook logits.
        logits_parallel: [B, n_cb-1, T, vocab] remaining codebook logits.
        target_tokens: [B, n_cb, T] target tokens.
        pad_id: Padding token ID.

    Returns:
        Dict with loss values.
    """
    B, n_cb, T = target_tokens.shape

    # AR loss (first codebook)
    loss_ar = F.cross_entropy(
        logits_ar.reshape(-1, logits_ar.size(-1)),
        target_tokens[:, 0, :].reshape(-1),
        ignore_index=pad_id,
    )

    # Parallel loss (remaining codebooks)
    loss_parallel = 0.0
    for i in range(n_cb - 1):
        loss_parallel += F.cross_entropy(
            logits_parallel[:, i].reshape(-1, logits_parallel.size(-1)),
            target_tokens[:, i + 1, :].reshape(-1),
            ignore_index=pad_id,
        )
    loss_parallel /= n_cb - 1

    total_loss = loss_ar + loss_parallel

    return {
        "loss": total_loss,
        "loss_ar": loss_ar,
        "loss_parallel": loss_parallel,
    }
