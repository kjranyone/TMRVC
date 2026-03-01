"""Disentangled UCLM model combining VC/TTS encoders with dual-stream transformer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .text_encoder import TextEncoder
from .text_features import TextFeatureExpander
from .duration_predictor import DurationPredictor
from .uclm import VCEncoder, VoiceStateEncoder
from .uclm_transformer import CodecTransformer


class DisentangledUCLM(nn.Module):
    """The complete SOTA Disentangled UCLM model tying all components together.

    Architecture:
        - VCEncoder: Information bottleneck to extract content from source audio
        - TextEncoder: Encode phonemes for TTS mode
        - DurationPredictor: Predict phoneme durations for TTS
        - TextFeatureExpander: Expand phoneme features to frame-level
        - VoiceStateEncoder: Encode explicit + SSL voice state parameters
        - CodecTransformer: Dual-stream token prediction (A_t, B_t)

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        rvq_vocab_size: Vocabulary size for acoustic tokens.
        n_codebooks: Number of RVQ codebooks.
        control_vocab_size: Vocabulary size for control tokens.
        d_explicit: Explicit voice state dimension.
        d_ssl: SSL latent dimension.
        d_speaker: Speaker embedding dimension.
        vq_bins: VQ bottleneck codebook size.
        vocab_size: Text vocabulary size.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        rvq_vocab_size: int = 1024,
        n_codebooks: int = 8,
        control_vocab_size: int = 64,
        d_explicit: int = 8,
        d_ssl: int = 128,
        d_speaker: int = 192,
        vq_bins: int = 128,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.rvq_vocab_size = rvq_vocab_size
        self.control_vocab_size = control_vocab_size

        self.vc_encoder = VCEncoder(n_codebooks, rvq_vocab_size, d_model, vq_bins)

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, d_model=d_model, n_layers=4
        )
        self.duration_predictor = DurationPredictor(d_model=d_model)
        self.feature_expander = TextFeatureExpander(d_model=d_model)

        self.voice_state_enc = VoiceStateEncoder(d_explicit, d_ssl, d_model)

        self.f0_proj = nn.Linear(2, d_model)  # F0 condition: [norm_f0, shift]

        self.uclm_core = CodecTransformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            rvq_vocab_size=rvq_vocab_size,
            n_codebooks=n_codebooks,
            control_vocab_size=control_vocab_size,
            d_speaker=d_speaker,
        )

    def forward_vc(
        self,
        source_a_t: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        speaker_embed: torch.Tensor,
        f0_condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> dict:
        """Forward pass for Voice Conversion mode.

        Args:
            source_a_t: [B, 8, T] source acoustic tokens
            explicit_state: [B, T, 8] explicit voice parameters
            ssl_state: [B, T, 128] SSL latent features
            speaker_embed: [B, 192] speaker embedding
            f0_condition: [B, T, 2] optional F0 conditioning
            cfg_scale: CFG amplification scale

        Returns:
            Dict with logits_a, logits_b, vq_loss
        """
        content_features, vq_loss = self.vc_encoder(source_a_t)

        state_out = self.voice_state_enc(explicit_state, ssl_state)
        if isinstance(state_out, tuple):
            state_cond, adv_logits = state_out
        else:
            state_cond, adv_logits = state_out, None

        if f0_condition is not None:
            content_features = content_features + self.f0_proj(f0_condition)

        logits_a, logits_b = self.uclm_core.forward_no_cache(
            content_features, state_cond, speaker_embed, cfg_scale
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "vq_loss": vq_loss,
            "adv_logits": adv_logits,
        }

    def forward_tts(
        self,
        phonemes: torch.Tensor,
        phoneme_lens: torch.Tensor,
        language_ids: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        speaker_embed: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        f0_condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> dict:
        """Forward pass for Text-to-Speech mode.

        Args:
            phonemes: [B, L] phoneme IDs
            phoneme_lens: [B] phoneme lengths
            language_ids: [B] language IDs
            explicit_state: [B, T, 8] explicit voice parameters
            ssl_state: [B, T, 128] SSL latent features
            speaker_embed: [B, 192] speaker embedding
            durations: [B, L] ground truth durations (for training)
            f0_condition: [B, T, 2] optional F0 conditioning
            cfg_scale: CFG amplification scale

        Returns:
            Dict with logits_a, logits_b, log_durations, adv_logits
        """
        # 1. Encode phonemes
        phoneme_features = self.text_encoder(phonemes, language_ids, phoneme_lens)
        phoneme_features = phoneme_features.transpose(1, 2)  # [B, L, d_model]

        # 2. Predict durations
        mask = (
            torch.arange(phonemes.shape[1], device=phonemes.device).unsqueeze(0)
            >= phoneme_lens.unsqueeze(1)
        )
        log_durations = self.duration_predictor(phoneme_features, mask)

        # 3. Expand to frame-level
        if durations is None:
            # Inference: use predicted durations
            durations = torch.round(torch.exp(log_durations) - 1.0).long()
            durations = torch.clamp(durations, min=1)

        target_length = explicit_state.shape[1]
        content_features = self.feature_expander(
            phoneme_features, durations, target_length
        )

        # 4. Apply F0 if provided
        if f0_condition is not None:
            content_features = content_features + self.f0_proj(f0_condition)

        # 5. Predict tokens
        state_out = self.voice_state_enc(explicit_state, ssl_state)
        if isinstance(state_out, tuple):
            state_cond, adv_logits = state_out
        else:
            state_cond, adv_logits = state_out, None

        logits_a, logits_b = self.uclm_core.forward_no_cache(
            content_features, state_cond, speaker_embed, cfg_scale
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "log_durations": log_durations,
            "adv_logits": adv_logits,
        }

    def forward_streaming(
        self,
        content_features: torch.Tensor,
        state_cond: torch.Tensor,
        speaker_embed: torch.Tensor,
        cfg_scale: float = 1.0,
        kv_cache_in: Optional[torch.Tensor] = None,
        max_seq_len: int = 200,
    ) -> dict:
        """Forward pass for streaming inference with KV cache.

        Args:
            content_features: [B, T, d_model] content features
            state_cond: [B, T, d_model] voice state condition
            speaker_embed: [B, 192] speaker embedding
            cfg_scale: CFG scale
            kv_cache_in: [B, kv_cache_size] input KV cache
            max_seq_len: Maximum sequence length for KV cache

        Returns:
            Dict with logits_a, logits_b, kv_cache_out
        """
        logits_a, logits_b, kv_cache_out = self.uclm_core(
            content_features,
            state_cond,
            speaker_embed,
            cfg_scale,
            kv_cache_in,
            max_seq_len,
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "kv_cache_out": kv_cache_out,
        }
