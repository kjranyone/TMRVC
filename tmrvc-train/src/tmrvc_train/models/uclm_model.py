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
    """The complete SOTA Disentangled UCLM model tying all components together."""

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
        num_speakers: int = 1000,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks
        self.rvq_vocab_size = rvq_vocab_size
        self.control_vocab_size = control_vocab_size

        self.vc_encoder = VCEncoder(
            n_codebooks=n_codebooks,
            vocab_size=rvq_vocab_size,
            d_model=d_model,
            vq_bins=vq_bins,
        )

        self.text_encoder = TextEncoder(
            vocab_size=vocab_size, d_model=d_model, n_layers=4
        )
        self.duration_predictor = DurationPredictor(d_model=d_model)
        self.feature_expander = TextFeatureExpander(d_model=d_model)

        self.voice_state_enc = VoiceStateEncoder(
            d_explicit=d_explicit, d_ssl=d_ssl, d_model=d_model, num_speakers=num_speakers
        )

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
        target_b: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        speaker_embed: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        f0_condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> dict:
        content_features, vq_loss = self.vc_encoder(source_a_t, source_mask=source_mask)

        v_out = self.voice_state_enc(explicit_state, ssl_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

        if f0_condition is not None:
            content_features = content_features + self.f0_proj(f0_condition)

        # Shift target_b to use as context B_{t-1}. target_b can include -1
        # padding for CE ignore_index, but embeddings require non-negative ids.
        B, n_slots, T = target_b.shape
        b_ctx = torch.zeros_like(target_b)
        b_ctx[:, :, 1:] = target_b[:, :, :-1]
        b_ctx = b_ctx.clamp_min(0)

        logits_a, logits_b = self.uclm_core.forward_no_cache(
            content_features, b_ctx, state_cond, speaker_embed, cfg_scale
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "vq_loss": vq_loss,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
        }

    def forward_tts(
        self,
        phonemes: torch.Tensor,
        phoneme_lens: torch.Tensor,
        language_ids: torch.Tensor,
        target_b: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        speaker_embed: torch.Tensor,
        durations: Optional[torch.Tensor] = None,
        f0_condition: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> dict:
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
        v_out = self.voice_state_enc(explicit_state, ssl_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

        # Shift target_b to use as context B_{t-1}. target_b can include -1
        # padding for CE ignore_index, but embeddings require non-negative ids.
        B, n_slots, T = target_b.shape
        b_ctx = torch.zeros_like(target_b)
        b_ctx[:, :, 1:] = target_b[:, :, :-1]
        b_ctx = b_ctx.clamp_min(0)

        logits_a, logits_b = self.uclm_core.forward_no_cache(
            content_features, b_ctx, state_cond, speaker_embed, cfg_scale
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "log_durations": log_durations,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
        }

    def forward_streaming(
        self,
        content_features: torch.Tensor,
        b_ctx: torch.Tensor,
        speaker_embed: torch.Tensor,
        state_cond: torch.Tensor,
        cfg_scale: float = 1.0,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        """Forward pass for streaming inference with KV cache list."""
        cfg_tensor = torch.tensor([cfg_scale], device=content_features.device)
        logits_a, logits_b, next_kv_caches = self.uclm_core(
            content_features,
            b_ctx,
            speaker_embed,
            state_cond,
            cfg_tensor,
            kv_caches,
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "kv_cache_out": next_kv_caches,
        }
