"""Disentangled UCLM model combining VC/TTS encoders with dual-stream transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .text_encoder import TextEncoder
from .uclm import VCEncoder, VoiceStateEncoder
from .uclm_transformer import CodecTransformer


@dataclass
class PointerState:
    """Tracks the current phoneme pointer position for streaming TTS.

    Attributes:
        text_index: [B] current phoneme index (integer).
        progress: [B] fractional progress within the current phoneme (0-1).
        boundary_confidence: confidence score for the last boundary decision.
        stall_frames: number of consecutive frames without pointer advance.
    """

    text_index: torch.Tensor  # [B]
    progress: torch.Tensor  # [B]
    finished: bool = False
    boundary_confidence: float = 0.0
    stall_frames: int = 0
    max_frames_per_unit: int = 50
    frames_on_current_unit: int = 0
    skip_protection_threshold: float = 0.3
    forced_advance_count: int = 0
    skip_protection_count: int = 0

    def clone(self) -> "PointerState":
        return PointerState(
            text_index=self.text_index.clone(),
            progress=self.progress.clone(),
            finished=self.finished,
            boundary_confidence=self.boundary_confidence,
            stall_frames=self.stall_frames,
            max_frames_per_unit=self.max_frames_per_unit,
            frames_on_current_unit=self.frames_on_current_unit,
            skip_protection_threshold=self.skip_protection_threshold,
            forced_advance_count=self.forced_advance_count,
            skip_protection_count=self.skip_protection_count,
        )

    def step_pointer(
        self,
        advance_prob: float,
        progress_delta: float,
        boundary_confidence: float = 0.0,
    ) -> bool:
        """Canonical pointer state-transition logic.

        Implements forced advance on stall, skip-protection against premature
        advance, and normal advance/hold behaviour.

        Returns:
            ``True`` if the pointer advanced to the next text unit.
        """
        # 1. Track time on the current unit
        self.frames_on_current_unit += 1

        # 2. Accumulate progress
        self.progress += progress_delta

        # 3. Forced advance when stuck too long on one unit
        if self.frames_on_current_unit >= self.max_frames_per_unit:
            self.text_index += 1
            self.progress = self.progress * 0 + 0.0  # keep tensor type if applicable
            self.frames_on_current_unit = 0
            self.stall_frames = 0
            self.forced_advance_count += 1
            return True

        # 4. High-confidence advance with skip-protection
        if advance_prob > 0.5 and self.progress >= 1.0:
            if boundary_confidence >= self.skip_protection_threshold:
                # Normal boundary advance
                self.text_index += 1
                self.progress = self.progress * 0 + 0.0
                self.frames_on_current_unit = 0
                self.stall_frames = 0
                return True
            else:
                # Skip-protection blocks the advance
                self.skip_protection_count += 1
                return False

        # 5. Advance on either signal alone
        if advance_prob > 0.5 or self.progress >= 1.0:
            self.text_index += 1
            self.progress = self.progress * 0 + 0.0
            self.frames_on_current_unit = 0
            self.stall_frames = 0
            return True

        # 6. Hold — no advance
        self.stall_frames += 1
        return False


class PointerHead(nn.Module):
    """Small head that predicts phoneme-pointer advance probability and progress.

    Given transformer hidden states of shape [B, T, d_model], outputs:
        advance_logit: [B, T, 1] - logit for advancing to the next phoneme.
        progress_delta: [B, T, 1] - predicted fractional progress within the
            current phoneme (0-1, after sigmoid).
    """

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.advance_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
        self.progress_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (advance_logit, progress_delta), each [B, T, 1]."""
        return self.advance_proj(x), self.progress_proj(x)


class SpeakerPromptEncoder(nn.Module):
    """Encodes short reference audio into speaker embedding and prompt KV cache.

    Supports few-shot speaker adaptation from 3-10 second reference clips.
    Designed to disentangle timbre (speaker identity) from prosody.
    """

    def __init__(self, d_model: int = 512, d_speaker: int = 192):
        super().__init__()
        # Project raw speaker embedding to model space
        self.speaker_proj = nn.Linear(d_speaker, d_model)
        # Prompt codec token embedding (reuses codec vocab)
        self.prompt_embed = nn.Embedding(1024, d_model)
        # Lightweight transformer to encode prompt context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=d_model * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.prompt_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_norm = nn.LayerNorm(d_model)
        # Timbre bottleneck: prevents prosody leakage from prompt
        self.timbre_bottleneck = nn.Sequential(
            nn.Linear(d_model, d_speaker),
            nn.Tanh(),
            nn.Linear(d_speaker, d_model),
        )

    def forward(
        self,
        prompt_codec_tokens: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode prompt tokens into speaker embedding and prompt KV features.

        Args:
            prompt_codec_tokens: [B, T_prompt, n_codebooks] codec tokens from reference audio.
            speaker_embed: [B, d_speaker] optional external speaker embedding.
        Returns:
            (refined_speaker_embed [B, d_model], prompt_features [B, T_prompt, d_model])
        """
        # Average across codebooks -> [B, T_prompt]
        avg_tokens = prompt_codec_tokens[:, :, 0].long()  # use first codebook
        x = self.prompt_embed(avg_tokens)  # [B, T_prompt, d_model]
        x = self.prompt_encoder(x)
        x = self.output_norm(x)

        # Extract timbre through bottleneck (blocks prosody leakage)
        timbre = self.timbre_bottleneck(x.mean(dim=1))  # [B, d_model]

        if speaker_embed is not None:
            # Fuse external speaker embedding with prompt-derived timbre
            timbre = timbre + self.speaker_proj(speaker_embed)

        return timbre, x


class ProsodyPredictor(nn.Module):
    """Predicts local prosody latent from text and context during inference.

    Training: target latent extracted from reference audio (teacher forcing).
    Inference: generates prosody latent from text + context.
    """

    def __init__(self, d_model: int = 512, d_prosody: int = 64):
        super().__init__()
        self.d_prosody = d_prosody
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_prosody * 2),  # mu and log_var
        )
        self.context_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        phoneme_features: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict prosody latent from text features.

        Args:
            phoneme_features: [B, L, d_model] text encoder output.
            dialogue_context: [B, D_ctx] optional dialogue context.
            speaker_embed: [B, d_model] optional speaker embedding.
        Returns:
            prosody_latent: [B, d_prosody] predicted prosody.
        """
        # Pool text features
        h = phoneme_features.mean(dim=1)  # [B, d_model]

        if dialogue_context is not None:
            h = h + self.context_proj(
                F.pad(dialogue_context, (0, h.shape[-1] - dialogue_context.shape[-1]))
                if dialogue_context.shape[-1] < h.shape[-1]
                else dialogue_context[:, :h.shape[-1]]
            )

        if speaker_embed is not None:
            if speaker_embed.shape[-1] != h.shape[-1]:
                speaker_embed = F.pad(speaker_embed, (0, h.shape[-1] - speaker_embed.shape[-1]))
            h = h + speaker_embed

        params = self.encoder(h)  # [B, d_prosody * 2]
        mu, log_var = params.chunk(2, dim=-1)

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu


class DialogueContextProjector(nn.Module):
    """Projects dialogue context, acting intent, and prosody latent into model space.

    Each input is optional. When provided, it is projected to d_model and added
    to the content features.  When absent, a zero contribution is used.

    Inputs (all optional):
        dialogue_context: [B, D_ctx] — scene/dialogue embedding.
        acting_intent: [B, D_act] — utterance-level acting intent vector.
        prosody_latent: [B, T, D_pro] — local prosody planning signal.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_dialogue: int = 256,
        d_acting: int = 64,
        d_prosody: int = 64,
    ):
        super().__init__()
        self.dialogue_proj = nn.Linear(d_dialogue, d_model)
        self.acting_proj = nn.Linear(d_acting, d_model)
        self.prosody_proj = nn.Linear(d_prosody, d_model)

    def forward(
        self,
        content_features: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Add projected conditioning to content_features [B, T, D]."""
        if dialogue_context is not None:
            # [B, D_ctx] -> [B, 1, d_model] broadcast over T
            content_features = content_features + self.dialogue_proj(dialogue_context).unsqueeze(1)
        if acting_intent is not None:
            # [B, D_act] -> [B, 1, d_model] broadcast over T
            content_features = content_features + self.acting_proj(acting_intent).unsqueeze(1)
        if prosody_latent is not None:
            # [B, T, D_pro] -> [B, T, d_model]
            T_content = content_features.shape[1]
            T_pro = prosody_latent.shape[1]
            if T_pro != T_content:
                prosody_latent = F.interpolate(
                    prosody_latent.transpose(1, 2), size=T_content, mode="nearest"
                ).transpose(1, 2)
            content_features = content_features + self.prosody_proj(prosody_latent)
        return content_features


class DisentangledUCLM(nn.Module):
    """The complete SOTA Disentangled UCLM model tying all components together (v3)."""

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
        d_dialogue: int = 256,
        d_acting: int = 64,
        d_prosody: int = 64,
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

        # Pointer-based TTS head (primary v3 text progression mechanism).
        self.pointer_head = PointerHead(d_model=d_model)

        # Dialogue/acting conditioning projector (v3 expressive path).
        self.context_projector = DialogueContextProjector(
            d_model=d_model,
            d_dialogue=d_dialogue,
            d_acting=d_acting,
            d_prosody=d_prosody,
        )

        # Speaker prompt encoder for few-shot adaptation (v3).
        self.speaker_prompt_encoder = SpeakerPromptEncoder(d_model=d_model, d_speaker=d_speaker)

        # Prosody predictor for inference-time prosody generation (v3).
        self.prosody_predictor = ProsodyPredictor(d_model=d_model, d_prosody=d_prosody)

    def encode_speaker_prompt(
        self,
        prompt_codec_tokens: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode short reference audio for few-shot speaker adaptation.

        Args:
            prompt_codec_tokens: [B, T_prompt, n_codebooks] from reference audio.
            speaker_embed: [B, d_speaker] optional external speaker embedding.
        Returns:
            (refined_speaker_embed [B, d_model], prompt_features [B, T_prompt, d_model])
        """
        return self.speaker_prompt_encoder(prompt_codec_tokens, speaker_embed)

    def predict_prosody(
        self,
        phoneme_ids: torch.Tensor,
        language_ids: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict prosody latent from text and context.

        Args:
            phoneme_ids: [B, L] phoneme token ids.
            language_ids: [B, L] language token ids.
            dialogue_context: [B, D_ctx] optional dialogue context.
            speaker_embed: [B, d_model] optional refined speaker embedding.
        Returns:
            prosody_latent: [B, d_prosody]
        """
        phoneme_lens = (phoneme_ids != 0).sum(dim=1)
        phoneme_features = self.text_encoder(phoneme_ids, language_ids, phoneme_lens)
        phoneme_features = phoneme_features.transpose(1, 2)  # [B, L, D]
        return self.prosody_predictor(phoneme_features, dialogue_context, speaker_embed)

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

        # Shift target_b to use as context B_{t-1}.
        B, n_slots, T = target_b.shape
        b_ctx = torch.zeros_like(target_b)
        b_ctx[:, :, 1:] = target_b[:, :, :-1]
        b_ctx = b_ctx.clamp_min(0)

        logits_a, logits_b, x_out = self.uclm_core.forward_no_cache(
            content_features, b_ctx, state_cond, speaker_embed, cfg_scale
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "vq_loss": vq_loss,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
            "hidden_states": x_out,
        }

    def forward_tts(self, **kwargs) -> dict:
        """Legacy TTS forward (v2 compatibility wrapper).

        Delegates to forward_tts_pointer. Retained for backward compatibility
        with code that calls forward_tts directly.
        """
        return self.forward_tts_pointer(**kwargs)

    def forward_tts_pointer(
        self,
        phoneme_ids: torch.Tensor,
        language_ids: torch.Tensor,
        pointer_state: PointerState | None,
        speaker_embed: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        target_b: torch.Tensor,
        target_length: int,
        f0_condition: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
        prompt_kv_cache: torch.Tensor | None = None,
        acoustic_history: torch.Tensor | None = None,
    ) -> dict:
        """Pointer-based TTS forward pass (UCLM v3).

        Args:
            phoneme_ids: [B, L] phoneme token ids.
            language_ids: [B, L] language token ids.
            pointer_state: optional :class:`PointerState` (used at inference).
            speaker_embed: [B, d_speaker] speaker embedding.
            explicit_state: [B, T, d_explicit] explicit voice state features.
            ssl_state: [B, T, d_ssl] SSL voice state features.
            target_b: [B, n_codebooks, T] ground-truth codec tokens.
            target_length: number of acoustic frames to produce.
            f0_condition: optional [B, T, 2] F0 conditioning tensor.
            cfg_scale: classifier-free guidance scale.
            dialogue_context: optional [B, D_ctx] scene/dialogue embedding.
            acting_intent: optional [B, D_act] utterance-level acting intent.
            prosody_latent: optional [B, T, D_pro] local prosody planning.

        Returns:
            dict with keys ``logits_a``, ``logits_b``, ``pointer_logits``,
            ``progress_delta``, and ``adv_logits``.
        """
        # 1. Encode phonemes
        phoneme_lens = (phoneme_ids != 0).sum(dim=1)
        phoneme_features = self.text_encoder(phoneme_ids, language_ids, phoneme_lens)
        phoneme_features = phoneme_features.transpose(1, 2)

        B, L, D = phoneme_features.shape

        # 2. Uniformly distribute phonemes across target_length (initial signal)
        frame_indices = torch.arange(target_length, device=phoneme_features.device)
        phoneme_indices = (frame_indices.float() * L / target_length).long().clamp(max=L - 1)
        content_features = phoneme_features[:, phoneme_indices, :]

        # 3. Apply F0 conditioning
        if f0_condition is not None:
            content_features = content_features + self.f0_proj(f0_condition)

        # 3b. Apply dialogue/acting/prosody conditioning
        content_features = self.context_projector(
            content_features,
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_latent,
        )

        # 4. Voice state conditioning
        v_out = self.voice_state_enc(explicit_state, ssl_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

        # 5. Shift target_b for causal context
        b_ctx = torch.zeros_like(target_b)
        b_ctx[:, :, 1:] = target_b[:, :, :-1]
        b_ctx = b_ctx.clamp_min(0)

        # 6. Run through the codec transformer
        logits_a, logits_b, x_out = self.uclm_core.forward_no_cache(
            content_features, b_ctx, state_cond, speaker_embed, cfg_scale
        )

        # 7. Pointer head
        pointer_logits, progress_delta = self.pointer_head(x_out)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "pointer_logits": pointer_logits,
            "progress_delta": progress_delta,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
            "hidden_states": x_out,
            "advance_logit": pointer_logits,  # alias for pointer_logits
            "boundary_confidence": torch.zeros(B, target_length, 1, device=phoneme_features.device),
            "next_pointer_state": None,  # populated at inference time
        }

    def forward_streaming(
        self,
        content_features: torch.Tensor,
        b_ctx: torch.Tensor,
        speaker_embed: torch.Tensor,
        state_cond: torch.Tensor,
        cfg_scale: float = 1.0,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass for streaming inference with KV cache list.

        Args:
            content_features: [B, T, D] text/VC content features.
            b_ctx: [B, n_slots, T] previous control tokens.
            speaker_embed: [B, d_speaker] speaker embedding.
            state_cond: [B, T, D] voice state conditioning.
            cfg_scale: classifier-free guidance scale.
            kv_caches: optional KV caches from previous step.
            dialogue_context: optional [B, D_ctx] scene/dialogue embedding.
            acting_intent: optional [B, D_act] acting intent vector.
            prosody_latent: optional [B, T, D_pro] local prosody planning.
        """
        # Apply dialogue/acting/prosody conditioning
        content_features = self.context_projector(
            content_features,
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_latent,
        )

        cfg_tensor = torch.tensor([cfg_scale], device=content_features.device)
        logits_a, logits_b, next_kv_caches, x_out = self.uclm_core(
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
            "hidden_states": x_out,
        }
