"""Disentangled UCLM model combining VC/TTS encoders with dual-stream transformer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.types import PointerState

from .text_encoder import TextEncoder
from .uclm import VCEncoder, VoiceStateEncoder
from .uclm_transformer import CodecTransformer


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
        self.boundary_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (advance_logit, progress_delta, boundary_confidence), each [B, T, 1]."""
        return self.advance_proj(x), self.progress_proj(x), self.boundary_proj(x)


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
    """Flow-Matching based prosody predictor.

    Training: learns a velocity field v(x_t, t, cond) via conditional flow matching.
        Sample t~U(0,1), compute x_t = (1-t)*noise + t*target, predict velocity,
        loss = MSE(v_predicted, target - noise).
    Inference: starts from Gaussian noise and integrates the learned ODE with
        Euler steps to produce a prosody latent.
    """

    def __init__(self, d_model: int = 512, d_prosody: int = 128, n_ode_steps: int = 4):
        super().__init__()
        self.d_prosody = d_prosody
        self.n_ode_steps = n_ode_steps

        # Velocity network: predicts v(x_t, t, cond) -> d_prosody
        # Input: concatenation of x_t (d_prosody) and conditioning h (d_model)
        # The timestep t is broadcast-added after a small embedding.
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )
        self.velocity_net = nn.Sequential(
            nn.Linear(d_prosody + d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_prosody),
        )

        self.context_proj = nn.Linear(d_model, d_model)

    def _build_condition(
        self,
        phoneme_features: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build conditioning vector h from text, dialogue context, and speaker embed.

        Returns:
            h: [B, d_model]
        """
        h = phoneme_features.mean(dim=1)  # [B, d_model]

        if dialogue_context is not None:
            # Handle both 2D [B, D] and 3D [B, C, D] dialogue context
            if dialogue_context.ndim == 3:
                dialogue_context = dialogue_context.mean(dim=1)
            h = h + self.context_proj(
                F.pad(dialogue_context, (0, h.shape[-1] - dialogue_context.shape[-1]))
                if dialogue_context.shape[-1] < h.shape[-1]
                else dialogue_context[:, :h.shape[-1]]
            )

        if speaker_embed is not None:
            if speaker_embed.shape[-1] != h.shape[-1]:
                speaker_embed = F.pad(speaker_embed, (0, h.shape[-1] - speaker_embed.shape[-1]))
            h = h + speaker_embed

        return h

    def _predict_velocity(
        self, x_t: torch.Tensor, t: torch.Tensor, h: torch.Tensor
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, h).

        Args:
            x_t: [B, d_prosody] noisy sample at time t.
            t: [B, 1] timestep in [0, 1].
            h: [B, d_model] conditioning vector.
        Returns:
            v: [B, d_prosody] predicted velocity.
        """
        t_emb = self.time_embed(t)  # [B, d_model]
        h_cond = h + t_emb  # [B, d_model]
        inp = torch.cat([x_t, h_cond], dim=-1)  # [B, d_prosody + d_model]
        return self.velocity_net(inp)

    def forward(
        self,
        phoneme_features: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict prosody latent from text features.

        During training, returns a dummy sample (use ``flow_matching_loss`` for
        the actual training objective). During inference, integrates the learned
        ODE from noise to produce a prosody latent.

        Args:
            phoneme_features: [B, L, d_model] text encoder output.
            dialogue_context: [B, D_ctx] or [B, C_ctx, D_ctx] optional dialogue context.
            speaker_embed: [B, d_model] optional speaker embedding.
        Returns:
            prosody_latent: [B, d_prosody] predicted prosody.
        """
        h = self._build_condition(phoneme_features, dialogue_context, speaker_embed)
        B = h.shape[0]

        if self.training:
            # During training forward, return zeros; actual loss computed via
            # flow_matching_loss() which is called separately.
            return torch.zeros(B, self.d_prosody, device=h.device)

        # Inference: Euler ODE integration from noise (t=0) to signal (t=1)
        x = torch.randn(B, self.d_prosody, device=h.device)
        dt = 1.0 / self.n_ode_steps
        for i in range(self.n_ode_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=h.device)
            v = self._predict_velocity(x, t, h)
            x = x + dt * v
        return x

    def flow_matching_loss(
        self,
        phoneme_features: torch.Tensor,
        target_prosody: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute conditional flow matching loss.

        Args:
            phoneme_features: [B, L, d_model] text encoder output.
            target_prosody: [B, d_prosody] ground-truth prosody latent.
            dialogue_context: [B, D_ctx] or [B, C_ctx, D_ctx] optional dialogue context.
            speaker_embed: [B, d_model] optional speaker embedding.
        Returns:
            Scalar MSE loss between predicted and true velocity.
        """
        h = self._build_condition(phoneme_features, dialogue_context, speaker_embed)
        B = h.shape[0]

        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(B, 1, device=h.device)

        # Sample noise
        noise = torch.randn_like(target_prosody)

        # Interpolate: x_t = (1 - t) * noise + t * target
        x_t = (1.0 - t) * noise + t * target_prosody

        # True velocity = target - noise (straight-line OT path)
        v_target = target_prosody - noise

        # Predict velocity
        v_pred = self._predict_velocity(x_t, t, h)

        return F.mse_loss(v_pred, v_target)


class DialogueContextProjector(nn.Module):
    """Projects dialogue context, acting intent, and prosody latent into model space.

    Each input is optional. When provided, it is projected to d_model and added
    to the content features.  When absent, a zero contribution is used.

    Inputs (all optional):
        dialogue_context: [B, D_ctx] or [B, C_ctx, D_ctx] — scene/dialogue embedding.
        acting_intent: [B, D_act] — utterance-level acting intent vector.
        prosody_latent: [B, D_pro] or [B, T, D_pro] — local prosody planning signal.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_dialogue: int = 256,
        d_acting: int = 64,
        d_prosody: int = 128,
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
            # Handle both 2D [B, D_ctx] and 3D [B, C_ctx, D_ctx]
            if dialogue_context.ndim == 3:
                dialogue_context = dialogue_context.mean(dim=1)
            # [B, D_ctx] -> [B, 1, d_model] broadcast over T
            content_features = content_features + self.dialogue_proj(dialogue_context).unsqueeze(1)
        if acting_intent is not None:
            # [B, D_act] -> [B, 1, d_model] broadcast over T
            content_features = content_features + self.acting_proj(acting_intent).unsqueeze(1)
        if prosody_latent is not None:
            # Handle both 2D [B, D_pro] and 3D [B, T, D_pro]
            if prosody_latent.ndim == 2:
                # Utterance-global: [B, D_pro] -> [B, 1, d_model] broadcast over T
                content_features = content_features + self.prosody_proj(prosody_latent).unsqueeze(1)
            else:
                # Time-local: [B, T_pro, D_pro] -> [B, T, d_model]
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
        d_prosody: int = 128,
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
            vocab_size=vocab_size, d_model=d_model, n_layers=6
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

        # Delta voice state projection (v3).
        self.delta_voice_state_proj = nn.Linear(d_explicit, d_model)

        # CFG scale embedding for distilled mode (v3).
        # Projects to d_speaker so it can be added to speaker_embed directly.
        self.cfg_scale_embed = nn.Linear(1, d_speaker)

    @staticmethod
    def apply_cfg_unconditional_mask(
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        speaker_embed: torch.Tensor,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        prompt_kv_cache: torch.Tensor | None = None,
    ) -> dict:
        """Apply CFG unconditional mask: zero all conditioning fields.

        Preserves phoneme_ids, language_ids, acoustic_history, and pointer_state.
        Returns a dict of zeroed tensors ready for the unconditional forward pass.
        """
        return {
            "explicit_state": torch.zeros_like(explicit_state),
            "ssl_state": torch.zeros_like(ssl_state),
            "speaker_embed": torch.zeros_like(speaker_embed),
            "dialogue_context": torch.zeros_like(dialogue_context) if dialogue_context is not None else None,
            "acting_intent": torch.zeros_like(acting_intent) if acting_intent is not None else None,
            "prosody_latent": torch.zeros_like(prosody_latent) if prosody_latent is not None else None,
            "delta_voice_state": torch.zeros_like(delta_voice_state) if delta_voice_state is not None else None,
            "prompt_kv_cache": torch.zeros_like(prompt_kv_cache) if prompt_kv_cache is not None else None,
        }

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
            (refined_speaker_embed [B, d_model], prompt_summary_tokens [B, n_summary, d_model])
        """
        refined_embed, prompt_feats = self.speaker_prompt_encoder(prompt_codec_tokens, speaker_embed)
        
        # Condense full prompt features into summary tokens for 10ms efficiency
        summary_tokens = self.uclm_core.prompt_resampler(prompt_feats)
        
        return refined_embed, summary_tokens

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
            dialogue_context: [B, D_ctx] or [B, C_ctx, D_ctx] optional dialogue context.
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

        # Shift target_a and target_b to use as context A_{t-1}, B_{t-1}.
        # VC uses source_a_t as content, but still needs self-history for Stream A continuity.
        a_ctx = torch.zeros_like(source_a_t)
        a_ctx[:, :, 1:] = source_a_t[:, :, :-1]
        a_ctx = a_ctx.clamp_min(0)

        b_ctx = torch.zeros_like(target_b)
        b_ctx[:, :, 1:] = target_b[:, :, :-1]
        b_ctx = b_ctx.clamp_min(0)

        logits_a, logits_b, x_out = self.uclm_core.forward_no_cache(
            queries=content_features,
            memory=content_features, # VC cross-attends to its own latent sequence
            a_ctx=a_ctx,
            b_ctx=b_ctx,
            state_cond=state_cond,
            speaker_embed=speaker_embed,
            cfg_scale=cfg_scale,
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
        target_a: torch.Tensor | None = None,
        target_b: torch.Tensor | None = None,
        target_length: int | None = None,
        f0_condition: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        prompt_kv_cache: torch.Tensor | None = None,
        acoustic_history: tuple[torch.Tensor, torch.Tensor] | None = None,
        bootstrap_alignment: dict | None = None,
        text_suprasegmentals: torch.Tensor | None = None,
    ) -> dict:
        """Pointer-based TTS forward pass (UCLM v3).

        Args:
            phoneme_ids: [B, L] phoneme token ids.
            language_ids: [B, L] language token ids.
            pointer_state: optional :class:`PointerState` (used at inference).
            speaker_embed: [B, d_speaker] speaker embedding.
            explicit_state: [B, T, d_explicit] explicit voice state features.
            ssl_state: [B, T, d_ssl] SSL voice state features.
            target_a: [B, n_codebooks, T] ground-truth acoustic tokens.
            target_b: [B, n_slots, T] ground-truth control tokens.
            target_length: number of acoustic frames. Optional; derived from
                target_b when None (teacher-forced training). Runtime must not
                require a precomputed target length.
            f0_condition: optional [B, T, 2] F0 conditioning tensor.
            cfg_scale: classifier-free guidance scale.
            dialogue_context: optional [B, D_ctx] or [B, C_ctx, D_ctx] scene/dialogue embedding.
            acting_intent: optional [B, D_act] utterance-level acting intent.
            prosody_latent: optional [B, D_pro] or [B, T, D_pro] local prosody planning.
            delta_voice_state: optional [B, T, d_explicit] delta voice state.
            prompt_kv_cache: optional [B, T_prompt, d_model] cached prompt features
                from encode_speaker_prompt, prepended to content for cross-attention.
            acoustic_history: optional tuple of (a_ctx, b_ctx) for causal decoding context.
            bootstrap_alignment: optional dict with 'phoneme_indices' for hard bootstrap.
            text_suprasegmentals: optional [B, L, d_suprasegmental] companion per-unit features.

        Returns:
            dict with keys ``logits_a``, ``logits_b``, ``advance_logit``,
            ``progress_delta``, ``boundary_confidence``, ``hidden_states``,
            ``adv_logits``, and ``next_pointer_state``.
        """
        # Derive target_length from targets if not explicitly provided
        if target_length is None:
            if target_b is not None:
                target_length = target_b.shape[-1]
            elif target_a is not None:
                target_length = target_a.shape[-1]
            else:
                raise ValueError(
                    "target_length must be provided when target_a and target_b are None"
                )

        # 1. Encode phonemes
        phoneme_lens = (phoneme_ids != 0).sum(dim=1)
        phoneme_features = self.text_encoder(
            phoneme_ids, language_ids, phoneme_lens,
            text_suprasegmentals=text_suprasegmentals,
        )
        phoneme_features = phoneme_features.transpose(1, 2)

        B, L, D = phoneme_features.shape

        # 2. Build base frame representation (Conditioning on text units)
        if pointer_state is not None and pointer_state.is_hard_bootstrapped:
            # Stage 2 Annealing - Hard Phase: Use external alignment as truth
            if bootstrap_alignment is None:
                raise ValueError("bootstrap_alignment required when is_hard_bootstrapped=True")
            # bootstrap_alignment is expected to be [B, T] mapping frame -> phoneme index
            phoneme_indices = bootstrap_alignment["phoneme_indices"]
            content_features = phoneme_features[torch.arange(B).unsqueeze(1), phoneme_indices, :]
        elif pointer_state is not None:
            # Inference or Latent phase: Use pointer_state.text_index
            # In streaming, text_index is usually [B], so we unsqueeze
            idx = pointer_state.text_index
            if idx.ndim == 1:
                idx = idx.unsqueeze(1).expand(-1, target_length)
            content_features = phoneme_features[torch.arange(B).unsqueeze(1), idx, :]
        else:
            # Teacher-forced training / Legacy fallback
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

        # 4b. Apply delta voice state if provided
        if delta_voice_state is not None:
            delta_proj = self.delta_voice_state_proj(delta_voice_state)
            state_cond = state_cond + delta_proj

        # 5. Shift targets for causal history context (A_{t-1}, B_{t-1})
        if acoustic_history is not None:
            a_ctx, b_ctx = acoustic_history
        elif target_a is not None and target_b is not None:
            a_ctx = torch.zeros_like(target_a)
            a_ctx[:, :, 1:] = target_a[:, :, :-1]
            a_ctx = a_ctx.clamp_min(0)

            b_ctx = torch.zeros_like(target_b)
            b_ctx[:, :, 1:] = target_b[:, :, :-1]
            b_ctx = b_ctx.clamp_min(0)
        else:
            # Inference mode: no targets, use zero context
            a_ctx = torch.zeros(
                B, self.n_codebooks, target_length,
                dtype=torch.long, device=phoneme_features.device,
            )
            b_ctx = torch.zeros(
                B, 4, target_length,
                dtype=torch.long, device=phoneme_features.device,
            )

        # 6. Run through the codec transformer (now with Stream A history + Cross-Attention)
        # Note: we pass phoneme_features (L tokens) as memory for global cross-attention,
        # while content_features (T frames) acts as the query/base.
        logits_a, logits_b, x_out = self.uclm_core.forward_no_cache(
            queries=content_features,
            memory=phoneme_features,
            a_ctx=a_ctx,
            b_ctx=b_ctx,
            state_cond=state_cond,
            speaker_embed=speaker_embed,
            prompt_tokens=prompt_kv_cache, # Pass prompt features for condensation
            cfg_scale=cfg_scale,
            f0_condition=f0_condition,
        )

        # 7. Pointer head
        advance_logit, progress_delta, boundary_confidence = self.pointer_head(x_out)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "advance_logit": advance_logit,
            "progress_delta": progress_delta,
            "boundary_confidence": boundary_confidence,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
            "hidden_states": x_out,
            "pointer_logits": advance_logit,  # backward-compat alias
            "next_pointer_state": None,  # populated at inference time
        }

    def forward_tts_distilled_cfg(
        self,
        phoneme_ids: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embed: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        cfg_scale: float,
        target_length: int | None = None,
        f0_condition: torch.Tensor | None = None,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        text_suprasegmentals: torch.Tensor | None = None,
    ) -> dict:
        """Single-pass forward with cfg_scale injected for distillation training."""
        # Project cfg_scale to model dimension and add to global speaker/style conditioning
        cfg_tensor = torch.tensor([[cfg_scale]], device=speaker_embed.device, dtype=torch.float32)
        cfg_emb = self.cfg_scale_embed(cfg_tensor)
        
        # Add cfg_emb to speaker_embed for distillation (simple injection hack)
        speaker_embed_with_cfg = speaker_embed + cfg_emb.squeeze(1)
        
        return self.forward_tts_pointer(
            phoneme_ids=phoneme_ids,
            language_ids=language_ids,
            pointer_state=None,  # distillation uses teacher-forced alignment
            speaker_embed=speaker_embed_with_cfg,
            explicit_state=explicit_state,
            ssl_state=ssl_state,
            target_a=target_a,
            target_b=target_b,
            target_length=target_length,
            f0_condition=f0_condition,
            cfg_scale=1.0, # scale is already baked into speaker_embed_with_cfg
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_latent,
            delta_voice_state=delta_voice_state,
        )

    @staticmethod
    def cfg_distillation_loss(
        teacher_logits_a: torch.Tensor,
        teacher_logits_b: torch.Tensor,
        student_logits_a: torch.Tensor,
        student_logits_b: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Kullback-Leibler divergence for CFG distillation."""
        def _kl(t, s):
            p = F.softmax(t / temperature, dim=-1)
            return F.kl_div(
                F.log_softmax(s / temperature, dim=-1),
                p,
                reduction="batchmean"
            ) * (temperature ** 2)
        
        loss_a = _kl(teacher_logits_a, student_logits_a)
        loss_b = _kl(teacher_logits_b, student_logits_b)
        return (loss_a + loss_b) * 0.5

    def forward_streaming(
        self,
        queries: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        a_ctx: torch.Tensor | None = None,
        b_ctx: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
        state_cond: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
        f0_condition: torch.Tensor | None = None,
        prompt_summary_tokens: torch.Tensor | None = None,
        content_features: torch.Tensor | None = None,
    ) -> dict:
        """Forward pass for streaming inference with KV cache list.

        Args:
            queries: [B, T_q, D] current frame base features.
                Also accepts ``content_features`` as a backward-compatible alias.
            memory: [B, L_mem, D] full phoneme sequence for global cross-attention.
                When None, queries are used as both query and memory.
            a_ctx: [B, n_codebooks, T_q] previous acoustic tokens.
            b_ctx: [B, n_slots, T_q] previous control tokens.
            speaker_embed: [B, d_speaker] speaker embedding.
            state_cond: [B, T_q, D] voice state conditioning.
            cfg_scale: classifier-free guidance scale.
            kv_caches: optional KV caches from previous step.
            dialogue_context: optional [B, D_ctx] or [B, C_ctx, D_ctx] scene/dialogue embedding.
            acting_intent: optional [B, D_act] acting intent vector.
            prosody_latent: optional [B, D_pro] or [B, T_q, D_pro] local prosody planning.
            f0_condition: optional [B, T_q, 2] F0 conditioning.
            prompt_summary_tokens: [B, n_summary, D] pre-condensed prompt features.
            content_features: backward-compatible alias for queries.
        """
        # Backward-compatible alias: content_features -> queries
        if queries is None and content_features is not None:
            queries = content_features
        if queries is None:
            raise ValueError("Either 'queries' or 'content_features' must be provided")

        # Default memory to queries if not provided
        if memory is None:
            memory = queries

        # Apply dialogue/acting/prosody conditioning
        queries = self.context_projector(
            queries,
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_latent,
        )

        cfg_tensor = torch.tensor([cfg_scale], device=queries.device)
        logits_a, logits_b, next_kv_caches, x_out = self.uclm_core(
            queries=queries,
            memory=memory,
            a_ctx=a_ctx,
            b_ctx=b_ctx,
            speaker_embed=speaker_embed,
            state_cond=state_cond,
            prompt_summary_tokens=prompt_summary_tokens,
            cfg_scale=cfg_tensor,
            kv_caches=kv_caches,
            f0_condition=f0_condition,
        )

        # Pointer head for streaming pointer-driven progression
        advance_logit, progress_delta, boundary_confidence = self.pointer_head(x_out)

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "advance_logit": advance_logit,
            "progress_delta": progress_delta,
            "boundary_confidence": boundary_confidence,
            "kv_cache_out": next_kv_caches,
            "hidden_states": x_out,
        }
