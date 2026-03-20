"""Disentangled UCLM model combining VC/TTS encoders with dual-stream transformer."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_MODEL, UCLM_N_HEADS, UCLM_N_LAYERS
from tmrvc_core.types import PointerState

from .acting_latent import ActingLatentConditioner, ActingMacroProjector
from .text_encoder import TextEncoder
from .voice_state_encoder import VoiceStateEncoder
from .uclm_transformer import CodecTransformer


class VectorQuantizer(nn.Module):
    """Information Bottleneck (VQ) for VC Encoder.
    Strips speaker and style info by mapping continuous embeddings to discrete codes.
    """

    def __init__(self, n_bins: int, d_model: int, beta: float = 0.25):
        super().__init__()
        self.n_bins = n_bins
        self.d_model = d_model
        self.beta = beta

        self.embedding = nn.Embedding(n_bins, d_model)
        self.embedding.weight.data.uniform_(-1.0 / n_bins, 1.0 / n_bins)

    def forward(
        self, z: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, T, d_model]
            mask: [B, T] valid-frame mask (True=valid), optional
        Returns:
            z_q: [B, T, d_model] quantized vectors
            loss: VQ commitment loss
            indices: [B, T] quantization indices
        """
        # Flatten
        z_flattened = z.reshape(-1, self.d_model)

        # Distances from z to embeddings
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        )

        # Find closest embeddings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_bins, device=z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # Loss: commitment loss (pulling encoder output to embeddings) + codebook loss
        if mask is None:
            loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean(
                (z_q - z.detach()) ** 2
            )
        else:
            mask_f = mask.to(z.dtype).unsqueeze(-1)  # [B, T, 1]
            denom = (mask_f.sum() * self.d_model).clamp_min(1.0)
            loss_commit = (((z_q.detach() - z) ** 2) * mask_f).sum() / denom
            loss_codebook = (((z_q - z.detach()) ** 2) * mask_f).sum() / denom
            loss = loss_commit + self.beta * loss_codebook

        return z_q, loss, min_encoding_indices.view(z.shape[:-1])


class VCEncoder(nn.Module):
    """Encodes source A_t tokens and applies VQ bottleneck to remove style/speaker info."""

    def __init__(self, n_codebooks=8, vocab_size=1024, d_model=D_MODEL, vq_bins=128):
        super().__init__()
        # Each codebook gets d_model // n_codebooks dimensions to concat into d_model
        self.codebook_embeds = nn.ModuleList(
            [
                nn.Embedding(vocab_size, d_model // n_codebooks)
                for _ in range(n_codebooks)
            ]
        )

        # Causal convolution instead of full transformer to keep it light and causal
        self.source_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2)

        self.vq_bottleneck = VectorQuantizer(vq_bins, d_model)

    def forward(
        self, source_a_t: torch.Tensor, source_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source_a_t: [B, 8, T] from EnCodec
            source_mask: [B, T] valid-frame mask (True=valid), optional
        Returns:
            content_features: [B, T, d_model]
            vq_loss: scalar tensor
        """
        B, n_cb, T = source_a_t.shape

        # Embed and concatenate along feature dim
        embeds = []
        for i, emb_layer in enumerate(self.codebook_embeds):
            embeds.append(emb_layer(source_a_t[:, i, :]))  # [B, T, d_model//8]

        x = torch.cat(embeds, dim=-1)  # [B, T, d_model]

        # Causal conv [B, d_model, T] -> [B, T, d_model]
        x = x.transpose(1, 2)
        x = F.relu(self.source_conv(x))
        x = x[:, :, :-2]  # Remove padding to keep causal
        x = x.transpose(1, 2)

        # Apply Information Bottleneck
        content_features, vq_loss, _ = self.vq_bottleneck(x, mask=source_mask)

        return content_features, vq_loss


class PointerHead(nn.Module):
    """Small head that predicts phoneme-pointer advance probability and progress.

    Given transformer hidden states of shape [B, T, d_model], outputs:
        advance_logit: [B, T, 1] - logit for advancing to the next phoneme.
        progress_delta: [B, T, 1] - predicted fractional progress within the
            current phoneme (0-1, after sigmoid).
    """

    def __init__(self, d_model: int = D_MODEL):
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

    def __init__(self, d_model: int = D_MODEL, d_speaker: int = 192):
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
        # Vector Quantizer for fine-grained timbre (Worker 01 Task 6)
        self.vq = VectorQuantizer(n_bins=256, d_model=d_model)

    def forward(
        self,
        prompt_codec_tokens: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode prompt tokens into speaker embedding and prompt KV features.

        Args:
            prompt_codec_tokens: [B, T_prompt, n_codebooks] codec tokens from reference audio.
            speaker_embed: [B, d_speaker] optional external speaker embedding.
        Returns:
            refined_speaker_embed: [B, d_model]
            prompt_features: [B, T_prompt, d_model] quantized timbre features
            vq_loss: scalar commitment loss
            indices: [B, T_prompt] VQ indices
        """
        # Average across codebooks -> [B, T_prompt]
        avg_tokens = prompt_codec_tokens[:, :, 0].long()  # use first codebook
        x = self.prompt_embed(avg_tokens)  # [B, T_prompt, d_model]
        x = self.prompt_encoder(x)
        x = self.output_norm(x)

        # Apply VQ bottleneck to the sequence features (prosody stripping)
        x_q, vq_loss, indices = self.vq(x)

        # Extract timbre through bottleneck (blocks prosody leakage)
        timbre = self.timbre_bottleneck(x_q.mean(dim=1))  # [B, d_model]

        if speaker_embed is not None:
            # Fuse external speaker embedding with prompt-derived timbre
            timbre = timbre + self.speaker_proj(speaker_embed)

        return timbre, x_q, vq_loss, indices


class ProsodyPredictor(nn.Module):
    """Flow-Matching based prosody predictor.

    Training: learns a velocity field v(x_t, t, cond) via conditional flow matching.
        Sample t~U(0,1), compute x_t = (1-t)*noise + t*target, predict velocity,
        loss = MSE(v_predicted, target - noise).
    Inference: starts from Gaussian noise and integrates the learned ODE with
        Euler steps to produce a prosody latent.
    """

    def __init__(self, d_model: int = D_MODEL, d_prosody: int = 128, n_ode_steps: int = 4):
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
            **Deprecated in v4**: superseded by ActingLatentConditioner (24-D).
            Retained for backward compat with v3 checkpoints and datasets.
            When both acting_intent and acting_texture_latent are provided,
            the caller must suppress acting_intent (set to None).
        prosody_latent: [B, D_pro] or [B, T, D_pro] — local prosody planning signal.
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
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
    """The complete SOTA Disentangled UCLM model tying all components together."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = UCLM_N_HEADS,
        n_layers: int = UCLM_N_LAYERS,
        rvq_vocab_size: int = 1024,
        n_codebooks: int = 8,
        control_vocab_size: int = 64,
        d_voice_state_explicit: int = 12,
        d_voice_state_ssl: int = 128,
        d_speaker: int = 192,
        vq_bins: int = 128,
        vocab_size: int = 256,
        num_speakers: int = 1000,
        d_dialogue: int = 256,
        d_acting: int = 64,
        d_prosody: int = 128,
        acting_tag_vocab_size: int = 0,
        codec_condition: str = "A",
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
            vocab_size=vocab_size, d_model=d_model, n_layers=6,
            acting_tag_vocab_size=acting_tag_vocab_size,
        )

        self.voice_state_enc = VoiceStateEncoder(
            d_voice_state_explicit=d_voice_state_explicit, d_voice_state_ssl=d_voice_state_ssl, d_model=d_model, num_speakers=num_speakers
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

        # Pointer-based TTS head (primary text progression mechanism).
        self.pointer_head = PointerHead(d_model=d_model)

        # Dialogue/acting conditioning projector (expressive path).
        self.context_projector = DialogueContextProjector(
            d_model=d_model,
            d_dialogue=d_dialogue,
            d_acting=d_acting,
            d_prosody=d_prosody,
        )

        # Speaker prompt encoder for few-shot adaptation.
        self.speaker_prompt_encoder = SpeakerPromptEncoder(d_model=d_model, d_speaker=d_speaker)

        # Prosody predictor for inference-time prosody generation.
        self.prosody_predictor = ProsodyPredictor(d_model=d_model, d_prosody=d_prosody)

        # Delta voice state projection.
        self.delta_voice_state_proj = nn.Linear(d_voice_state_explicit, d_model)

        # CFG scale embedding for distilled mode.
        # Projects to d_model as an independent conditioning signal,
        # separate from speaker_embed (v4 conditioning separation).
        self.cfg_scale_embed = nn.Linear(1, d_model)

        # v4: Acting texture latent conditioner (24-D -> d_model)
        self.acting_latent_conditioner = ActingLatentConditioner()

        # v4: Acting macro projector (6-D user-facing -> 24-D latent)
        # Registered as a submodule so weights are checkpointed and loaded.
        self.acting_macro_proj = ActingMacroProjector()

        # v4: Dedicated physical prediction head from hidden states (d_model -> 12)
        self.physical_prediction_head = nn.Linear(d_model, 12)

        # v4 codec condition routing
        self.codec_condition = codec_condition

        # Condition-specific modules (v4 codec strategy)
        if codec_condition == "B":
            self.nar_head = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, (n_codebooks - 1) * rvq_vocab_size),
            )
        elif codec_condition == "C":
            self.delay_offset = n_codebooks  # delay pattern: CB_i delayed by i frames
        elif codec_condition == "D":
            self.single_cb_embed = nn.Embedding(8192, d_model)
            self.single_cb_head = nn.Linear(d_model, 8192)

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
        acting_texture_latent: torch.Tensor | None = None,
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
            # Use None (not zeros) so the conditioner MLP is fully bypassed.
            # zeros would still inject the MLP's bias vector, breaking the
            # unconditional semantics required by CFG.
            "acting_texture_latent": None,
        }

    def encode_speaker_prompt(
        self,
        prompt_codec_tokens: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode short reference audio for few-shot speaker adaptation.

        Args:
            prompt_codec_tokens: [B, T_prompt, n_codebooks] from reference audio.
            speaker_embed: [B, d_speaker] optional external speaker embedding.
        Returns:
            refined_speaker_embed: [B, d_model]
            prompt_summary_tokens: [B, n_summary, d_model]
            vq_loss: scalar commitment loss
            indices: [B, T_prompt] VQ indices
        """
        refined_embed, prompt_feats, vq_loss, indices = self.speaker_prompt_encoder(prompt_codec_tokens, speaker_embed)
        
        # Condense full prompt features into summary tokens for 10ms efficiency
        summary_tokens = self.uclm_core.prompt_resampler(prompt_feats)
        
        return refined_embed, summary_tokens, vq_loss, indices

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
        cross_attn_mask: torch.Tensor | None = None, # New SOTA field
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

        # SOTA: Explicitly calculate frame indices for Progress-aware RoPE
        T = content_features.shape[1]
        frame_indices = torch.arange(T, device=content_features.device).unsqueeze(0).expand(source_a_t.shape[0], -1)

        logits_a, logits_b, x_out = self.uclm_core.forward_no_cache(
            queries=content_features,
            memory=content_features, # VC cross-attends to its own latent sequence
            a_ctx=a_ctx,
            b_ctx=b_ctx,
            state_cond=state_cond,
            speaker_embed=speaker_embed,
            cfg_scale=cfg_scale,
            position_indices=frame_indices,
            frame_offsets=frame_indices, # 1:1 mapping for VC
            cross_attn_mask=cross_attn_mask,
        )

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "vq_loss": vq_loss,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
            "hidden_states": x_out,
        }

    @torch.no_grad()
    def forward_streaming(
        self,
        queries: torch.Tensor,
        memory: torch.Tensor,
        a_ctx: torch.Tensor,
        b_ctx: torch.Tensor,
        speaker_embed: torch.Tensor,
        explicit_state: Optional[torch.Tensor] = None,
        ssl_state: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        f0_condition: Optional[torch.Tensor] = None,
        dialogue_context: Optional[torch.Tensor] = None,
        acting_intent: Optional[torch.Tensor] = None,
        prosody_latent: Optional[torch.Tensor] = None,
        position_indices: Optional[torch.Tensor] = None,
        precomputed_film_params: Optional[torch.Tensor] = None,
        state_cond: Optional[torch.Tensor] = None, # Explicitly named for older tests
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Incremental forward pass for low-latency streaming (Worker 04)."""
        
        # SOTA: Prioritize explicit_state but fallback to state_cond if provided
        eff_explicit = explicit_state if explicit_state is not None else state_cond
        
        if eff_explicit is None:
            eff_explicit = torch.zeros((queries.shape[0], queries.shape[1], 8), device=queries.device)
        if ssl_state is None:
            d_ssl = self.voice_state_enc.ssl_proj[0].in_features
            ssl_state = torch.zeros((queries.shape[0], queries.shape[1], d_ssl), device=queries.device)

        # 1. Apply F0 conditioning to queries
        if f0_condition is not None:
            queries = queries + self.f0_proj(f0_condition)

        # 2. Apply dialogue/acting/prosody
        queries = self.context_projector(
            queries,
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_latent,
        )

        # 3. Voice state conditioning
        v_out = self.voice_state_enc(explicit_state, ssl_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

        # 4. Core transformer with KV cache
        return self.uclm_core(
            queries=queries,
            memory=memory,
            a_ctx=a_ctx,
            b_ctx=b_ctx,
            speaker_embed=speaker_embed,
            state_cond=state_cond,
            cfg_scale=cfg_scale,
            kv_caches=kv_caches,
            position_indices=position_indices,
        )

    @torch.no_grad()
    def bake_film_params(self, speaker_embed: torch.Tensor) -> torch.Tensor:
        """Pre-compute FiLM parameters from speaker embedding (SOTA efficiency)."""
        spk_flat = self.uclm_core.speaker_proj(speaker_embed)
        film_params = self.uclm_core.speaker_film_gen(spk_flat)
        # [B, n_layers, 2, d_model]
        return film_params.view(speaker_embed.shape[0], self.uclm_core.n_layers, 2, self.uclm_core.d_model)

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
        acting_texture_latent: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        prompt_kv_cache: torch.Tensor | None = None,
        acoustic_history: tuple[torch.Tensor, torch.Tensor] | None = None,
        bootstrap_alignment: dict | None = None,
        text_suprasegmentals: torch.Tensor | None = None,
        position_indices: torch.Tensor | None = None,
        frame_offsets: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        distilled_cfg_cond: torch.Tensor | None = None,
    ) -> dict:
        """Pointer-based TTS forward pass.

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
            phoneme_indices: optional [B, T] phoneme indices per frame for training.
                Overrides internal pointer-state or uniform fallback when provided.

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
            idx = bootstrap_alignment["phoneme_indices"]
            content_features = phoneme_features[torch.arange(B).unsqueeze(1), idx, :]
        elif pointer_state is not None:
            # Inference or Latent phase: Use pointer_state.text_index
            # In streaming, text_index is usually [B], so we unsqueeze
            idx = pointer_state.text_index
            if idx.ndim == 1:
                idx = idx.unsqueeze(1).expand(-1, target_length)
            content_features = phoneme_features[torch.arange(B).unsqueeze(1), idx, :]
        elif position_indices is not None:
            # Explicit teacher-forced indices (from MAS or bootstrap)
            content_features = phoneme_features[torch.arange(B).unsqueeze(1), position_indices, :]
        else:
            # Fallback: uniform distribution over phoneme sequence
            frame_indices = torch.arange(target_length, device=phoneme_features.device)
            idx = (frame_indices.float() * L / target_length).long().clamp(max=L - 1)
            content_features = phoneme_features[:, idx, :]

        # 3. Apply F0 conditioning
        if f0_condition is not None:
            content_features = content_features + self.f0_proj(f0_condition)

        # 3b. Apply dialogue/acting/prosody conditioning
        # v4 topology: acting_texture_latent (24-D) supersedes acting_intent (64-D).
        # When both are provided, acting_intent is suppressed to prevent
        # double-conditioning on the same semantic axis.
        effective_acting_intent = acting_intent
        if acting_texture_latent is not None and acting_intent is not None:
            effective_acting_intent = None  # suppressed: 24-D latent takes over
        content_features = self.context_projector(
            content_features,
            dialogue_context=dialogue_context,
            acting_intent=effective_acting_intent,
            prosody_latent=prosody_latent,
        )

        # 3c. v4: Apply acting texture latent conditioning (24-D -> d_model)
        if acting_texture_latent is not None:
            act_cond = self.acting_latent_conditioner(acting_texture_latent)  # [B, d_model]
            content_features = content_features + act_cond.unsqueeze(1)

        # 3d. Distilled CFG conditioning (independent of speaker_embed)
        if distilled_cfg_cond is not None:
            content_features = content_features + distilled_cfg_cond.unsqueeze(1)

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
            if hasattr(self, 'codec_condition') and self.codec_condition == "C":
                # Delay pattern: CB_k context at position t is target_a[:, k, t-k-1]
                a_ctx = torch.zeros_like(target_a)
                T_a = target_a.shape[2]
                for k in range(self.n_codebooks):
                    shift = k + 1
                    if shift < T_a:
                        a_ctx[:, k, shift:] = target_a[:, k, : T_a - shift]
                a_ctx = a_ctx.clamp_min(0)
            else:
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
        # SOTA: Hierarchical RoPE - use semantic position + temporal offset
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
            position_indices=position_indices,
            frame_offsets=frame_offsets,
            cross_attn_mask=cross_attn_mask,
        )

        # 7. Pointer head
        advance_logit, progress_delta, boundary_confidence = self.pointer_head(x_out)

        # 8. Codec condition routing (v4)
        if hasattr(self, 'codec_condition') and self.codec_condition == "B":
            # CB0: standard AR head, CB1-7: NAR refinement
            nar_logits = self.nar_head(x_out)
            B_sz, T_sz, _ = nar_logits.shape
            nar_logits = nar_logits.view(B_sz, T_sz, self.n_codebooks - 1, self.rvq_vocab_size)
            nar_logits = nar_logits.permute(0, 2, 1, 3)  # [B, 7, T, V]
            # Replace CB1-7 in logits_a
            logits_a[:, 1:, :, :] = nar_logits
        elif hasattr(self, 'codec_condition') and self.codec_condition == "D":
            # Single codebook: project hidden states to 8192-vocab
            single_logits = self.single_cb_head(x_out)  # [B, T, 8192]
            logits_a = single_logits.unsqueeze(1)  # [B, 1, T, 8192]

        return {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "advance_logit": advance_logit,
            "progress_delta": progress_delta,
            "boundary_confidence": boundary_confidence,
            "adv_logits": v_out[1] if isinstance(v_out, tuple) else None,
            "hidden_states": x_out,
            "physical_pred": self.physical_prediction_head(x_out),  # [B, T, 12]
            "next_pointer_state": None,  # populated at inference time
        }

    @torch.no_grad()
    def generate(
        self,
        text_ids: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
        max_frames: int = 1000,
        acting_texture_latent: torch.Tensor | None = None,
        **kw,
    ) -> dict:
        """RL rollout generation — autoregressive codec token sampling.

        Wraps forward_tts_pointer in autoregressive mode: at each frame,
        sample codec tokens from logits and feed back as context.

        Args:
            text_ids: [B, L] phoneme / acting-tag token IDs.
            speaker_embed: [B, d_speaker] speaker embedding.
            max_frames: maximum number of frames to generate.

        Returns:
            dict with ``audio`` (None), ``codec_tokens`` [B, n_codebooks, T],
            ``log_probs`` [B, T], ``hidden_states`` [B, T, D].
        """
        B, L = text_ids.shape
        device = text_ids.device

        if speaker_embed is None:
            speaker_embed = torch.zeros(B, self.uclm_core.d_speaker, device=device)

        # Create dummy conditioning
        language_ids = torch.zeros_like(text_ids)
        d_explicit = self.voice_state_enc.explicit_proj[0].in_features
        d_ssl = self.voice_state_enc.ssl_proj[0].in_features
        explicit_state = torch.zeros(B, max_frames, d_explicit, device=device)
        ssl_state = torch.zeros(B, max_frames, d_ssl, device=device)

        # Teacher-forced pass with zero targets to get logits over all frames
        target_a = torch.zeros(B, self.n_codebooks, max_frames, dtype=torch.long, device=device)
        target_b = torch.zeros(B, 4, max_frames, dtype=torch.long, device=device)

        out = self.forward_tts_pointer(
            phoneme_ids=text_ids,
            language_ids=language_ids,
            pointer_state=None,
            speaker_embed=speaker_embed,
            explicit_state=explicit_state,
            ssl_state=ssl_state,
            target_a=target_a,
            target_b=target_b,
            target_length=max_frames,
            acting_texture_latent=acting_texture_latent,
        )

        logits_a = out["logits_a"]  # [B, n_codebooks, T, V]
        hidden_states = out["hidden_states"]  # [B, T, D]

        # Sample codec tokens from logits (greedy for stability)
        # logits_a shape: [B, n_codebooks, T, rvq_vocab_size]
        codec_tokens = logits_a.argmax(dim=-1)  # [B, n_codebooks, T]

        # Compute log probs of sampled tokens
        log_probs_all = F.log_softmax(logits_a, dim=-1)  # [B, n_codebooks, T, V]
        # Gather log probs for the sampled tokens
        sampled_lp = log_probs_all.gather(
            -1, codec_tokens.unsqueeze(-1),
        ).squeeze(-1)  # [B, n_codebooks, T]
        # Average across codebooks for per-frame log prob
        log_probs = sampled_lp.mean(dim=1)  # [B, T]

        return {
            "audio": None,  # RL reward side runs vocoder
            "codec_tokens": codec_tokens,
            "log_probs": log_probs,
            "hidden_states": hidden_states,
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
        acting_texture_latent: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        text_suprasegmentals: torch.Tensor | None = None,
        position_indices: torch.Tensor | None = None,
        frame_offsets: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None, # New SOTA field
    ) -> dict:
        """Single-pass forward with cfg_scale injected for distillation training.

        The cfg_scale is projected to d_model and injected as an independent
        conditioning signal on content_features, separate from speaker_embed
        (v4 conditioning separation: speaker identity must remain timbre-only).
        """
        cfg_tensor = torch.tensor([[cfg_scale]], device=speaker_embed.device, dtype=torch.float32)
        cfg_cond = self.cfg_scale_embed(cfg_tensor)  # [1, d_model]

        return self.forward_tts_pointer(
            phoneme_ids=phoneme_ids,
            language_ids=language_ids,
            pointer_state=None,  # distillation uses teacher-forced alignment
            speaker_embed=speaker_embed,
            explicit_state=explicit_state,
            ssl_state=ssl_state,
            target_a=target_a,
            target_b=target_b,
            target_length=target_length,
            f0_condition=f0_condition,
            cfg_scale=1.0,
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_latent,
            acting_texture_latent=acting_texture_latent,
            delta_voice_state=delta_voice_state,
            position_indices=position_indices,
            frame_offsets=frame_offsets,
            cross_attn_mask=cross_attn_mask,
            distilled_cfg_cond=cfg_cond,
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

    def bake_film_params(self, speaker_embed: torch.Tensor) -> torch.Tensor:
        """SOTA: Pre-compute FiLM parameters from speaker embedding for efficient inference."""
        with torch.no_grad():
            spk_flat = self.uclm_core.speaker_proj(speaker_embed)
            film_params = self.uclm_core.speaker_film_gen(spk_flat)
            return film_params.view(speaker_embed.shape[0], self.uclm_core.n_layers, 2, self.d_model)

    def forward_streaming(
        self,
        queries: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        a_ctx: torch.Tensor | None = None,
        b_ctx: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
        explicit_state: torch.Tensor | None = None,
        ssl_state: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        kv_caches: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        prosody_latent: torch.Tensor | None = None,
        acting_texture_latent: torch.Tensor | None = None,
        f0_condition: torch.Tensor | None = None,
        prompt_summary_tokens: torch.Tensor | None = None,
        frame_index: int = 0,
        precomputed_film_params: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        frame_offsets: int | torch.Tensor | None = None, # New SOTA field
    ) -> dict:
        """Forward pass for streaming inference with internal voice state encoding."""
        if queries is None:
            raise ValueError("'queries' must be provided")

        # Default memory to queries if not provided
        if memory is None:
            memory = queries

        # 1. Encode Voice State (Matching training path)
        if explicit_state is not None:
            _ssl = ssl_state if ssl_state is not None else torch.zeros(
                explicit_state.shape[0], explicit_state.shape[1], self.voice_state_enc.ssl_proj[0].in_features,
                device=explicit_state.device
            )
            v_out = self.voice_state_enc(explicit_state, _ssl)
            state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

            # Apply delta voice state if provided
            if delta_voice_state is not None:
                state_cond = state_cond + self.delta_voice_state_proj(delta_voice_state)
        else:
            # Fallback if no state provided
            state_cond = torch.zeros(queries.shape[0], queries.shape[1], self.d_model, device=queries.device)

        # 2. Apply dialogue/acting/prosody conditioning
        # v4 topology: suppress legacy acting_intent when acting_texture_latent is present
        effective_acting_intent = acting_intent
        if acting_texture_latent is not None and acting_intent is not None:
            effective_acting_intent = None
        queries = self.context_projector(
            queries,
            dialogue_context=dialogue_context,
            acting_intent=effective_acting_intent,
            prosody_latent=prosody_latent,
        )

        # 2b. v4: Apply acting texture latent conditioning
        if acting_texture_latent is not None:
            act_cond = self.acting_latent_conditioner(acting_texture_latent)  # [B, d_model]
            queries = queries + act_cond.unsqueeze(1)

        cfg_tensor = torch.tensor([cfg_scale], device=queries.device)
        
        # SOTA: Hierarchical RoPE - pass both semantic index and temporal offset
        pos_idx = torch.tensor([[frame_index]], device=queries.device, dtype=torch.long)
        f_offs = None
        if frame_offsets is not None:
            if isinstance(frame_offsets, int):
                f_offs = torch.tensor([[frame_offsets]], device=queries.device, dtype=torch.long)
            else:
                f_offs = frame_offsets.to(queries.device)

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
            position_indices=pos_idx,
            precomputed_film_params=precomputed_film_params,
            cross_attn_mask=cross_attn_mask,
            frame_offsets=f_offs,
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
