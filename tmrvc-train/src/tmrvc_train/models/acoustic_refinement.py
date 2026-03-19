"""Acoustic Refinement Module (v3.1 interface).

Refines coarse AR codec tokens (first 1-2 RVQ layers) to full RVQ depth
(8 layers) using conditional flow matching. This bridges the quality gap
between fast AR generation (coarse tokens) and high-fidelity decoding
(full-depth tokens).

Architecture:
    1. Embed coarse tokens -> continuous representation
    2. Condition on speaker_embed and voice_state
    3. Flow-matching ODE from noise to refined token logits
    4. Output full-depth token predictions
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import D_MODEL


class RefinementBlock(nn.Module):
    """Single transformer block for the refinement network.

    Uses pre-norm architecture with multi-head self-attention and FFN.
    Conditioning is injected via adaptive layer norm (FiLM-style).
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = 8, d_cond: int = D_MODEL):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1),
        )

        # Adaptive layer norm conditioning (FiLM)
        self.cond_proj = nn.Linear(d_cond, d_model * 4)  # gamma1, beta1, gamma2, beta2

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with conditioning.

        Args:
            x: [B, T, D] input features.
            cond: [B, D_cond] conditioning vector.

        Returns:
            [B, T, D] refined features.
        """
        # Compute FiLM parameters from conditioning
        film = self.cond_proj(cond)  # [B, 4*D]
        gamma1, beta1, gamma2, beta2 = film.chunk(4, dim=-1)  # each [B, D]
        gamma1 = gamma1.unsqueeze(1)  # [B, 1, D]
        beta1 = beta1.unsqueeze(1)
        gamma2 = gamma2.unsqueeze(1)
        beta2 = beta2.unsqueeze(1)

        # Self-attention with adaptive norm
        h = self.norm1(x) * (1 + gamma1) + beta1
        h, _ = self.attn(h, h, h)
        x = x + h

        # FFN with adaptive norm
        h = self.norm2(x) * (1 + gamma2) + beta2
        h = self.ffn(h)
        x = x + h

        return x


class AcousticRefinementModule(nn.Module):
    """v3.1 interface: refines coarse AR codec tokens to full RVQ depth.

    Input:  coarse_tokens [B, T, n_coarse_codebooks] (e.g., first 1-2 RVQ layers)
    Output: refined_tokens [B, T, n_full_codebooks] (full 8 RVQ layers)

    Training uses conditional flow matching:
        - Sample t ~ U(0, 1)
        - Interpolate x_t = (1-t) * noise + t * target_embed
        - Predict velocity v(x_t, t, cond) -> target_embed - noise
        - Loss = MSE(v_pred, v_target)

    Inference uses Euler ODE integration from noise to signal.
    """

    def __init__(
        self,
        n_coarse: int = 2,
        n_full: int = 8,
        codebook_size: int = 1024,
        d_model: int = D_MODEL,
        n_heads: int = 8,
        n_layers: int = 4,
        n_steps: int = 4,
        d_speaker: int = 192,
        d_voice_state: int = D_MODEL,
    ):
        super().__init__()
        self.n_coarse = n_coarse
        self.n_full = n_full
        self.n_residual = n_full - n_coarse  # codebooks to predict
        self.codebook_size = codebook_size
        self.d_model = d_model
        self.n_steps = n_steps

        # Embeddings for coarse tokens (one per codebook)
        self.coarse_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model // n_coarse)
            for _ in range(n_coarse)
        ])

        # Project concatenated coarse embeddings to d_model
        self.coarse_proj = nn.Linear(d_model // n_coarse * n_coarse, d_model)

        # Conditioning projections
        self.speaker_proj = nn.Linear(d_speaker, d_model)
        self.voice_state_proj = nn.Linear(d_voice_state, d_model)

        # Time embedding for flow matching
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
        )

        # Noisy input projection (for the residual codebook embeddings)
        self.noise_proj = nn.Linear(d_model, d_model)

        # Refinement transformer blocks
        self.blocks = nn.ModuleList([
            RefinementBlock(d_model=d_model, n_heads=n_heads, d_cond=d_model)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        # Per-codebook output heads for residual codebooks
        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, codebook_size)
            for _ in range(self.n_residual)
        ])

        # Target embedding for residual codebooks (used during training)
        self.target_embeds = nn.ModuleList([
            nn.Embedding(codebook_size, d_model // self.n_residual)
            for _ in range(self.n_residual)
        ])
        self.target_proj = nn.Linear(
            d_model // self.n_residual * self.n_residual, d_model
        )

    def _build_condition(
        self,
        coarse_features: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
        voice_state: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build conditioning vector from coarse features, speaker, voice state, and time.

        Args:
            coarse_features: [B, T, D] embedded coarse tokens.
            speaker_embed: [B, d_speaker] optional speaker embedding.
            voice_state: [B, T, d_voice_state] or [B, d_voice_state] optional voice state.
            t: [B, 1] timestep in [0, 1].

        Returns:
            cond: [B, D] conditioning vector.
        """
        # Global summary of coarse features
        cond = coarse_features.mean(dim=1)  # [B, D]

        if speaker_embed is not None:
            cond = cond + self.speaker_proj(speaker_embed)

        if voice_state is not None:
            if voice_state.ndim == 3:
                voice_state = voice_state.mean(dim=1)
            cond = cond + self.voice_state_proj(voice_state)

        if t is not None:
            cond = cond + self.time_embed(t)

        return cond

    def _embed_coarse(self, coarse_tokens: torch.Tensor) -> torch.Tensor:
        """Embed coarse tokens to continuous features.

        Args:
            coarse_tokens: [B, T, n_coarse] integer tokens.

        Returns:
            [B, T, D] continuous features.
        """
        embeddings = []
        for i, embed in enumerate(self.coarse_embeds):
            embeddings.append(embed(coarse_tokens[:, :, i]))
        x = torch.cat(embeddings, dim=-1)  # [B, T, D//n_coarse * n_coarse]
        return self.coarse_proj(x)  # [B, T, D]

    def _embed_target(self, target_tokens: torch.Tensor) -> torch.Tensor:
        """Embed target residual tokens to continuous features.

        Args:
            target_tokens: [B, T, n_residual] integer tokens for residual codebooks.

        Returns:
            [B, T, D] continuous features.
        """
        embeddings = []
        for i, embed in enumerate(self.target_embeds):
            embeddings.append(embed(target_tokens[:, :, i]))
        x = torch.cat(embeddings, dim=-1)
        return self.target_proj(x)

    def _predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        coarse_features: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
        voice_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field v(x_t, t, cond).

        Args:
            x_t: [B, T, D] noisy feature at time t.
            t: [B, 1] timestep in [0, 1].
            coarse_features: [B, T, D] embedded coarse tokens.
            speaker_embed: [B, d_speaker] optional.
            voice_state: [B, T, d_vs] or [B, d_vs] optional.

        Returns:
            v: [B, T, D] predicted velocity.
        """
        cond = self._build_condition(coarse_features, speaker_embed, voice_state, t)

        # Add coarse features to noisy input
        x = self.noise_proj(x_t) + coarse_features

        for block in self.blocks:
            x = block(x, cond)

        x = self.output_norm(x)
        return x

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
        voice_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference: refine coarse tokens to full-depth token predictions.

        Uses Euler ODE integration from noise to produce refined residual
        codebook logits, then argmax to get token indices.

        Args:
            coarse_tokens: [B, T, n_coarse] coarse codec tokens (first n_coarse RVQ layers).
            speaker_embed: [B, d_speaker] optional speaker embedding.
            voice_state: [B, T, d_vs] or [B, d_vs] optional voice state.

        Returns:
            full_tokens: [B, T, n_full] complete codec tokens (coarse + refined).
        """
        B, T, _ = coarse_tokens.shape
        device = coarse_tokens.device

        coarse_features = self._embed_coarse(coarse_tokens)

        # Start from Gaussian noise
        x = torch.randn(B, T, self.d_model, device=device)

        # Euler ODE integration: t=0 (noise) -> t=1 (signal)
        dt = 1.0 / self.n_steps
        for i in range(self.n_steps):
            t_val = i * dt
            t = torch.full((B, 1), t_val, device=device)
            v = self._predict_velocity(x, t, coarse_features, speaker_embed, voice_state)
            x = x + dt * v

        # Decode refined features to token logits per residual codebook
        refined_tokens_list = []
        for head in self.output_heads:
            logits = head(x)  # [B, T, codebook_size]
            tokens = logits.argmax(dim=-1)  # [B, T]
            refined_tokens_list.append(tokens)

        refined_tokens = torch.stack(refined_tokens_list, dim=-1)  # [B, T, n_residual]

        # Concatenate coarse + refined
        full_tokens = torch.cat([coarse_tokens, refined_tokens], dim=-1)  # [B, T, n_full]

        return full_tokens

    def flow_matching_loss(
        self,
        coarse_tokens: torch.Tensor,
        target_full_tokens: torch.Tensor,
        speaker_embed: torch.Tensor | None = None,
        voice_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute conditional flow matching loss for training.

        Args:
            coarse_tokens: [B, T, n_coarse] coarse codec tokens.
            target_full_tokens: [B, T, n_full] ground-truth full-depth tokens.
            speaker_embed: [B, d_speaker] optional.
            voice_state: [B, T, d_vs] or [B, d_vs] optional.

        Returns:
            Dict with 'loss' (scalar), 'loss_flow' (flow matching MSE),
            and 'loss_ce' (cross-entropy on output heads).
        """
        B, T, _ = coarse_tokens.shape
        device = coarse_tokens.device

        coarse_features = self._embed_coarse(coarse_tokens)

        # Extract residual codebook targets
        target_residual = target_full_tokens[:, :, self.n_coarse:]  # [B, T, n_residual]

        # Embed target to continuous space
        target_embed = self._embed_target(target_residual)  # [B, T, D]

        # Sample random timestep
        t = torch.rand(B, 1, device=device)

        # Sample noise
        noise = torch.randn_like(target_embed)

        # Interpolate
        t_expanded = t.unsqueeze(-1)  # [B, 1, 1]
        x_t = (1.0 - t_expanded) * noise + t_expanded * target_embed

        # True velocity (straight-line OT path)
        v_target = target_embed - noise

        # Predict velocity
        v_pred = self._predict_velocity(x_t, t, coarse_features, speaker_embed, voice_state)

        # Flow matching MSE loss
        loss_flow = F.mse_loss(v_pred, v_target)

        # Cross-entropy loss on output heads (auxiliary, for faster convergence)
        loss_ce = torch.tensor(0.0, device=device)
        for i, head in enumerate(self.output_heads):
            logits = head(v_pred)  # [B, T, codebook_size]
            targets = target_residual[:, :, i]  # [B, T]
            loss_ce = loss_ce + F.cross_entropy(
                logits.reshape(-1, self.codebook_size),
                targets.reshape(-1),
            )
        loss_ce = loss_ce / len(self.output_heads)

        total_loss = loss_flow + 0.1 * loss_ce

        return {
            "loss": total_loss,
            "loss_flow": loss_flow,
            "loss_ce": loss_ce,
        }
