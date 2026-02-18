"""Shared building blocks: CausalConvNeXtBlock, FiLMConditioner, timestep embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConvNeXtBlock(nn.Module):
    """ConvNeXt-v2 style block with causal (left-only) padding.

    Supports two modes:
    - Training (T > 1): full causal convolution via left-padding.
    - Streaming (T = 1): uses ``state_in`` buffer and returns ``state_out``.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        dilation: int = 1,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Context frames this layer needs from the past
        self.context_size = (kernel_size - 1) * dilation

        # Depthwise causal conv (groups=channels)
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=True,
            padding=0,  # We handle padding manually
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * expansion)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Linear(channels * expansion, channels)

    def forward(
        self,
        x: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: ``[B, C, T]`` input tensor.
            state_in: ``[B, C, context_size]`` past context for streaming.
                If ``None``, uses causal left-padding (training mode).

        Returns:
            Tuple of output ``[B, C, T]`` and ``state_out`` (or ``None``).
        """
        residual = x

        if state_in is not None:
            # Streaming: prepend past context
            x_padded = torch.cat([state_in, x], dim=-1)
            # Save new state: last context_size frames
            state_out = x_padded[:, :, -self.context_size:] if self.context_size > 0 else state_in
        else:
            # Training: causal left-padding
            x_padded = F.pad(x, (self.context_size, 0))
            state_out = None

        # Depthwise conv
        h = self.dwconv(x_padded)

        # LayerNorm (channels-last)
        h = h.transpose(1, 2)  # [B, T, C]
        h = self.norm(h)

        # Pointwise expand → SiLU → pointwise shrink
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)

        h = h.transpose(1, 2)  # [B, C, T]

        # Residual
        return residual + h, state_out


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation for speaker/IR conditioning."""

    def __init__(self, d_cond: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_cond, d_model * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: ``[B, C, T]`` feature tensor.
            cond: ``[B, d_cond]`` conditioning vector.

        Returns:
            ``[B, C, T]`` modulated features.
        """
        gamma_beta = self.proj(cond)  # [B, 2*C]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, C]
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class GlobalTimbreMemory(nn.Module):
    """Expand speaker embedding into a memory bank for cross-attention.

    Maps a static speaker vector into a set of memory entries that can be
    attended to by content features, enabling time-varying speaker modulation.
    """

    def __init__(
        self,
        d_speaker: int = 192,
        n_entries: int = 8,
        d_entry: int = 48,
    ) -> None:
        super().__init__()
        self.n_entries = n_entries
        self.d_entry = d_entry
        self.proj = nn.Linear(d_speaker, n_entries * d_entry)

    def forward(self, spk_embed: torch.Tensor) -> torch.Tensor:
        """Expand speaker embedding to memory bank.

        Args:
            spk_embed: ``[B, d_speaker]`` speaker embedding.

        Returns:
            ``[B, n_entries, d_entry]`` memory bank.
        """
        return self.proj(spk_embed).reshape(-1, self.n_entries, self.d_entry)


class TimbreCrossAttention(nn.Module):
    """Cross-attention from content features to timbre memory bank.

    Queries come from content features (d_model), keys/values from
    the timbre memory (d_entry). Multi-head attention with linear projections.
    """

    def __init__(
        self,
        d_model: int = 384,
        d_entry: int = 48,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.q_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.k_proj = nn.Linear(d_entry, d_model)
        self.v_proj = nn.Linear(d_entry, d_model)
        self.out_proj = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.scale = self.d_head ** -0.5

    def forward(
        self,
        content: torch.Tensor,
        memory: torch.Tensor,
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            content: ``[B, d_model, T]`` content features (queries).
            memory: ``[B, N, d_entry]`` timbre memory (keys/values).

        Returns:
            ``[B, d_model, T]`` attended features.
        """
        B, C, T = content.shape
        N = memory.shape[1]

        # Q from content: [B, d_model, T] → [B, n_heads, d_head, T]
        q = self.q_proj(content).reshape(B, self.n_heads, self.d_head, T)

        # K, V from memory: [B, N, d_entry] → [B, n_heads, d_head, N]
        k = self.k_proj(memory).reshape(B, N, self.n_heads, self.d_head)
        k = k.permute(0, 2, 3, 1)  # [B, n_heads, d_head, N]
        v = self.v_proj(memory).reshape(B, N, self.n_heads, self.d_head)
        v = v.permute(0, 2, 3, 1)  # [B, n_heads, d_head, N]

        # Attention: [B, n_heads, T, N]
        attn = torch.einsum("bhdt,bhdn->bhtn", q, k) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Weighted sum: [B, n_heads, d_head, T]
        out = torch.einsum("bhtn,bhdn->bhdt", attn, v)
        out = out.reshape(B, C, T)

        return self.out_proj(out)


class SemiCausalConvNeXtBlock(nn.Module):
    """ConvNeXt-v2 block with asymmetric (left + right) padding.

    Training: pad (left_context, right_context), output T = input T.
    Streaming: state_in provides left context, input contains right context.
      Output T = input T - right_context. State advances by 1 frame only.

    When ``right_context=0`` this is functionally identical to
    :class:`CausalConvNeXtBlock` (fully causal).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        expansion: int = 4,
        right_context: int = 0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        total_context = (kernel_size - 1) * dilation
        self.right_context = right_context
        self.left_context = total_context - right_context
        assert self.left_context >= 0, (
            f"right_context ({right_context}) exceeds total context ({total_context})"
        )

        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,
            bias=True,
            padding=0,
        )
        self.norm = nn.LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, channels * expansion)
        self.act = nn.SiLU()
        self.pwconv2 = nn.Linear(channels * expansion, channels)

    def forward(
        self,
        x: torch.Tensor,
        state_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            x: ``[B, C, T]`` input tensor.
            state_in: ``[B, C, left_context]`` past context for streaming.
                If ``None``, uses symmetric padding (training mode).

        Returns:
            Training: ``([B, C, T], None)`` — output length matches input.
            Streaming: ``([B, C, T - right_context], state_out)`` — trimmed.
        """
        T = x.shape[-1]

        if state_in is not None:
            # Streaming mode
            T_out = T - self.right_context
            residual = x[:, :, :T_out]

            x_padded = torch.cat([state_in, x], dim=-1)

            # State update: advance by 1 frame only (not T frames)
            if self.left_context > 0:
                state_concat = torch.cat([state_in, x[:, :, :1]], dim=-1)
                state_out = state_concat[:, :, -self.left_context:]
            else:
                state_out = state_in  # zero-size tensor passthrough
        else:
            # Training mode: pad both sides, output T = input T
            T_out = T
            residual = x
            x_padded = F.pad(x, (self.left_context, self.right_context))
            state_out = None

        h = self.dwconv(x_padded)
        assert h.shape[-1] == T_out, f"dwconv output T={h.shape[-1]} != T_out={T_out}"

        h = h.transpose(1, 2)  # [B, T_out, C]
        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)
        h = h.transpose(1, 2)  # [B, C, T_out]

        return residual + h, state_out


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep.

        Args:
            t: ``[B]`` or ``[B, 1]`` timestep values in [0, 1].

        Returns:
            ``[B, d_model]`` embedding.
        """
        if t.dim() == 2:
            t = t.squeeze(-1)

        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [B, d_model]

        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb
