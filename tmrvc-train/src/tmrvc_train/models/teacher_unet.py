"""TeacherUNet: 4-stage U-Net with cross-attention for v-prediction diffusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import (
    D_CONTENT_VEC,
    D_SPEAKER,
    D_TEACHER_HIDDEN,
    N_IR_PARAMS,
    N_MELS,
    TEACHER_DOWN_CHANNELS,
    TEACHER_N_HEADS,
)
from tmrvc_train.modules import FiLMConditioner, SinusoidalTimestepEmbedding


class ResBlock(nn.Module):
    """Residual block with optional channel change."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention: Q from x, K/V from context."""

    def __init__(self, d_model: int, d_context: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_context, d_model)
        self.v_proj = nn.Linear(d_context, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attention.

        Args:
            x: ``[B, C, T]`` query features.
            context: ``[B, d_context, T_ctx]`` key/value context.

        Returns:
            ``[B, C, T]`` attended features + residual.
        """
        B, C, T = x.shape
        residual = x

        # Transpose to [B, T, C] for attention
        x_t = x.transpose(1, 2)
        ctx_t = context.transpose(1, 2)

        x_t = self.norm(x_t)

        q = self.q_proj(x_t)   # [B, T, C]
        k = self.k_proj(ctx_t)  # [B, T_ctx, C]
        v = self.v_proj(ctx_t)  # [B, T_ctx, C]

        # Reshape for multi-head
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, T, C)

        out = self.out_proj(attn)  # [B, T, C]
        return out.transpose(1, 2) + residual


class DownBlock(nn.Module):
    """Encoder block: downsample + ResBlock + optional CrossAttention."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        d_context: int,
        n_heads: int,
        has_attn: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_ch, out_ch)
        self.has_attn = has_attn
        if has_attn:
            self.attn = CrossAttention(out_ch, d_context, n_heads)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.downsample(x)
        x = self.res(x)
        if self.has_attn and context is not None:
            # Align context length to x
            if context.shape[-1] != x.shape[-1]:
                context = F.interpolate(context, size=x.shape[-1], mode="linear", align_corners=False)
            x = self.attn(x, context)
        return x


class UpBlock(nn.Module):
    """Decoder block: upsample + skip concat + ResBlock + optional CrossAttention."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        d_context: int,
        n_heads: int,
        has_attn: bool = False,
    ) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.res = ResBlock(out_ch + skip_ch, out_ch)
        self.has_attn = has_attn
        if has_attn:
            self.attn = CrossAttention(out_ch, d_context, n_heads)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.upsample(x)
        # Align time dimension with skip
        if x.shape[-1] != skip.shape[-1]:
            x = F.pad(x, (0, skip.shape[-1] - x.shape[-1]))
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        if self.has_attn and context is not None:
            if context.shape[-1] != x.shape[-1]:
                context = F.interpolate(context, size=x.shape[-1], mode="linear", align_corners=False)
            x = self.attn(x, context)
        return x


class TeacherUNet(nn.Module):
    """Teacher U-Net for v-prediction flow matching (~80M params).

    4-stage encoder/decoder with cross-attention at stages 3, 4 and bottleneck.
    Conditioning: content (cross-attn K/V), F0/speaker/IR/timestep (FiLM).
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        d_content: int = D_CONTENT_VEC,
        d_speaker: int = D_SPEAKER,
        n_ir_params: int = N_IR_PARAMS,
        d_hidden: int = D_TEACHER_HIDDEN,
        down_channels: list[int] | None = None,
        n_heads: int = TEACHER_N_HEADS,
    ) -> None:
        super().__init__()
        ch = down_channels or list(TEACHER_DOWN_CHANNELS)  # [128, 256, 384, 512]

        # Content projection for cross-attention
        self.content_proj = nn.Conv1d(d_content, d_hidden, kernel_size=1)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(d_hidden)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
        )

        # FiLM conditioners for bottleneck
        self.film_time = FiLMConditioner(d_hidden, d_hidden)
        self.film_f0 = FiLMConditioner(1, d_hidden)
        self.film_spk = FiLMConditioner(d_speaker, d_hidden)
        self.film_ir = FiLMConditioner(n_ir_params, d_hidden)

        # Input conv
        self.input_conv = nn.Conv1d(n_mels, ch[0], kernel_size=3, padding=1)

        # Encoder
        self.down1 = DownBlock(ch[0], ch[0], d_hidden, n_heads, has_attn=False)
        self.down2 = DownBlock(ch[0], ch[1], d_hidden, n_heads, has_attn=False)
        self.down3 = DownBlock(ch[1], ch[2], d_hidden, n_heads, has_attn=True)
        self.down4 = DownBlock(ch[2], ch[3], d_hidden, n_heads, has_attn=True)

        # Bottleneck
        self.bottleneck_res = ResBlock(ch[3], d_hidden)
        self.bottleneck_attn = CrossAttention(d_hidden, d_hidden, n_heads)
        self.bottleneck_out = ResBlock(d_hidden, ch[3])

        # Decoder (skip_ch must match encoder output channels)
        # s4 = down3 output = ch[2], s3 = down2 output = ch[1],
        # s2 = down1 output = ch[0], s1 = input_conv output = ch[0]
        self.up4 = UpBlock(ch[3], ch[2], ch[2], d_hidden, n_heads, has_attn=True)
        self.up3 = UpBlock(ch[2], ch[1], ch[1], d_hidden, n_heads, has_attn=True)
        self.up2 = UpBlock(ch[1], ch[0], ch[0], d_hidden, n_heads, has_attn=False)
        self.up1 = UpBlock(ch[0], ch[0], ch[0], d_hidden, n_heads, has_attn=False)

        # Output conv
        self.output_conv = nn.Conv1d(ch[0], n_mels, kernel_size=3, padding=1)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content: torch.Tensor,
        f0: torch.Tensor,
        spk_embed: torch.Tensor,
        ir_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_t: ``[B, 80, T]`` noisy mel at timestep t.
            t: ``[B]`` or ``[B, 1]`` diffusion timestep in [0, 1].
            content: ``[B, 768, T]`` content features (from ContentVec/WavLM).
            f0: ``[B, 1, T]`` log-F0 contour.
            spk_embed: ``[B, 192]`` speaker embedding.
            ir_params: ``[B, 24]`` IR parameters (optional, zeros if not available).

        Returns:
            ``[B, 80, T]`` predicted velocity.
        """
        B = x_t.shape[0]

        # Prepare conditioning
        ctx = self.content_proj(content)  # [B, d_hidden, T]

        # Timestep embedding
        if t.dim() == 2:
            t = t.squeeze(-1)
        t_emb = self.time_embed(t)  # [B, d_hidden]
        t_emb = self.time_mlp(t_emb)  # [B, d_hidden]

        # IR params default to zeros
        if ir_params is None:
            ir_params = torch.zeros(B, N_IR_PARAMS, device=x_t.device)

        # F0 mean for FiLM (collapse time to scalar)
        f0_mean = f0.mean(dim=-1)  # [B, 1]

        # Encoder
        h = self.input_conv(x_t)  # [B, ch[0], T]
        s1 = h
        h = self.down1(h, ctx)
        s2 = h
        h = self.down2(h, ctx)
        s3 = h
        h = self.down3(h, ctx)
        s4 = h
        h = self.down4(h, ctx)

        # Bottleneck with FiLM conditioning
        h = self.bottleneck_res(h)

        # Align context for bottleneck attention
        ctx_bn = ctx
        if ctx_bn.shape[-1] != h.shape[-1]:
            ctx_bn = F.interpolate(ctx_bn, size=h.shape[-1], mode="linear", align_corners=False)
        h = self.bottleneck_attn(h, ctx_bn)

        # FiLM conditioning
        h = self.film_time(h, t_emb)
        h = self.film_f0(h, f0_mean)
        h = self.film_spk(h, spk_embed)
        h = self.film_ir(h, ir_params)

        h = self.bottleneck_out(h)

        # Decoder
        h = self.up4(h, s4, ctx)
        h = self.up3(h, s3, ctx)
        h = self.up2(h, s2, ctx)
        h = self.up1(h, s1, ctx)

        # Output
        v_pred = self.output_conv(h)  # [B, 80, T]

        # Ensure output matches input time dimension
        if v_pred.shape[-1] != x_t.shape[-1]:
            v_pred = v_pred[:, :, :x_t.shape[-1]]

        return v_pred
