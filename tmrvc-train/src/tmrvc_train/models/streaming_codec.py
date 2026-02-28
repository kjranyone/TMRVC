"""
Streaming Neural Audio Codec for TMRVC Codec-Latent Pipeline

Causal encoder-decoder with Residual Vector Quantization (RVQ).
Frame size: 20ms (480 samples @ 24kHz)
Token rate: 50 Hz (50 frames/sec × 4 codebooks = 200 tokens/sec)

State Layout (flat tensors for ONNX/Rust):
- Encoder state: [B, 512, 32] = 16384 floats (~64KB)
- Decoder state: [B, 256, 32] = 8192 floats (~32KB)

References:
- SoundStream (arXiv:2107.03312)
- EnCodec (arXiv:2210.13438)
- DAC (arXiv:2306.06546)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# State tensor dimensions (must match Rust constants.rs)
ENCODER_STATE_DIM = 512
ENCODER_STATE_FRAMES = 32
DECODER_STATE_DIM = 256
DECODER_STATE_FRAMES = 32


@dataclass
class CodecConfig:
    frame_size: int = 480
    sample_rate: int = 24000
    n_codebooks: int = 4
    codebook_size: int = 1024
    codebook_dim: int = 128
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    decoder_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64, 32])
    latent_dim: int = 512

    @property
    def frame_rate(self) -> int:
        return self.sample_rate // self.frame_size

    @property
    def token_rate(self) -> int:
        return self.frame_rate * self.n_codebooks


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            state = torch.zeros(
                x.shape[0], x.shape[1], self.padding, device=x.device, dtype=x.dtype
            )

        x = torch.cat([state, x], dim=2)
        new_state = x[:, :, -self.padding :]

        out = self.conv(x)
        return out, new_state


class CausalConvTranspose1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride=stride
        )
        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = out[:, :, : -self.padding]
        return out


class CausalConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = CausalConv1d(
            in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
        )
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.SiLU()

    @property
    def state_size(self) -> int:
        return self.conv.padding * self.in_channels

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, new_state = self.conv(x, state)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        return x, new_state


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_codebooks: int, codebook_size: int, codebook_dim: int):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.codebooks = nn.Parameter(
            torch.randn(n_codebooks, codebook_size, codebook_dim) * 0.1,
            requires_grad=True,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, D, T = x.shape
        assert D == self.codebook_dim, f"Expected dim {self.codebook_dim}, got {D}"

        x = x.transpose(1, 2)

        residual = x
        quantized = torch.zeros_like(x)
        indices = []
        commitment_loss = 0.0

        for i in range(self.n_codebooks):
            codebook = self.codebooks[i]
            dist = torch.cdist(residual.reshape(-1, self.codebook_dim), codebook)
            idx = dist.argmin(dim=1)
            idx = idx.view(B, T)
            indices.append(idx)

            quant_i = F.embedding(idx, codebook)
            quantized = quantized + quant_i
            residual = residual - quant_i

            commitment_loss = commitment_loss + F.mse_loss(residual, quant_i.detach())

        indices = torch.stack(indices, dim=1)
        quantized = quantized.transpose(1, 2)

        return quantized, indices, commitment_loss

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1)
        B, n_cb, T = indices.shape
        assert n_cb == self.n_codebooks

        quantized = torch.zeros(B, self.codebook_dim, T, device=indices.device)

        for i in range(self.n_codebooks):
            codebook = self.codebooks[i]
            idx = indices[:, i, :]
            quant_i = F.embedding(idx, codebook)
            quantized = quantized + quant_i.transpose(1, 2)

        return quantized


class StreamingCodecEncoder(nn.Module):
    def __init__(self, config: CodecConfig):
        super().__init__()
        self.config = config

        self.blocks = nn.ModuleList()
        in_ch = 1
        for i, out_ch in enumerate(config.encoder_channels):
            self.blocks.append(CausalConvBlock(in_ch, out_ch, kernel_size=7, stride=1))
            in_ch = out_ch

        self.proj = nn.Conv1d(config.encoder_channels[-1], config.latent_dim, 1)

        self._init_state_layout()

    def _init_state_layout(self):
        self._state_offsets = [0]
        self._state_shapes = []
        total = 0
        in_ch = 1
        for block in self.blocks:
            size = block.state_size
            self._state_shapes.append((in_ch, block.conv.padding))
            total += size
            self._state_offsets.append(total)
            in_ch = block.out_channels
        self._total_state_size = total

    @property
    def flat_state_size(self) -> int:
        return ENCODER_STATE_DIM * ENCODER_STATE_FRAMES

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, ENCODER_STATE_DIM, ENCODER_STATE_FRAMES, device=device
        )

    def _pack_states(self, states: List[torch.Tensor]) -> torch.Tensor:
        B = states[0].shape[0]
        flat = torch.zeros(
            B, self.flat_state_size, device=states[0].device, dtype=states[0].dtype
        )
        offset = 0
        for i, s in enumerate(states):
            sz = s.shape[1] * s.shape[2]
            if offset + sz <= self.flat_state_size:
                flat[:, offset : offset + sz] = s.reshape(B, -1)
            offset += sz
        return flat.view(B, ENCODER_STATE_DIM, ENCODER_STATE_FRAMES)

    def _unpack_states(self, flat: torch.Tensor, batch_size: int) -> List[torch.Tensor]:
        B = flat.shape[0]
        flat = flat.view(B, -1)
        states = []
        offset = 0
        in_ch = 1
        for block in self.blocks:
            sz = in_ch * block.conv.padding
            if offset + sz <= flat.shape[1]:
                s = flat[:, offset : offset + sz].view(B, in_ch, block.conv.padding)
            else:
                s = torch.zeros(
                    B, in_ch, block.conv.padding, device=flat.device, dtype=flat.dtype
                )
            states.append(s)
            offset += sz
            in_ch = block.out_channels
        return states

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        if state is None:
            states = [None] * len(self.blocks)
        else:
            states = self._unpack_states(state, B)

        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block(x, states[i])
            new_states.append(new_state)

        x = self.proj(x)

        packed_state = self._pack_states(new_states)
        return x, packed_state


class StreamingCodecDecoder(nn.Module):
    def __init__(self, config: CodecConfig):
        super().__init__()
        self.config = config

        self.proj = nn.Conv1d(config.codebook_dim, config.decoder_channels[0], 1)

        self.blocks = nn.ModuleList()
        in_ch = config.decoder_channels[0]
        for i, out_ch in enumerate(config.decoder_channels[1:], 1):
            self.blocks.append(CausalConvBlock(in_ch, out_ch, kernel_size=7))
            in_ch = out_ch

        self.final_conv = CausalConv1d(config.decoder_channels[-1], 1, kernel_size=7)
        self.final_act = nn.Tanh()

    @property
    def flat_state_size(self) -> int:
        return DECODER_STATE_DIM * DECODER_STATE_FRAMES

    def init_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, DECODER_STATE_DIM, DECODER_STATE_FRAMES, device=device
        )

    def forward(
        self, x: torch.Tensor, state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]

        if state is None:
            final_state = None
        else:
            state_flat = state.view(B, -1)
            sz = self.final_conv.padding
            final_state = (
                state_flat[:, : sz * self.final_conv.in_channels].view(
                    B, self.final_conv.in_channels, sz
                )
                if state_flat.shape[1] >= sz * self.final_conv.in_channels
                else None
            )

        x = self.proj(x)

        for block in self.blocks:
            if isinstance(block, CausalConvTranspose1d):
                x = block(x)
            else:
                x, _ = block(x, None)

        x, new_final_state = self.final_conv(x, final_state)
        x = self.final_act(x)

        B_out = x.shape[0]
        new_state = torch.zeros(
            B_out, self.flat_state_size, device=x.device, dtype=x.dtype
        )
        if new_final_state is not None:
            sz = new_final_state.shape[1] * new_final_state.shape[2]
            new_state[:, :sz] = new_final_state.reshape(B_out, -1)
        new_state = new_state.view(B_out, DECODER_STATE_DIM, DECODER_STATE_FRAMES)

        return x, new_state


class StreamingCodec(nn.Module):
    def __init__(self, config: Optional[CodecConfig] = None):
        super().__init__()
        self.config = config or CodecConfig()

        self.encoder = StreamingCodecEncoder(self.config)
        self.decoder = StreamingCodecDecoder(self.config)
        self.rvq = ResidualVectorQuantizer(
            self.config.n_codebooks, self.config.codebook_size, self.config.codebook_dim
        )

    def encode(
        self, x: torch.Tensor, encoder_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent, new_encoder_state = self.encoder(x, encoder_state)
        latent_for_rvq = latent[:, : self.config.codebook_dim, :]
        quantized, indices, commit_loss = self.rvq(latent_for_rvq)
        return indices, quantized, new_encoder_state

    def decode(
        self, indices: torch.Tensor, decoder_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        quantized = self.rvq.decode(indices)
        audio, new_decoder_state = self.decoder(quantized, decoder_state)
        return audio, new_decoder_state

    def forward(
        self,
        x: torch.Tensor,
        encoder_state: Optional[torch.Tensor] = None,
        decoder_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices, quantized, new_encoder_state = self.encode(x, encoder_state)
        audio, new_decoder_state = self.decode(indices, decoder_state)

        _, _, commit_loss = self.rvq(
            self.encoder(x, encoder_state)[0][:, : self.config.codebook_dim, :]
        )

        return audio, indices, commit_loss, new_encoder_state, new_decoder_state

    def init_states(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_state = self.encoder.init_state(batch_size, device)
        dec_state = self.decoder.init_state(batch_size, device)
        return enc_state, dec_state


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList()

        for i in range(scales):
            disc = nn.Sequential(
                nn.Conv1d(1, 16, 15, padding=7),
                nn.LeakyReLU(0.2),
                nn.Conv1d(16, 64, 41, stride=4, padding=20, groups=4),
                nn.LeakyReLU(0.2),
                nn.Conv1d(64, 256, 41, stride=4, padding=20, groups=16),
                nn.LeakyReLU(0.2),
                nn.Conv1d(256, 1024, 41, stride=4, padding=20, groups=64),
                nn.LeakyReLU(0.2),
                nn.Conv1d(1024, 1024, 5, padding=2),
                nn.LeakyReLU(0.2),
                nn.Conv1d(1024, 1, 3, padding=1),
            )
            self.discriminators.append(disc)

        self.pool = nn.AvgPool1d(4, stride=2, padding=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = []
        for disc in self.discriminators:
            outputs.append(disc(x))
            x = self.pool(x)
        return outputs


class MultiScaleSTFTLoss(nn.Module):
    def __init__(self, n_ffts: Optional[List[int]] = None):
        super().__init__()
        self.n_ffts = n_ffts or [512, 1024, 2048]

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n_fft in self.n_ffts:
            hop = n_fft // 4
            win = n_fft

            pred_stft = torch.stft(
                pred.squeeze(1), n_fft, hop, win, return_complex=True
            )
            target_stft = torch.stft(
                target.squeeze(1), n_fft, hop, win, return_complex=True
            )

            pred_mag = pred_stft.abs()
            target_mag = target_stft.abs()

            sc_loss = torch.norm(pred_mag - target_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-8
            )

            eps = 1e-5
            log_pred_mag = (pred_mag + eps).log()
            log_target_mag = (target_mag + eps).log()
            mag_loss = F.l1_loss(log_pred_mag, log_target_mag)

            loss = loss + sc_loss + mag_loss

        return loss / len(self.n_ffts)


def create_streaming_codec(config: Optional[CodecConfig] = None) -> StreamingCodec:
    return StreamingCodec(config)
