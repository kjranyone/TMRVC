import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .voice_state_film import VoiceStateFiLM
from .control_encoder import ControlEncoderTemporal


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if state is not None:
            x = torch.cat([state, x], dim=-1)
        else:
            x = F.pad(x, (self.padding, 0))
        out = self.conv(x)
        next_state = x[:, :, -self.padding:] if self.padding > 0 else None
        return out, next_state


class SimpleUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels * stride, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = self.conv(x)
        return x.view(B, -1, T * self.stride)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_codebooks: int = 8, codebook_size: int = 1024, codebook_dim: int = 64):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, codebook_dim) for _ in range(n_codebooks)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        z_q = torch.zeros_like(z)
        indices = []
        residual = z.view(B, T, self.n_codebooks, self.codebook_dim).clone()
        for i, cb in enumerate(self.codebooks):
            res_i = residual[:, :, i, :]
            dist = (res_i**2).sum(-1, keepdim=True) + (cb.weight**2).sum(-1) - 2 * (res_i @ cb.weight.T)
            idx = dist.argmin(-1)
            indices.append(idx)
            q = cb(idx)
            z_q.view(B, T, self.n_codebooks, self.codebook_dim)[:, :, i, :] = q
            if i < self.n_codebooks - 1: residual[:, :, i+1, :] += (res_i - q)
        return z_q.view(B, T, D), torch.stack(indices, dim=1)


class EmotionAwareEncoder(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.conv1 = CausalConv1d(1, 64, 7)
        self.conv2 = CausalConv1d(64, 128, 5)
        self.conv3 = CausalConv1d(128, 256, 5)
        self.conv4 = CausalConv1d(256, d_model, 3)
        self.rvq = ResidualVectorQuantizer(n_codebooks=8, codebook_dim=d_model // 8)
        self.control_head = nn.ModuleList([nn.Linear(d_model, 64) for _ in range(4)])

    def forward(self, audio: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        new_states = []
        x, s1 = self.conv1(audio, states[0] if states else None); x = F.elu(x); new_states.append(s1)
        x, s2 = self.conv2(x, states[1] if states else None); x = F.elu(x); new_states.append(s2)
        x, s3 = self.conv3(x, states[2] if states else None); x = F.elu(x); new_states.append(s3)
        x, s4 = self.conv4(x, states[3] if states else None); x = F.elu(x); new_states.append(s4)
        
        indices = torch.arange(239, x.shape[-1], 240, device=x.device)
        x_sub = x.index_select(-1, indices)
        z_q, a_tokens = self.rvq(x_sub.transpose(1, 2))
        b_logits = torch.stack([head(z_q) for head in self.control_head], dim=1)
        return a_tokens, b_logits, new_states


class EmotionAwareDecoder(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.codebook_embeds = nn.ModuleList([nn.Embedding(1024, d_model // 8) for _ in range(8)])
        self.control_encoder = ControlEncoderTemporal(vocab_size=64, d_model=d_model)
        self.film = VoiceStateFiLM(8, d_model)
        self.up1 = SimpleUpsample(d_model, 256, 6)
        self.up2 = SimpleUpsample(256, 128, 5)
        self.up3 = SimpleUpsample(128, 64, 4)
        self.up4 = SimpleUpsample(64, 1, 2)

    def forward(self, a_tokens, b_tokens, voice_state, states=None):
        z_a = torch.cat([self.codebook_embeds[i](a_tokens[:, i, :]) for i in range(8)], dim=-1)
        z_b = self.control_encoder(b_tokens.transpose(1, 2))
        z = self.film((z_a + z_b).transpose(1, 2), voice_state)
        x = F.elu(self.up1(z))
        x = F.elu(self.up2(x))
        x = F.elu(self.up3(x))
        audio = torch.tanh(self.up4(x))
        return audio, [torch.empty(0)]*4


class EmotionAwareCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = EmotionAwareEncoder()
        self.decoder = EmotionAwareDecoder()
    def encode(self, audio, states=None): return self.encoder(audio, states)
    def decode(self, a, b, v, states=None): return self.decoder(a, b, v, states)
