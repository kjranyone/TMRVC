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
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels * stride, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = self.conv(x)
        x = x.view(B, self.out_channels, self.stride, T)
        x = x.transpose(2, 3).reshape(B, self.out_channels, T * self.stride)
        return x


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_codebooks: int = 8, codebook_size: int = 1024, codebook_dim: int = 64):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_dim = codebook_dim
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, codebook_dim) for _ in range(n_codebooks)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = z.shape
        indices = []
        logits_list = []
        z_chunks = z.view(B, T, self.n_codebooks, self.codebook_dim)
        residual_chunks = [z_chunks[:, :, i, :].clone() for i in range(self.n_codebooks)]
        q_chunks = []

        for i, cb in enumerate(self.codebooks):
            res_i = residual_chunks[i]
            # dist shape: [B, T, codebook_size]
            dist = (res_i**2).sum(-1, keepdim=True) + (cb.weight**2).sum(-1) - 2 * (res_i @ cb.weight.T)
            
            # For distillation, we output negative distances as logits (so closest is highest probability)
            logits_list.append(-dist)
            
            idx = dist.argmin(-1)
            indices.append(idx)
            
            # Keep quantized chunk in a list to avoid in-place writes on autograd-tracked views.
            q_diff = cb(idx)
            q_chunks.append(q_diff)
            
            # For next step residual, use detached q to stop gradients from future codebooks
            if i < self.n_codebooks - 1:
                residual_chunks[i + 1] = residual_chunks[i + 1] + (res_i - q_diff.detach())

        z_q_differentiable = torch.stack(q_chunks, dim=2).reshape(B, T, D)
            
        # Straight-Through Estimator (STE):
        # Forward pass uses z_q_differentiable, but backward pass flows gradient to both z and codebooks
        z_q = z + (z_q_differentiable - z).detach() + (z_q_differentiable - z_q_differentiable.detach())
        
        # logits shape: [B, n_codebooks, T, codebook_size] -> [B, 8, T, 1024]
        logits = torch.stack(logits_list, dim=1)
        return z_q.view(B, T, D), torch.stack(indices, dim=1), logits


class EmotionAwareEncoder(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.conv1 = CausalConv1d(1, 64, 7)
        self.conv2 = CausalConv1d(64, 128, 5)
        self.conv3 = CausalConv1d(128, 256, 5)
        self.conv4 = CausalConv1d(256, d_model, 3)
        self.rvq = ResidualVectorQuantizer(n_codebooks=8, codebook_dim=d_model // 8)
        self.control_head = nn.ModuleList([nn.Linear(d_model, 64) for _ in range(4)])

    def forward(self, audio: torch.Tensor, states: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        new_states = []
        x, s1 = self.conv1(audio, states[0] if states else None); x = F.elu(x); new_states.append(s1)
        x, s2 = self.conv2(x, states[1] if states else None); x = F.elu(x); new_states.append(s2)
        x, s3 = self.conv3(x, states[2] if states else None); x = F.elu(x); new_states.append(s3)
        x, s4 = self.conv4(x, states[3] if states else None); x = F.elu(x); new_states.append(s4)
        
        indices = torch.arange(239, x.shape[-1], 240, device=x.device)
        x_sub = x.index_select(-1, indices)
        z_q, a_tokens, a_logits = self.rvq(x_sub.transpose(1, 2))
        b_logits = torch.stack([head(z_q) for head in self.control_head], dim=1)
        return a_tokens, b_logits, new_states, a_logits


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
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.encoder = EmotionAwareEncoder(d_model=d_model)
        self.decoder = EmotionAwareDecoder(d_model=d_model)

    def encode(self, audio, states=None): return self.encoder(audio, states)
    def decode(self, a, b, v, states=None): return self.decoder(a, b, v, states)


def multiscale_stft_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Multi-scale STFT loss for high-fidelity reconstruction."""
    pred_flat = pred.squeeze(1)
    target_flat = target.squeeze(1)
    
    total_loss = 0.0
    scales = [512, 1024, 2048]
    
    for n_fft in scales:
        hop = n_fft // 4
        win = n_fft
        
        stft_pred = torch.stft(
            pred_flat, n_fft=n_fft, hop_length=hop, win_length=win,
            window=torch.hann_window(win, device=pred.device),
            return_complex=True
        )
        stft_target = torch.stft(
            target_flat, n_fft=n_fft, hop_length=hop, win_length=win,
            window=torch.hann_window(win, device=target.device),
            return_complex=True
        )
        
        mag_pred = torch.abs(stft_pred) + 1e-7
        mag_target = torch.abs(stft_target) + 1e-7
        
        sc_loss = torch.norm(mag_pred - mag_target, p="fro") / (torch.norm(mag_target, p="fro") + 1e-7)
        log_loss = F.l1_loss(torch.log(mag_pred), torch.log(mag_target))
        total_loss += (sc_loss + log_loss)
        
    return total_loss / len(scales)


class CodecLoss(nn.Module):
    """Loss for EmotionAwareCodec: STFT + B_t CrossEntropy + A_t Distillation."""
    def __init__(self, lambda_stft: float = 1.0, lambda_control: float = 0.1, lambda_distill: float = 1.0):
        super().__init__()
        self.lambda_stft = lambda_stft
        self.lambda_control = lambda_control
        self.lambda_distill = lambda_distill

    def forward(self, audio_pred, audio_target, b_logits, b_target, a_logits=None, a_target=None):
        # 1. Reconstruction Loss
        loss_stft = multiscale_stft_loss(audio_pred, audio_target)
        
        # 2. Control Stream Loss (B_t)
        # b_logits: [B, 4, T, 64], b_target: [B, 4, T]
        B, n_slots, T, vocab = b_logits.shape
        loss_control = F.cross_entropy(
            b_logits.reshape(-1, vocab),
            b_target.reshape(-1),
            ignore_index=-1
        )
        
        # 3. Acoustic Distillation Loss (A_t)
        loss_distill = torch.tensor(0.0, device=audio_pred.device)
        if a_logits is not None and a_target is not None:
            # a_logits: [B, 8, T, 1024], a_target: [B, 8, T]
            B_a, n_cb, T_a, vocab_a = a_logits.shape
            loss_distill = F.cross_entropy(
                a_logits.reshape(-1, vocab_a),
                a_target.reshape(-1),
                ignore_index=-1
            )
        
        total = self.lambda_stft * loss_stft + self.lambda_control * loss_control + self.lambda_distill * loss_distill
        return {
            "loss": total,
            "loss_stft": loss_stft,
            "loss_control": loss_control,
            "loss_distill": loss_distill
        }
