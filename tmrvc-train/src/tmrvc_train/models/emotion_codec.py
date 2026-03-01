import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .voice_state_film import VoiceStateFiLM
from .control_encoder import ControlEncoder


class CausalConv1d(nn.Module):
    """Causal 1D convolution with padding on left side only."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantization for acoustic tokens.

    Args:
        n_codebooks: Number of RVQ codebooks (8)
        codebook_size: Vocabulary size per codebook (1024)
        codebook_dim: Dimension per codebook (128)
    """

    def __init__(
        self, n_codebooks: int = 8, codebook_size: int = 1024, codebook_dim: int = 128
    ):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        self.codebooks = nn.ModuleList(
            [nn.Embedding(codebook_size, codebook_dim) for _ in range(n_codebooks)]
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, T, latent_dim] where latent_dim = n_codebooks * codebook_dim

        Returns:
            z_q: [B, T, latent_dim] quantized latent
            indices: [B, n_codebooks, T] codebook indices
            vq_loss: scalar commitment loss
        """
        B, T, D = z.shape
        assert D == self.n_codebooks * self.codebook_dim

        z_reshaped = z.view(B, T, self.n_codebooks, self.codebook_dim)
        z_reshaped = z_reshaped.permute(0, 2, 1, 3)

        indices_list = []
        z_q_list = []
        total_loss = 0.0

        residual = z_reshaped.clone()

        for i, codebook in enumerate(self.codebooks):
            residual_flat = residual[:, i, :, :].reshape(-1, self.codebook_dim)

            distances = (
                torch.sum(residual_flat**2, dim=1, keepdim=True)
                + torch.sum(codebook.weight**2, dim=1)
                - 2 * torch.matmul(residual_flat, codebook.weight.t())
            )

            idx = torch.argmin(distances, dim=1)
            idx = idx.view(B, T)

            indices_list.append(idx)

            q_vec = codebook(idx)
            z_q_list.append(q_vec)

            commitment_loss = F.mse_loss(q_vec.detach(), residual[:, i, :, :])
            total_loss = total_loss + commitment_loss

        indices = torch.stack(indices_list, dim=1)
        z_q = torch.stack(z_q_list, dim=2).view(B, T, D)

        return z_q, indices, total_loss


class EmotionAwareEncoder(nn.Module):
    """Dual-stream encoder producing acoustic (A_t) and control (B_t) tokens.

    Architecture:
        - Encoder backbone: Multi-scale causal convolutions
        - Acoustic head: RVQ -> A_t [B, 8, T]
        - Control head: -> B_t [B, 4, T]
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 512,
        n_codebooks: int = 8,
        rvq_vocab_size: int = 1024,
        control_vocab_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks

        self.encoder = nn.Sequential(
            CausalConv1d(in_channels, 64, kernel_size=7),
            nn.ELU(),
            CausalConv1d(64, 128, kernel_size=5),
            nn.ELU(),
            CausalConv1d(128, 256, kernel_size=5),
            nn.ELU(),
            CausalConv1d(256, 512, kernel_size=3),
            nn.ELU(),
            CausalConv1d(512, d_model, kernel_size=3),
        )

        self.rvq = ResidualVectorQuantizer(
            n_codebooks=n_codebooks,
            codebook_size=rvq_vocab_size,
            codebook_dim=d_model // n_codebooks,
        )

        self.control_head = nn.ModuleList(
            [nn.Linear(d_model, control_vocab_size) for _ in range(4)]
        )

    def forward(
        self, audio: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            audio: [B, 1, T] input audio waveform

        Returns:
            a_tokens: [B, 8, T] acoustic RVQ tokens
            b_logits: [B, 4, T, 64] control stream logits
            vq_loss: scalar VQ commitment loss
        """
        z = self.encoder(audio)
        z = z.transpose(1, 2)

        z_q, a_tokens, vq_loss = self.rvq(z)

        b_logits_list = []
        for head in self.control_head:
            b_logits_list.append(head(z_q))

        b_logits = torch.stack(b_logits_list, dim=1)

        return a_tokens, b_logits, vq_loss


class EmotionAwareDecoder(nn.Module):
    """Decoder with VoiceStateFiLM and ControlEncoder conditioning.

    Architecture:
        - Input: RVQ latent + ControlEncoder(B_t) + VoiceStateFiLM(voice_state)
        - Decoder backbone: Transposed causal convolutions
        - Output: Reconstructed audio
    """

    def __init__(
        self,
        out_channels: int = 1,
        d_model: int = 512,
        n_codebooks: int = 8,
        rvq_vocab_size: int = 1024,
        control_vocab_size: int = 64,
        d_voice_state: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_codebooks = n_codebooks

        self.codebook_embeds = nn.ModuleList(
            [
                nn.Embedding(rvq_vocab_size, d_model // n_codebooks)
                for _ in range(n_codebooks)
            ]
        )

        self.control_encoder = ControlEncoder(
            vocab_size=control_vocab_size,
            d_model=d_model // 4,
        )

        self.film = VoiceStateFiLM(d_voice_state, d_model)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(d_model, 512, kernel_size=3),
            nn.ELU(),
            nn.ConvTranspose1d(512, 256, kernel_size=5),
            nn.ELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=5),
            nn.ELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=7),
            nn.ELU(),
            nn.ConvTranspose1d(64, out_channels, kernel_size=7),
            nn.Tanh(),
        )

    def forward(
        self,
        a_tokens: torch.Tensor,
        b_tokens: torch.Tensor,
        voice_state: torch.Tensor,
        event_trace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            a_tokens: [B, 8, T] acoustic tokens
            b_tokens: [B, T, 4] control tokens
            voice_state: [B, T, 8] voice state parameters
            event_trace: [B, T, 64] optional event hysteresis trace

        Returns:
            audio: [B, 1, T'] reconstructed audio
        """
        B, _, T = a_tokens.shape

        embeds = []
        for i, emb in enumerate(self.codebook_embeds):
            embeds.append(emb(a_tokens[:, i, :]))
        z_a = torch.cat(embeds, dim=-1)

        z_b = self.control_encoder(b_tokens)
        z_b = z_b.transpose(1, 2)

        z = z_a + z_b

        if event_trace is not None:
            z = z + event_trace

        z = z.transpose(1, 2)
        z = self.film(z, voice_state)
        z = z.transpose(1, 2)

        audio = self.decoder(z)

        return audio


class EmotionAwareCodec(nn.Module):
    """Complete Emotion-Aware Neural Codec with dual streams (A_t, B_t).

    Token Spec v2:
        - A_t: [B, 8, T] acoustic RVQ tokens (0..1023 per codebook)
        - B_t: [B, 4, T] control event tuples [op, type, dur, int]
    """

    def __init__(
        self,
        d_model: int = 512,
        n_codebooks: int = 8,
        rvq_vocab_size: int = 1024,
        control_vocab_size: int = 64,
        d_voice_state: int = 8,
    ):
        super().__init__()

        self.encoder = EmotionAwareEncoder(
            d_model=d_model,
            n_codebooks=n_codebooks,
            rvq_vocab_size=rvq_vocab_size,
            control_vocab_size=control_vocab_size,
        )

        self.decoder = EmotionAwareDecoder(
            d_model=d_model,
            n_codebooks=n_codebooks,
            rvq_vocab_size=rvq_vocab_size,
            control_vocab_size=control_vocab_size,
            d_voice_state=d_voice_state,
        )

    def encode(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode audio to dual-stream tokens.

        Args:
            audio: [B, 1, T] input audio

        Returns:
            a_tokens: [B, 8, T] acoustic tokens
            b_tokens: [B, 4, T] control tokens (argmax of logits)
        """
        a_tokens, b_logits, _ = self.encoder(audio)
        b_tokens = b_logits.argmax(dim=-1)
        return a_tokens, b_tokens

    def decode(
        self,
        a_tokens: torch.Tensor,
        b_tokens: torch.Tensor,
        voice_state: torch.Tensor,
        event_trace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode dual-stream tokens to audio.

        Args:
            a_tokens: [B, 8, T] acoustic tokens
            b_tokens: [B, 4, T] control tokens
            voice_state: [B, T, 8] voice state
            event_trace: optional event hysteresis

        Returns:
            audio: [B, 1, T'] reconstructed audio
        """
        return self.decoder(a_tokens, b_tokens, voice_state, event_trace)

    def forward(
        self,
        audio: torch.Tensor,
        voice_state: torch.Tensor,
        event_trace: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full encode-decode pass.

        Args:
            audio: [B, 1, T] input audio
            voice_state: [B, T, 8] voice state
            event_trace: optional hysteresis

        Returns:
            audio_recon: [B, 1, T'] reconstructed audio
            a_tokens: [B, 8, T] acoustic tokens
            b_logits: [B, 4, T, 64] control logits
            vq_loss: scalar VQ loss
        """
        a_tokens, b_logits, vq_loss = self.encoder(audio)
        b_tokens = b_logits.argmax(dim=-1)
        audio_recon = self.decoder(a_tokens, b_tokens, voice_state, event_trace)

        return audio_recon, a_tokens, b_logits, vq_loss


def multiscale_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_ffts: list = [512, 1024, 2048],
) -> torch.Tensor:
    """Multi-scale STFT reconstruction loss."""
    loss = 0.0
    for n_fft in n_ffts:
        hop = n_fft // 4
        win = n_fft

        pred_spec = torch.stft(pred.squeeze(1), n_fft, hop, win, return_complex=True)
        target_spec = torch.stft(
            target.squeeze(1), n_fft, hop, win, return_complex=True
        )

        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()

        loss = loss + F.l1_loss(pred_mag, target_mag)
        loss = loss + F.l1_loss(
            torch.log(pred_mag + 1e-7),
            torch.log(target_mag + 1e-7),
        )

    return loss


class CodecLoss(nn.Module):
    """Combined loss for codec training."""

    def __init__(self, lambda_vq: float = 1.0, lambda_stft: float = 1.0):
        super().__init__()
        self.lambda_vq = lambda_vq
        self.lambda_stft = lambda_stft

    def forward(
        self,
        audio_pred: torch.Tensor,
        audio_target: torch.Tensor,
        vq_loss: torch.Tensor,
        b_logits: torch.Tensor,
        b_target: torch.Tensor,
    ) -> dict:
        """Compute all losses.

        Args:
            audio_pred: [B, 1, T] reconstructed audio
            audio_target: [B, 1, T] target audio
            vq_loss: scalar VQ commitment loss
            b_logits: [B, 4, T, 64] control logits
            b_target: [B, 4, T] control targets

        Returns:
            Dict with total loss and components
        """
        loss_stft = multiscale_stft_loss(audio_pred, audio_target)

        loss_control = 0.0
        for i in range(4):
            loss_control = loss_control + F.cross_entropy(
                b_logits[:, i, :, :].reshape(-1, b_logits.size(-1)),
                b_target[:, i, :].reshape(-1),
                ignore_index=-1,
            )
        loss_control = loss_control / 4

        total_loss = (
            self.lambda_stft * loss_stft + self.lambda_vq * vq_loss + loss_control
        )

        return {
            "loss": total_loss,
            "loss_stft": loss_stft,
            "loss_vq": vq_loss,
            "loss_control": loss_control,
        }
