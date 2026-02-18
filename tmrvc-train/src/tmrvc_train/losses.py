"""Loss functions for TMRVC training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.audio import compute_stft


class FlowMatchingLoss(nn.Module):
    """MSE loss between predicted and target velocity fields."""

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute flow matching loss.

        Args:
            v_pred: ``[B, C, T]`` predicted velocity.
            v_target: ``[B, C, T]`` target velocity.
            mask: ``[B, 1, T]`` optional length mask (1=valid, 0=pad).

        Returns:
            Scalar loss.
        """
        loss = (v_pred - v_target).pow(2)
        if mask is not None:
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1) / v_pred.shape[1]
        return loss.mean()


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1).

    Uses three FFT sizes for multi-scale spectral comparison.
    """

    def __init__(
        self,
        fft_sizes: list[int] | None = None,
        hop_sizes: list[int] | None = None,
        win_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.fft_sizes = fft_sizes or [256, 512, 1024]
        self.hop_sizes = hop_sizes or [64, 128, 240]
        self.win_sizes = win_sizes or [256, 512, 960]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-resolution STFT loss.

        Args:
            pred: ``[B, T_samples]`` predicted waveform or ``[B, C, T]`` mel.
            target: ``[B, T_samples]`` target waveform or ``[B, C, T]`` mel.

        Returns:
            Scalar loss (sum of spectral convergence + log-mag L1 across resolutions).
        """
        # If inputs are mel spectrograms [B, C, T], flatten to [B, C*T] for STFT
        # But typically this is used on waveforms. For mel, we use direct L1.
        if pred.dim() == 3:
            # Mel-domain: compute L1 directly across resolutions is not applicable
            # Fall back to L1 on mel features
            return F.l1_loss(pred, target)

        total_loss = torch.tensor(0.0, device=pred.device)

        for n_fft, hop, win in zip(
            self.fft_sizes, self.hop_sizes, self.win_sizes
        ):
            pred_mag = compute_stft(pred, n_fft=n_fft, hop_length=hop, window_length=win)
            target_mag = compute_stft(target, n_fft=n_fft, hop_length=hop, window_length=win)

            # Spectral convergence
            sc = torch.norm(target_mag - pred_mag, p="fro") / (
                torch.norm(target_mag, p="fro") + 1e-8
            )

            # Log-magnitude L1
            log_pred = torch.log(pred_mag.clamp(min=1e-10))
            log_target = torch.log(target_mag.clamp(min=1e-10))
            log_mag_l1 = F.l1_loss(log_pred, log_target)

            total_loss = total_loss + sc + log_mag_l1

        return total_loss / len(self.fft_sizes)


class DMD2Loss(nn.Module):
    """GAN-based distribution matching loss for DMD2 distillation.

    Computes hinge-style GAN losses for both generator and discriminator.
    No regression loss — purely distribution-level matching.
    """

    def forward(
        self,
        logits_real: torch.Tensor,
        logits_fake: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute generator and discriminator losses.

        Args:
            logits_real: ``[B, 1]`` discriminator output for real samples.
            logits_fake: ``[B, 1]`` discriminator output for generated samples.

        Returns:
            Tuple of (generator_loss, discriminator_loss).
        """
        # Generator wants discriminator to think generated samples are real
        loss_gen = -logits_fake.mean()

        # Discriminator wants to correctly classify real/fake
        loss_disc = F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean()

        return loss_gen, loss_disc


class SVLoss(nn.Module):
    """Speaker Verification loss: 1 - cosine_similarity.

    Uses a frozen speaker encoder to compare generated and target speaker
    embeddings. Encourages the student to preserve speaker identity.
    """

    def __init__(self, speaker_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.speaker_encoder = speaker_encoder
        if speaker_encoder is not None:
            speaker_encoder.eval()
            for p in speaker_encoder.parameters():
                p.requires_grad = False

    def forward(
        self,
        pred_embed: torch.Tensor,
        target_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SV loss.

        Args:
            pred_embed: ``[B, D]`` speaker embedding from generated audio.
            target_embed: ``[B, D]`` speaker embedding from target audio.

        Returns:
            Scalar loss (1 - mean cosine similarity).
        """
        cos_sim = F.cosine_similarity(pred_embed, target_embed, dim=-1)
        return (1.0 - cos_sim).mean()


class SpeakerConsistencyLoss(nn.Module):
    """Speaker consistency loss: 1 - cosine similarity of speaker embeddings.

    Uses a frozen speaker encoder to extract embeddings from generated and
    target mel spectrograms.
    """

    def __init__(self, speaker_embed_fn: callable | None = None) -> None:
        super().__init__()
        self.speaker_embed_fn = speaker_embed_fn

    def forward(
        self,
        pred_embed: torch.Tensor,
        target_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Compute speaker consistency loss.

        Args:
            pred_embed: ``[B, D]`` speaker embedding from generated audio.
            target_embed: ``[B, D]`` speaker embedding from target audio.

        Returns:
            Scalar loss (1 - mean cosine similarity).
        """
        cos_sim = F.cosine_similarity(pred_embed, target_embed, dim=-1)
        return (1.0 - cos_sim).mean()
