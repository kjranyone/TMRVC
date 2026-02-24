"""StyleEncoder: emotion/style conditioning for expressive TTS.

Extends acoustic_params[32] with emotion_style[32] to form style_params[64].
Supports two input modes:
1. Audio reference: mel spectrogram → style vector
2. Text prompt: style description → style vector (Phase 3+)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tmrvc_core.constants import D_STYLE, N_EMOTION_CATEGORIES, N_MELS


class AudioStyleEncoder(nn.Module):
    """Extract emotion/style vector from mel spectrogram reference.

    Architecture::

        mel_ref[B, 80, T] → Conv2d stack (4 layers)
            → GlobalAvgPool → MLP
            → style[B, d_style]

    The output represents emotion_style[32d]:
    - [0:3]  Valence / Arousal / Dominance
    - [3:6]  VAD uncertainty
    - [6:9]  Speech rate / Energy / Pitch range
    - [9:21] Emotion category softmax (12 categories)
    - [21:29] Learned latent (unlabeled nuances)
    - [29:32] Reserved

    Args:
        n_mels: Input mel channels (default: 80).
        d_style: Output style dimension (default: D_STYLE=32).
        channels: Conv2d channel progression.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        d_style: int = D_STYLE,
        channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        channels = channels or [32, 64, 128, 256]
        self.d_style = d_style

        # Conv stack operating on mel as single-channel image
        layers: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            ])
            in_ch = out_ch
        self.conv_stack = nn.Sequential(*layers)

        # MLP: pool → style vector
        # After 4x stride-2: freq_bins = n_mels / 16 = 5
        freq_out = n_mels // (2 ** len(channels))
        self.mlp = nn.Sequential(
            nn.Linear(channels[-1] * max(1, freq_out), 256),
            nn.SiLU(),
            nn.Linear(256, d_style),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract style vector from mel spectrogram.

        Args:
            mel: ``[B, n_mels, T]`` mel spectrogram.

        Returns:
            ``[B, d_style]`` style conditioning vector.
        """
        # Reshape to [B, 1, n_mels, T] for Conv2d
        x = mel.unsqueeze(1)

        x = self.conv_stack(x)  # [B, C, F', T']

        # Global average pooling over time
        x = x.mean(dim=-1)  # [B, C, F']
        x = x.flatten(1)  # [B, C * F']

        return self.mlp(x)  # [B, d_style]


class StyleEncoder(nn.Module):
    """Unified style encoder with audio and text input modes.

    Combines acoustic_params[32] (IR + voice source) with
    emotion_style[32] to produce full style_params[64].

    Args:
        n_mels: Input mel channels.
        d_style: Emotion style dimension (default: D_STYLE=32).
        n_emotion_categories: Number of emotion categories for auxiliary loss.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        d_style: int = D_STYLE,
        n_emotion_categories: int = N_EMOTION_CATEGORIES,
    ) -> None:
        super().__init__()
        self.d_style = d_style

        # Audio reference encoder
        self.audio_encoder = AudioStyleEncoder(n_mels, d_style)

        # Auxiliary heads for supervised training
        self.emotion_head = nn.Linear(d_style, n_emotion_categories)
        self.vad_head = nn.Linear(d_style, 3)  # Valence, Arousal, Dominance
        self.prosody_head = nn.Linear(d_style, 3)  # Rate, Energy, Pitch range

    def forward(
        self,
        mel_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Extract style vector from audio reference.

        Args:
            mel_ref: ``[B, n_mels, T]`` reference mel spectrogram.

        Returns:
            ``[B, d_style]`` emotion style vector.
        """
        return self.audio_encoder(mel_ref)

    def predict_emotion(
        self,
        style: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict emotion labels from style vector (for auxiliary loss).

        Args:
            style: ``[B, d_style]`` style vector.

        Returns:
            Dict with keys "emotion_logits", "vad", "prosody".
        """
        return {
            "emotion_logits": self.emotion_head(style),  # [B, n_categories]
            "vad": self.vad_head(style),  # [B, 3]
            "prosody": self.prosody_head(style),  # [B, 3]
        }

    @staticmethod
    def combine_style_params(
        acoustic_params: torch.Tensor,
        emotion_style: torch.Tensor,
    ) -> torch.Tensor:
        """Combine acoustic_params[32] and emotion_style[32] into style_params[64].

        Args:
            acoustic_params: ``[B, 32]`` IR + voice source parameters.
            emotion_style: ``[B, 32]`` emotion style vector.

        Returns:
            ``[B, 64]`` combined style parameters.
        """
        return torch.cat([acoustic_params, emotion_style], dim=-1)

    @staticmethod
    def make_vc_style_params(
        acoustic_params: torch.Tensor,
    ) -> torch.Tensor:
        """Create style_params for VC mode (emotion_style = 0).

        Backward compatible: zero emotion_style preserves VC behavior
        since FiLM is initialized to identity for zero-valued dims.

        Args:
            acoustic_params: ``[B, 32]`` acoustic parameters.

        Returns:
            ``[B, 64]`` style params with zero emotion_style.
        """
        zeros = torch.zeros(
            acoustic_params.shape[0], 32,
            device=acoustic_params.device,
            dtype=acoustic_params.dtype,
        )
        return torch.cat([acoustic_params, zeros], dim=-1)
