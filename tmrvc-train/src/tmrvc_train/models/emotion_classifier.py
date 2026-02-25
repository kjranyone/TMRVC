"""Lightweight emotion classifier for pseudo-labeling.

Separate from StyleEncoder — this is a throwaway classifier used only
to generate pseudo-labels for unlabeled data. It is intentionally simple
(3-layer CNN + attention pooling) so it trains quickly on limited data.

Trained on labeled emotion datasets (Expresso, JVNV, EmoV-DB, RAVDESS),
then applied to unlabeled datasets (VCTK, JVS) to generate pseudo_emotion.json.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmrvc_core.constants import N_EMOTION_CATEGORIES, N_MELS


class EmotionClassifier(nn.Module):
    """Lightweight mel-based emotion classifier for pseudo-labeling.

    Architecture::

        mel[B, 80, T]
          → Conv1d(80, 128, k=5, s=2)  → BN → SiLU
          → Conv1d(128, 128, k=3, s=2) → BN → SiLU
          → Conv1d(128, 128, k=3, s=1) → BN → SiLU
          → AttentionPool (weighted average over T)
          → Linear(128, n_classes)
          → logits[B, n_classes]

    Also predicts VAD regression (3d) for richer pseudo-labels.

    Args:
        n_mels: Number of mel channels.
        hidden: Hidden dimension for conv layers.
        n_classes: Number of emotion categories.
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        hidden: int = 128,
        n_classes: int = N_EMOTION_CATEGORIES,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
        )

        # Attention pooling: learn which frames matter most
        self.attn = nn.Linear(hidden, 1)

        # Heads
        self.emotion_head = nn.Linear(hidden, n_classes)
        self.vad_head = nn.Linear(hidden, 3)

    def forward(
        self, mel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Classify emotion from mel spectrogram.

        Args:
            mel: ``[B, n_mels, T]`` mel spectrogram.

        Returns:
            Dict with:
            - ``emotion_logits``: ``[B, n_classes]``
            - ``vad``: ``[B, 3]`` predicted VAD values
        """
        x = self.conv(mel)  # [B, hidden, T']

        # Attention pooling
        attn_weights = self.attn(x.transpose(1, 2))  # [B, T', 1]
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x.transpose(1, 2) * attn_weights).sum(dim=1)  # [B, hidden]

        return {
            "emotion_logits": self.emotion_head(pooled),
            "vad": self.vad_head(pooled),
        }

    def predict(
        self, mel: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Predict emotion with softmax probabilities (inference mode).

        Args:
            mel: ``[B, n_mels, T]`` mel spectrogram.

        Returns:
            Dict with:
            - ``emotion_probs``: ``[B, n_classes]`` softmax probabilities
            - ``emotion_ids``: ``[B]`` predicted class indices
            - ``confidence``: ``[B]`` max probability per sample
            - ``vad``: ``[B, 3]``
        """
        with torch.no_grad():
            out = self.forward(mel)
            probs = F.softmax(out["emotion_logits"], dim=-1)
            confidence, ids = probs.max(dim=-1)
            return {
                "emotion_probs": probs,
                "emotion_ids": ids,
                "confidence": confidence,
                "vad": out["vad"],
            }
