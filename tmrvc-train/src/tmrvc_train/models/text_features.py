"""Text feature utilities for UCLM TTS (legacy duration expansion).

.. deprecated:: v3
    Legacy component retained for v2 checkpoint compatibility and ablation.
    The mainline v3 path uses pointer-based uniform distribution instead.

Expands phoneme-level features to frame-level using durations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def expand_phonemes_to_frames(
    phoneme_features: torch.Tensor,
    durations: torch.Tensor,
    target_length: int | None = None,
) -> torch.Tensor:
    """Expand phoneme-level features to frame-level using durations.

    Args:
        phoneme_features: [B, L, d] phoneme-level features.
        durations: [B, L] duration in frames for each phoneme.
        target_length: If set, pad/trim to this length.

    Returns:
        frame_features: [B, T, d] frame-level features.
    """
    B, L, d = phoneme_features.shape
    device = phoneme_features.device

    # Compute total frames
    total_frames = durations.sum(dim=-1).max().item()
    if target_length is not None:
        total_frames = max(total_frames, target_length)

    # Expand each phoneme
    frame_features = torch.zeros(B, total_frames, d, device=device)

    for b in range(B):
        frame_idx = 0
        for p in range(L):
            dur = int(durations[b, p].item())
            if dur == 0:
                continue
            if frame_idx + dur > total_frames:
                dur = total_frames - frame_idx
            frame_features[b, frame_idx : frame_idx + dur, :] = phoneme_features[
                b, p, :
            ]
            frame_idx += dur

    if target_length is not None and frame_features.shape[1] < target_length:
        # Pad to target length
        pad_len = target_length - frame_features.shape[1]
        frame_features = torch.cat(
            [frame_features, frame_features[:, -1:, :].expand(-1, pad_len, -1)],
            dim=1,
        )
    elif target_length is not None and frame_features.shape[1] > target_length:
        # Trim to target length
        frame_features = frame_features[:, :target_length, :]

    return frame_features


class TextFeatureExpander(nn.Module):
    """Expand phoneme-level text features to frame-level.

    Combines TextEncoder output with duration expansion.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        text_features: torch.Tensor,
        durations: torch.Tensor,
        target_length: int | None = None,
    ) -> torch.Tensor:
        """Expand text features to frame-level.

        Args:
            text_features: [B, L, d_model] phoneme-level features.
            durations: [B, L] duration in frames.
            target_length: Target frame length (optional).

        Returns:
            [B, T, d_model] frame-level features.
        """
        return expand_phonemes_to_frames(text_features, durations, target_length)
