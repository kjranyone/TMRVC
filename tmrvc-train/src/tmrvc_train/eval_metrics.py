"""Evaluation metrics: SECS, UTMOS proxy, F0 correlation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def speaker_embedding_cosine_similarity(
    embed_pred: torch.Tensor,
    embed_target: torch.Tensor,
) -> float:
    """Speaker Embedding Cosine Similarity (SECS).

    Args:
        embed_pred: ``[D]`` or ``[B, D]`` predicted speaker embedding.
        embed_target: ``[D]`` or ``[B, D]`` target speaker embedding.

    Returns:
        Mean cosine similarity (higher is better, max 1.0).
    """
    if embed_pred.dim() == 1:
        embed_pred = embed_pred.unsqueeze(0)
    if embed_target.dim() == 1:
        embed_target = embed_target.unsqueeze(0)

    cos_sim = F.cosine_similarity(embed_pred, embed_target, dim=-1)
    return cos_sim.mean().item()


def f0_correlation(
    f0_pred: torch.Tensor,
    f0_target: torch.Tensor,
    voiced_threshold: float = 10.0,
) -> float:
    """Pearson correlation of F0 on voiced frames.

    Args:
        f0_pred: ``[T]`` or ``[1, T]`` predicted F0 in Hz.
        f0_target: ``[T]`` or ``[1, T]`` target F0 in Hz.
        voiced_threshold: Minimum F0 value to consider a frame voiced.

    Returns:
        Pearson correlation coefficient (higher is better, max 1.0).
    """
    f0_pred = f0_pred.flatten()
    f0_target = f0_target.flatten()

    # Voiced frames only
    voiced = (f0_target > voiced_threshold) & (f0_pred > voiced_threshold)
    if voiced.sum() < 2:
        return 0.0

    pred_v = f0_pred[voiced]
    target_v = f0_target[voiced]

    # Pearson correlation
    pred_mean = pred_v.mean()
    target_mean = target_v.mean()
    pred_centered = pred_v - pred_mean
    target_centered = target_v - target_mean

    num = (pred_centered * target_centered).sum()
    denom = (pred_centered.pow(2).sum() * target_centered.pow(2).sum()).sqrt()

    if denom < 1e-8:
        return 0.0

    return (num / denom).item()


def utmos_proxy(
    mel_pred: torch.Tensor,
    mel_target: torch.Tensor,
) -> float:
    """UTMOS proxy score based on mel-spectrogram similarity.

    Simple proxy: 5.0 * (1 - normalized_mel_distance).
    For true UTMOS, use the SpeechMOS model.

    Args:
        mel_pred: ``[C, T]`` or ``[B, C, T]`` predicted mel.
        mel_target: ``[C, T]`` or ``[B, C, T]`` target mel.

    Returns:
        Proxy MOS score in [0, 5] range (higher is better).
    """
    if mel_pred.dim() == 2:
        mel_pred = mel_pred.unsqueeze(0)
    if mel_target.dim() == 2:
        mel_target = mel_target.unsqueeze(0)

    # Align time dimension
    T = min(mel_pred.shape[-1], mel_target.shape[-1])
    mel_pred = mel_pred[:, :, :T]
    mel_target = mel_target[:, :, :T]

    # Normalized distance
    distance = F.l1_loss(mel_pred, mel_target)
    # Rough normalization (typical mel range is about -10 to 2)
    normalized = (distance / 12.0).clamp(0, 1)

    return (5.0 * (1.0 - normalized)).item()
