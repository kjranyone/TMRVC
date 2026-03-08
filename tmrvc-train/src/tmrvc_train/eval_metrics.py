"""Evaluation metrics for TMRVC v3 validation (Worker 06).

Core metrics: SECS, UTMOS proxy, F0 correlation.
Extended metrics: acting alignment, CFG responsiveness, disentanglement,
prosody transfer leakage, few-shot speaker, voice state utility/calibration,
suprasegmental integrity, external baseline delta.
"""

from __future__ import annotations

from typing import Sequence

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


# ---------------------------------------------------------------------------
# Extended evaluation metrics (Worker 06 sign-off)
# ---------------------------------------------------------------------------


def acting_alignment_score(
    dialogue_context_embed: torch.Tensor,
    prosody_embed: torch.Tensor,
) -> float:
    """Semantic correlation between dialogue context and generated prosody.

    Args:
        dialogue_context_embed: ``[B, D]`` dialogue context embeddings.
        prosody_embed: ``[B, D]`` generated prosody embeddings.

    Returns:
        Mean cosine similarity in ``[-1, 1]`` (higher = better alignment).
    """
    if dialogue_context_embed.dim() == 1:
        dialogue_context_embed = dialogue_context_embed.unsqueeze(0)
    if prosody_embed.dim() == 1:
        prosody_embed = prosody_embed.unsqueeze(0)
    return F.cosine_similarity(dialogue_context_embed, prosody_embed, dim=-1).mean().item()


def cfg_responsiveness_score(
    f0_variances: Sequence[float],
    cfg_scales: Sequence[float],
) -> float:
    """Monotonic correlation between cfg_scale sweeps and acting intensity.

    Measures Spearman-like rank correlation between ``cfg_scales`` and
    ``f0_variances`` (proxy for acting intensity).

    Args:
        f0_variances: F0 variance measured at each cfg_scale point.
        cfg_scales: Corresponding cfg_scale values (ascending recommended).

    Returns:
        Correlation in ``[-1, 1]`` (positive = responsive to CFG).
    """
    if len(f0_variances) < 2:
        return 0.0
    t_var = torch.tensor(f0_variances, dtype=torch.float32)
    t_cfg = torch.tensor(cfg_scales, dtype=torch.float32)
    return _pearson(t_var, t_cfg)


def timbre_prosody_disentanglement_score(
    prosody_features_per_context: Sequence[torch.Tensor],
) -> float:
    """Prosody variance for same speaker-prompt under different contexts.

    High score means prosody changes with context despite identical timbre,
    demonstrating successful disentanglement.

    Args:
        prosody_features_per_context: List of ``[D]`` prosody feature vectors,
            one per dialogue-context condition (same speaker, same text).

    Returns:
        Between-context variance (higher = better disentanglement).
    """
    if len(prosody_features_per_context) < 2:
        return 0.0
    stacked = torch.stack([f.flatten().float() for f in prosody_features_per_context])
    mean = stacked.mean(dim=0)
    variance = ((stacked - mean) ** 2).mean().item()
    return variance


def prosody_transfer_leakage_score(
    ref_f0: torch.Tensor,
    gen_f0: torch.Tensor,
    voiced_threshold: float = 10.0,
) -> float:
    """Prompt-prosody leakage under cross-context prompting.

    Measures correlation between reference-prompt F0 contour and generated F0
    when target text/context differ. Low correlation = low leakage.

    Args:
        ref_f0: ``[T]`` reference prompt F0 contour.
        gen_f0: ``[T]`` generated F0 contour.
        voiced_threshold: Minimum F0 to consider voiced.

    Returns:
        Absolute Pearson correlation in ``[0, 1]`` (lower = less leakage).
    """
    return abs(f0_correlation(gen_f0, ref_f0, voiced_threshold))


def few_shot_speaker_score(
    speaker_similarity: float,
    intelligibility: float,
    sim_weight: float = 0.6,
) -> float:
    """Combined speaker-similarity and intelligibility under fixed short-reference.

    Args:
        speaker_similarity: Cosine similarity of speaker embeddings in ``[0, 1]``.
        intelligibility: CER-based intelligibility in ``[0, 1]`` (1 = perfect).
        sim_weight: Weight for speaker similarity (remainder for intelligibility).

    Returns:
        Weighted score in ``[0, 1]`` (higher is better).
    """
    return sim_weight * speaker_similarity + (1.0 - sim_weight) * intelligibility


def voice_state_label_utility_score(
    metric_with_labels: float,
    metric_without_labels: float,
) -> float:
    """Controllability uplift from curated voice_state supervision.

    Args:
        metric_with_labels: Controllability metric with voice_state supervision.
        metric_without_labels: Same metric without voice_state supervision (ablation).

    Returns:
        Relative uplift (positive = labels help). ``0.1`` means 10% improvement.
    """
    if abs(metric_without_labels) < 1e-8:
        return 0.0
    return (metric_with_labels - metric_without_labels) / abs(metric_without_labels)


def voice_state_calibration_error(
    pseudo_label_confidences: torch.Tensor,
    residual_errors: torch.Tensor,
) -> float:
    """Pseudo-label confidence vs downstream residual error.

    Bins samples by confidence and computes expected calibration error (ECE).

    Args:
        pseudo_label_confidences: ``[N]`` confidence scores in ``[0, 1]``.
        residual_errors: ``[N]`` residual errors (lower = better prediction).

    Returns:
        ECE-style calibration error in ``[0, 1]`` (lower is better).
    """
    n = pseudo_label_confidences.numel()
    if n == 0:
        return 0.0

    conf = pseudo_label_confidences.float().flatten()
    err = residual_errors.float().flatten()

    n_bins = min(10, n)
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = mask | (conf == bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = conf[mask].mean().item()
        avg_err = err[mask].mean().item()
        # Ideal: high confidence → low error. Calibration gap:
        ece += mask.sum().item() / n * abs(avg_conf - (1.0 - avg_err))
    return ece


def suprasegmental_integrity_score(
    accent_features: torch.Tensor | None,
    tone_features: torch.Tensor | None,
    boundary_features: torch.Tensor | None,
) -> float:
    """Alignment and non-emptiness of suprasegmental features.

    Checks that exported accent/tone/phrase-boundary features are present
    and internally consistent (non-zero, finite).

    Args:
        accent_features: ``[T]`` or ``[T, D]`` accent features (or None).
        tone_features: ``[T]`` or ``[T, D]`` tone features (or None).
        boundary_features: ``[T]`` or ``[T, D]`` boundary features (or None).

    Returns:
        Score in ``[0, 1]``. 1.0 = all features present, non-empty, finite.
    """
    checks = []
    for feat in (accent_features, tone_features, boundary_features):
        if feat is None:
            checks.append(0.0)
            continue
        feat = feat.float()
        present = feat.numel() > 0
        non_zero = feat.abs().sum().item() > 1e-8 if present else False
        finite = torch.isfinite(feat).all().item() if present else False
        checks.append(1.0 if (present and non_zero and finite) else 0.0)
    return sum(checks) / len(checks)


def external_baseline_delta(
    tmrvc_score: float,
    baseline_score: float,
) -> float:
    """Directional gap between TMRVC and a fixed public baseline.

    Args:
        tmrvc_score: TMRVC metric value (higher = better assumed).
        baseline_score: Pinned external baseline metric value.

    Returns:
        Signed delta: positive means TMRVC is better.
    """
    return tmrvc_score - baseline_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    if x.numel() < 2:
        return 0.0
    x = x.float()
    y = y.float()
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    denom = (xc.pow(2).sum() * yc.pow(2).sum()).sqrt()
    if denom < 1e-8:
        return 0.0
    return (num / denom).item()
