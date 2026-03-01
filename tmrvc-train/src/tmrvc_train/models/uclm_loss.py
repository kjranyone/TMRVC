import torch
import torch.nn as nn
import torch.nn.functional as F

from .duration_predictor import duration_loss


def uclm_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    vq_loss: torch.Tensor | None = None,
    log_durations: torch.Tensor | None = None,
    dur_target: torch.Tensor | None = None,
    phoneme_mask: torch.Tensor | None = None,
    adv_logits: torch.Tensor | None = None,
    speaker_labels: torch.Tensor | None = None,
    lambda_vq: float = 1.0,
    lambda_dur: float = 0.1,
    lambda_adv: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute the multi-task loss for Disentangled UCLM.

    Args:
        logits_a: [B, 8, T, 1024] Acoustic stream predictions
        logits_b: [B, 4, T, 64] Control stream predictions
        target_a: [B, 8, T] Ground truth acoustic tokens
        target_b: [B, 4, T] Ground truth control tokens
        vq_loss: Information bottleneck loss from VC Encoder (if in VC mode)
        log_durations: [B, L] Predicted log-durations (TTS mode)
        dur_target: [B, L] Ground truth durations (TTS mode)
        phoneme_mask: [B, L] Mask for padded phonemes
        adv_logits: [B, T, num_speakers] Adversarial logits from VoiceStateEncoder
        speaker_labels: [B] Ground truth speaker labels
        lambda_vq: Weight for VQ loss
        lambda_dur: Weight for duration loss
        lambda_adv: Weight for adversarial loss

    Returns:
        Dict with total loss and individual components
    """
    B, n_cb, T, vocab_a = logits_a.shape
    _, n_slots, _, vocab_b = logits_b.shape

    # 1. Acoustic Loss (A_t)
    loss_a = 0.0
    for i in range(n_cb):
        loss_a += F.cross_entropy(
            logits_a[:, i, :, :].reshape(-1, vocab_a),
            target_a[:, i, :].reshape(-1),
            ignore_index=-1,
        )
    loss_a = loss_a / n_cb

    # 2. Control Loss (B_t)
    loss_b = 0.0
    for i in range(n_slots):
        loss_b += F.cross_entropy(
            logits_b[:, i, :, :].reshape(-1, vocab_b),
            target_b[:, i, :].reshape(-1),
            ignore_index=-1,
        )
    loss_b = loss_b / n_slots

    # 3. Total Loss
    total_loss = loss_a + loss_b

    # 4. Optional Losses
    components = {
        "loss_a": loss_a,
        "loss_b": loss_b,
    }

    if vq_loss is not None:
        total_loss = total_loss + lambda_vq * vq_loss
        components["loss_vq"] = vq_loss

    if log_durations is not None and dur_target is not None:
        loss_dur = duration_loss(log_durations, dur_target, phoneme_mask)
        total_loss = total_loss + lambda_dur * loss_dur
        components["loss_dur"] = loss_dur

    if adv_logits is not None and speaker_labels is not None:
        # Cross-entropy for adversarial training
        # speaker_labels is [B], expand to [B, T]
        B_adv, T_adv, n_spk = adv_logits.shape
        labels_expanded = speaker_labels.unsqueeze(1).expand(B_adv, T_adv)
        loss_adv = F.cross_entropy(
            adv_logits.reshape(-1, n_spk), labels_expanded.reshape(-1)
        )
        total_loss = total_loss + lambda_adv * loss_adv
        components["loss_adv"] = loss_adv

    components["loss"] = total_loss
    return components
