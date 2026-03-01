import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper for GRL."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class AdversarialClassifier(nn.Module):
    """Simple classifier for adversarial disentanglement."""

    def __init__(self, d_input: int, n_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DisentanglementLoss(nn.Module):
    """GRL-based adversarial losses for disentanglement.

    - A_t should not contain emotion info
    - B_t should not contain phoneme/speaker info
    """

    def __init__(
        self,
        d_acoustic: int = 512,
        d_control: int = 128,
        n_emotions: int = 12,
        n_phonemes: int = 200,
        n_speakers: int = 100,
    ):
        super().__init__()

        self.grl_acoustic = GradientReversalLayer()
        self.grl_control = GradientReversalLayer()

        self.emotion_classifier = AdversarialClassifier(d_acoustic, n_emotions)
        self.phoneme_classifier = AdversarialClassifier(d_control, n_phonemes)
        self.speaker_classifier = AdversarialClassifier(d_control, n_speakers)

    def forward(
        self,
        acoustic_features: torch.Tensor,
        control_features: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
        phoneme_labels: Optional[torch.Tensor] = None,
        speaker_labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            acoustic_features: [B, T, d_acoustic] from A_t
            control_features: [B, T, d_control] from B_t
            emotion_labels: [B] or [B, T] emotion class indices
            phoneme_labels: [B, T] phoneme indices
            speaker_labels: [B] speaker indices

        Returns:
            Dict with adversarial losses
        """
        losses = {}
        total_loss = 0.0

        reversed_acoustic = self.grl_acoustic(acoustic_features)
        reversed_control = self.grl_control(control_features)

        if emotion_labels is not None:
            emotion_logits = self.emotion_classifier(reversed_acoustic)
            if emotion_labels.dim() == 1:
                emotion_logits = emotion_logits.mean(dim=1)
                emotion_labels_expanded = emotion_labels
            else:
                emotion_labels_expanded = emotion_labels

            loss_emotion = F.cross_entropy(
                emotion_logits.view(-1, emotion_logits.size(-1)),
                emotion_labels_expanded.view(-1),
            )
            losses["loss_adv_emotion"] = loss_emotion
            total_loss = total_loss + loss_emotion

        if phoneme_labels is not None:
            phoneme_logits = self.phoneme_classifier(reversed_control)
            loss_phoneme = F.cross_entropy(
                phoneme_logits.view(-1, phoneme_logits.size(-1)),
                phoneme_labels.view(-1),
                ignore_index=-1,
            )
            losses["loss_adv_phoneme"] = loss_phoneme
            total_loss = total_loss + loss_phoneme

        if speaker_labels is not None:
            speaker_logits = self.speaker_classifier(reversed_control.mean(dim=1))
            loss_speaker = F.cross_entropy(speaker_logits, speaker_labels)
            losses["loss_adv_speaker"] = loss_speaker
            total_loss = total_loss + loss_speaker

        losses["loss_disentangle"] = total_loss
        return losses


def orthogonality_loss(
    acoustic_proj: torch.Tensor,
    control_proj: torch.Tensor,
) -> torch.Tensor:
    """Orthogonality loss between acoustic and control projections.

    Forces the two streams to be orthogonal in the latent space.
    """
    B, T, D = acoustic_proj.shape

    acoustic_flat = acoustic_proj.view(-1, D)
    control_flat = control_proj.view(-1, D)

    acoustic_norm = F.normalize(acoustic_flat, dim=1)
    control_norm = F.normalize(control_flat, dim=1)

    cosine_sim = torch.matmul(acoustic_norm, control_norm.t())
    identity = torch.eye(cosine_sim.size(0), device=cosine_sim.device)

    return F.mse_loss(cosine_sim * (1 - identity), torch.zeros_like(cosine_sim))


def transition_smoothness_loss(acoustic_tokens: torch.Tensor) -> torch.Tensor:
    """Penalize abrupt jumps in A_t tokens.

    Args:
        acoustic_tokens: [B, 8, T] discrete acoustic tokens
    """
    diff = acoustic_tokens[:, :, 1:] - acoustic_tokens[:, :, :-1]
    jump_penalty = (diff != 0).float().mean()
    return jump_penalty


def breath_energy_coupling_loss(
    control_tokens: torch.Tensor,
    energy: torch.Tensor,
) -> torch.Tensor:
    """Ensure breath events correlate with high-frequency energy.

    Args:
        control_tokens: [B, 4, T] with type_id in slot 1
        energy: [B, T] high-frequency energy
    """
    type_ids = control_tokens[:, 1, :]
    breath_mask = (type_ids == 11).float()

    energy_mean = energy.mean(dim=1, keepdim=True)
    energy_high = (energy > energy_mean).float()

    coupling = (breath_mask * energy_high).sum() / (breath_mask.sum() + 1e-8)

    target_coupling = 0.7
    return F.mse_loss(coupling, torch.tensor(target_coupling, device=energy.device))


def delta_state_consistency_loss(
    delta_voice_state: torch.Tensor,
    acoustic_diff: torch.Tensor,
) -> torch.Tensor:
    """Ensure delta_voice_state correlates with acoustic change magnitude.

    Args:
        delta_voice_state: [B, T, 8] voice state changes
        acoustic_diff: [B, T] acoustic feature change magnitude
    """
    delta_magnitude = delta_voice_state.norm(dim=-1)
    delta_magnitude = delta_magnitude / (delta_magnitude.max() + 1e-8)
    acoustic_diff = acoustic_diff / (acoustic_diff.max() + 1e-8)

    return F.mse_loss(delta_magnitude, acoustic_diff)


def long_event_consistency_loss(
    control_tokens: torch.Tensor,
    event_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Ensure start/hold/end consistency for long events.

    Args:
        control_tokens: [B, 4, T] with op_id in slot 0
    """
    op_ids = control_tokens[:, 0, :]

    start_mask = (op_ids == 5).float()
    hold_mask = (op_ids == 6).float()
    end_mask = (op_ids == 7).float()

    start_followed_by_hold = (start_mask[:, :-1] * hold_mask[:, 1:]).sum()
    hold_followed_by_end = (hold_mask[:, :-1] * end_mask[:, 1:]).sum()

    total_starts = start_mask.sum() + 1e-8
    total_holds = hold_mask.sum() + 1e-8

    consistency = (
        start_followed_by_hold / total_starts + hold_followed_by_end / total_holds
    ) / 2

    return 1.0 - consistency


def duration_calibration_loss(
    predicted_dur_bins: torch.Tensor,
    actual_duration_frames: torch.Tensor,
) -> torch.Tensor:
    """Calibrate predicted duration bins with actual durations.

    Args:
        predicted_dur_bins: [B, T] predicted duration bin indices
        actual_duration_frames: [B, T] ground truth duration in frames
    """
    predicted_ms = (predicted_dur_bins - 14 + 1) * 50
    actual_ms = actual_duration_frames * 10

    valid_mask = (predicted_dur_bins >= 14) & (predicted_dur_bins <= 53)
    valid_mask = valid_mask.float()

    diff = (predicted_ms - actual_ms).abs() * valid_mask
    return diff.sum() / (valid_mask.sum() + 1e-8)
