import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pointer_advance_loss(
    pointer_logits: torch.Tensor,
    advance_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy loss for pointer advance decisions.

    Args:
        pointer_logits: [B, T, 1] raw logits for advance probability.
        advance_targets: [B, T] binary targets (1 = advance, 0 = hold).
        mask: [B, T] padding mask (True = ignore).
    """
    logits = pointer_logits.squeeze(-1)  # [B, T]
    if mask is not None:
        valid = ~mask
        logits = logits[valid]
        targets = advance_targets[valid].float()
    else:
        logits = logits.reshape(-1)
        targets = advance_targets.reshape(-1).float()
    if logits.numel() == 0:
        return torch.tensor(0.0, device=pointer_logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets)


def progress_regression_loss(
    progress_delta: torch.Tensor,
    progress_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss for progress within current phoneme.

    Args:
        progress_delta: [B, T, 1] predicted progress (0-1).
        progress_targets: [B, T] target progress values (0-1).
        mask: [B, T] padding mask (True = ignore).
    """
    pred = progress_delta.squeeze(-1)  # [B, T]
    if mask is not None:
        valid = ~mask
        pred = pred[valid]
        targets = progress_targets[valid]
    else:
        pred = pred.reshape(-1)
        targets = progress_targets.reshape(-1)
    if pred.numel() == 0:
        return torch.tensor(0.0, device=progress_delta.device)
    return F.mse_loss(pred, targets)


def boundary_confidence_loss(
    boundary_confidence: torch.Tensor,
    boundary_targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """BCE loss for boundary confidence prediction.

    Args:
        boundary_confidence: [B, T, 1] predicted boundary confidence (after sigmoid).
        boundary_targets: [B, T] target boundary values (0-1, soft targets allowed).
        mask: [B, T] padding mask (True = ignore).
    """
    pred = boundary_confidence.squeeze(-1)  # [B, T]
    if mask is not None:
        valid = ~mask
        pred = pred[valid]
        targets = boundary_targets[valid].float()
    else:
        pred = pred.reshape(-1)
        targets = boundary_targets.reshape(-1).float()
    if pred.numel() == 0:
        return torch.tensor(0.0, device=boundary_confidence.device)
    # boundary_confidence is already after sigmoid, use BCE (not with_logits)
    return F.binary_cross_entropy(pred.clamp(1e-7, 1.0 - 1e-7), targets)


def pointer_progress_loss(
    progress_delta: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss for pointer progress delta. Alias for progress_regression_loss."""
    return progress_regression_loss(progress_delta, target, mask)


def context_diversity_loss(
    hidden_states: torch.Tensor,
    context_groups: torch.Tensor | None = None,
    margin: float = 0.1,
) -> torch.Tensor:
    """Anti-collapse regularizer: same text under different context should not
    produce identical hidden representations.

    When ``context_groups`` is provided (integer group IDs per sample), pairs
    within the same text group but different context are pushed apart by at
    least ``margin`` in cosine distance.

    Args:
        hidden_states: [B, T, D] transformer hidden states.
        context_groups: [B] integer group ids (samples sharing same text get
            the same id).  If None, returns 0.
        margin: minimum cosine distance between different-context pairs.

    Returns:
        Scalar loss (0 if no valid pairs exist).
    """
    if context_groups is None:
        return torch.tensor(0.0, device=hidden_states.device)

    # Pool over time -> [B, D]
    pooled = hidden_states.mean(dim=1)
    pooled = F.normalize(pooled, dim=-1)

    B = pooled.shape[0]
    loss = torch.tensor(0.0, device=pooled.device)
    n_pairs = 0

    for i in range(B):
        for j in range(i + 1, B):
            if context_groups[i] == context_groups[j]:
                cos_sim = (pooled[i] * pooled[j]).sum()
                # Push cosine similarity below (1 - margin)
                pair_loss = F.relu(cos_sim - (1.0 - margin))
                loss = loss + pair_loss
                n_pairs += 1

    if n_pairs > 0:
        loss = loss / n_pairs
    return loss


def context_separation_score(
    hidden_states: torch.Tensor,
    context_groups: torch.Tensor,
) -> float:
    """Compute mean pairwise distance between same-text different-context pairs.

    Higher = more diverse deliveries for same text under different context.
    """
    pooled = F.normalize(hidden_states.mean(dim=1), dim=-1)
    B = pooled.shape[0]
    dists = []
    for i in range(B):
        for j in range(i + 1, B):
            if context_groups[i] == context_groups[j]:
                cos_sim = (pooled[i] * pooled[j]).sum().item()
                dists.append(1.0 - cos_sim)
    return float(np.mean(dists)) if dists else 0.0


def prosody_collapse_score(
    hidden_states: torch.Tensor,
    context_groups: torch.Tensor,
) -> float:
    """Ratio of between-context variance to total variance on same-text samples.

    Higher = less collapse (good). Lower = more collapse (bad).
    """
    pooled = hidden_states.mean(dim=1)  # [B, D]
    total_var = pooled.var(dim=0).mean().item()
    if total_var < 1e-8:
        return 0.0

    unique_groups = context_groups.unique()
    between_var = 0.0
    n_groups = 0
    group_means = []
    for g in unique_groups:
        mask = context_groups == g
        if mask.sum() < 2:
            continue
        group_means.append(pooled[mask].mean(dim=0))
        n_groups += 1

    if n_groups < 2:
        return 0.0

    group_means_t = torch.stack(group_means)
    between_var = group_means_t.var(dim=0).mean().item()
    return between_var / (total_var + 1e-8)


def control_response_score(
    output_durations: list[float],
    control_values: list[float],
) -> float:
    """Monotonic correlation between control sweep and output metric.

    Returns Spearman-like rank correlation (1.0 = perfect monotonic, 0.0 = none).
    """
    if len(output_durations) < 2:
        return 0.0
    n = len(output_durations)
    # Simple rank correlation
    ranks_out = sorted(range(n), key=lambda i: output_durations[i])
    ranks_ctrl = sorted(range(n), key=lambda i: control_values[i])
    d_sq = sum((ranks_out[i] - ranks_ctrl[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def flow_matching_prosody_loss(
    predictor: "torch.nn.Module",
    phoneme_features: torch.Tensor,
    target_prosody: torch.Tensor,
    dialogue_context: torch.Tensor | None = None,
    speaker_embed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute flow-matching prosody loss via the predictor's own method.

    Args:
        predictor: A :class:`ProsodyPredictor` instance (flow-matching based).
        phoneme_features: [B, L, d_model] text encoder output.
        target_prosody: [B, d_prosody] ground-truth prosody latent.
        dialogue_context: [B, D_ctx] optional dialogue context.
        speaker_embed: [B, d_model] optional speaker embedding.
    Returns:
        Scalar MSE loss between predicted and true velocity field.
    """
    return predictor.flow_matching_loss(
        phoneme_features, target_prosody, dialogue_context, speaker_embed
    )


def timbre_prosody_contrastive_loss(
    speaker_embeds: torch.Tensor,
    prosody_latents: torch.Tensor,
    context_groups: torch.Tensor | None = None,
    margin: float = 0.2,
) -> torch.Tensor:
    """Contrastive loss pushing prosody apart for same speaker under different contexts.

    For pairs of samples that share the same speaker (identified via
    ``context_groups``), their prosody latents should be dissimilar — the
    speaker timbre is already captured by the speaker embedding, so prosody
    should encode *different* expressive variation.

    Args:
        speaker_embeds: [B, D_spk] speaker embeddings.
        prosody_latents: [B, D_pro] prosody latent vectors.
        context_groups: [B] integer group ids. Pairs with the same group id
            are considered same-speaker pairs whose prosody should differ.
            If None, returns 0.
        margin: minimum cosine distance between prosody of same-speaker pairs.

    Returns:
        Scalar contrastive loss (0 if no valid pairs).
    """
    if context_groups is None:
        return torch.tensor(0.0, device=prosody_latents.device)

    prosody_norm = F.normalize(prosody_latents, dim=-1)
    B = prosody_norm.shape[0]
    loss = torch.tensor(0.0, device=prosody_norm.device)
    n_pairs = 0

    for i in range(B):
        for j in range(i + 1, B):
            if context_groups[i] == context_groups[j]:
                cos_sim = (prosody_norm[i] * prosody_norm[j]).sum()
                # Push cosine similarity below (1 - margin)
                pair_loss = F.relu(cos_sim - (1.0 - margin))
                loss = loss + pair_loss
                n_pairs += 1

    if n_pairs > 0:
        loss = loss / n_pairs
    return loss


@torch.jit.script
def monotonic_alignment_search(log_probs: torch.Tensor) -> torch.Tensor:
    """Monotonic Alignment Search (MAS) implementation.

    Finds the most likely monotonic path between phonemes and frames.
    Implementation based on VITS (Kim et al., 2021).

    Args:
        log_probs: [B, L, T] log-likelihood matrix where L is phoneme count
            and T is acoustic frame count.

    Returns:
        path: [B, L, T] binary mask representing the optimal path.
    """
    B, L, T = log_probs.shape
    device = log_probs.device
    
    # Calculate cumulative log probabilities using dynamic programming
    # v_prev[i] stores the max log prob to reach state i at current frame t
    v = torch.zeros((B, L, T), device=device)
    
    # Initialize first frame
    v[:, 0, 0] = log_probs[:, 0, 0]
    for i in range(1, L):
        v[:, i, 0] = -1e9  # Impossible to start at phoneme i > 0
        
    for t in range(1, T):
        # State 0: can only come from state 0
        v[:, 0, t] = v[:, 0, t-1] + log_probs[:, 0, t]
        
        # States 1 to L-1
        for i in range(1, L):
            # Monotonic constraint: can come from same state i (hold)
            # or previous state i-1 (advance)
            v_prev_hold = v[:, i, t-1]
            v_prev_adv = v[:, i-1, t-1]
            v[:, i, t] = torch.max(v_prev_hold, v_prev_adv) + log_probs[:, i, t]
            
    # Backtrack to find the optimal path
    path = torch.zeros((B, L, T), device=device)
    curr_phoneme = torch.full((B,), L - 1, dtype=torch.long, device=device)
    
    for t in range(T - 1, -1, -1):
        for b in range(B):
            idx = curr_phoneme[b]
            path[b, idx, t] = 1.0
            
            if t > 0:
                # Decide whether to stay or go back to i-1
                if idx > 0:
                    v_hold = v[b, idx, t-1]
                    v_adv = v[b, idx-1, t-1]
                    if v_adv > v_hold:
                        curr_phoneme[b] -= 1
                        
    return path


def alignment_loss_placeholder(
    phoneme_features: torch.Tensor,
    frame_features: torch.Tensor,
    alignment_type: str = "none",
) -> torch.Tensor:
    """Placeholder for future MAS/CTC alignment loss."""
    if alignment_type == "none":
        return torch.tensor(0.0, device=phoneme_features.device)
    raise NotImplementedError(f"Alignment loss type '{alignment_type}' not yet implemented.")


def voice_state_supervision_loss(
    predicted_state: torch.Tensor,
    target_state: torch.Tensor,
    mask: torch.Tensor | None = None,
    observed_mask: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """MSE loss for voice state prediction with proper masking.

    Missing or low-confidence dimensions are excluded from loss, never
    treated as zero-valued neutral.

    Args:
        predicted_state: [B, T, D] predicted voice state.
        target_state: [B, T, D] target voice state.
        mask: [B, T] padding mask (True = ignore frame entirely).
        observed_mask: [B, T, D] bool — True where dimension has usable evidence.
            When None, all dimensions are assumed observed.
        confidence: [B, T, D] or [B, T, 1] — per-dimension confidence weights.
            When None, uniform confidence of 1.0 is used.
    """
    device = predicted_state.device

    # Per-element squared error
    sq_err = (predicted_state - target_state).pow(2)  # [B, T, D]

    # Build combined weight mask
    weight = torch.ones_like(sq_err)

    if observed_mask is not None:
        weight = weight * observed_mask.float()

    if confidence is not None:
        if confidence.shape[-1] == 1 and sq_err.shape[-1] > 1:
            confidence = confidence.expand_as(sq_err)
        weight = weight * confidence

    if mask is not None:
        # mask: [B, T] True = ignore → expand to [B, T, D]
        frame_mask = (~mask).float().unsqueeze(-1).expand_as(sq_err)
        weight = weight * frame_mask

    total_weight = weight.sum()
    if total_weight < 1e-8:
        return torch.tensor(0.0, device=device)

    return (sq_err * weight).sum() / total_weight


def uclm_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    vq_loss: torch.Tensor | None = None,
    adv_logits: torch.Tensor | None = None,
    speaker_labels: torch.Tensor | None = None,
    pointer_logits: torch.Tensor | None = None,
    advance_targets: torch.Tensor | None = None,
    progress_delta: torch.Tensor | None = None,
    progress_targets: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
    hidden_states: torch.Tensor | None = None,
    context_groups: torch.Tensor | None = None,
    lambda_vq: float = 1.0,
    lambda_adv: float = 0.1,
    lambda_pointer: float = 0.5,
    lambda_progress: float = 0.2,
    lambda_diversity: float = 0.05,
    lambda_voice_state: float = 0.0,
    lambda_delta_voice_state: float = 0.0,
    lambda_prosody: float = 0.0,
    lambda_contrastive: float = 0.0,
    voice_state_pred: torch.Tensor | None = None,
    voice_state_target: torch.Tensor | None = None,
    delta_voice_state_pred: torch.Tensor | None = None,
    delta_voice_state_target: torch.Tensor | None = None,
    prosody_predictor: "torch.nn.Module | None" = None,
    phoneme_features: torch.Tensor | None = None,
    target_prosody: torch.Tensor | None = None,
    prosody_dialogue_context: torch.Tensor | None = None,
    prosody_speaker_embed: torch.Tensor | None = None,
    speaker_embeds: torch.Tensor | None = None,
    prosody_latents: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the multi-task loss for Disentangled UCLM."""
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

    if adv_logits is not None and speaker_labels is not None:
        B_adv, T_adv, n_spk = adv_logits.shape
        labels_expanded = speaker_labels.unsqueeze(1).expand(B_adv, T_adv)
        loss_adv = F.cross_entropy(
            adv_logits.reshape(-1, n_spk), labels_expanded.reshape(-1)
        )
        total_loss = total_loss + lambda_adv * loss_adv
        components["loss_adv"] = loss_adv

    # Pointer losses
    if pointer_logits is not None and advance_targets is not None:
        loss_ptr = pointer_advance_loss(pointer_logits, advance_targets, frame_mask)
        total_loss = total_loss + lambda_pointer * loss_ptr
        components["loss_pointer"] = loss_ptr

    if progress_delta is not None and progress_targets is not None:
        loss_prog = progress_regression_loss(progress_delta, progress_targets, frame_mask)
        total_loss = total_loss + lambda_progress * loss_prog
        components["loss_progress"] = loss_prog

    # Anti-collapse diversity regularizer
    if hidden_states is not None and context_groups is not None:
        loss_div = context_diversity_loss(hidden_states, context_groups)
        total_loss = total_loss + lambda_diversity * loss_div
        components["loss_diversity"] = loss_div

    # Voice state supervision losses
    if lambda_voice_state > 0 and voice_state_pred is not None and voice_state_target is not None:
        loss_vs = voice_state_supervision_loss(voice_state_pred, voice_state_target, frame_mask)
        total_loss = total_loss + lambda_voice_state * loss_vs
        components["loss_voice_state"] = loss_vs

    if lambda_delta_voice_state > 0 and delta_voice_state_pred is not None and delta_voice_state_target is not None:
        loss_dvs = voice_state_supervision_loss(delta_voice_state_pred, delta_voice_state_target, frame_mask)
        total_loss = total_loss + lambda_delta_voice_state * loss_dvs
        components["loss_delta_voice_state"] = loss_dvs

    # Flow-matching prosody loss
    if lambda_prosody > 0 and prosody_predictor is not None and phoneme_features is not None and target_prosody is not None:
        loss_prosody = flow_matching_prosody_loss(
            prosody_predictor, phoneme_features, target_prosody,
            prosody_dialogue_context, prosody_speaker_embed,
        )
        total_loss = total_loss + lambda_prosody * loss_prosody
        components["loss_prosody"] = loss_prosody

    # Timbre-prosody contrastive loss
    if lambda_contrastive > 0 and speaker_embeds is not None and prosody_latents is not None:
        loss_contrastive = timbre_prosody_contrastive_loss(
            speaker_embeds, prosody_latents, context_groups,
        )
        total_loss = total_loss + lambda_contrastive * loss_contrastive
        components["loss_contrastive"] = loss_contrastive

    components["loss"] = total_loss
    return components
