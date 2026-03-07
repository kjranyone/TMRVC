import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .models import DisentangledUCLM
from .models.uclm_loss import uclm_loss, monotonic_alignment_search


class CurriculumScheduler:
    """3-stage training curriculum for UCLM v3.

    Stage 1 (Base LM): Next-token codec prediction, VC-focused. Steps 0 to stage2_start.
    Stage 2 (Alignment & Pointer): Text-aligned pointer training. Steps stage2_start to stage3_start.
    Stage 3 (Drama & Dialogue): Expressive finetuning with CFG. Steps stage3_start onwards.
    """

    def __init__(self, stage2_start: int = 5000, stage3_start: int = 15000):
        self.stage2_start = stage2_start
        self.stage3_start = stage3_start

    def get_stage(self, step: int) -> int:
        if step < self.stage2_start:
            return 1
        if step < self.stage3_start:
            return 2
        return 3

    def get_config(self, step: int) -> dict:
        """Return training hyper-parameter overrides for the current step."""
        stage = self.get_stage(step)
        if stage == 1:
            return {
                "tts_prob": 0.2,
                "pointer_loss_weight": 0.0,
            }
        if stage == 2:
            return {
                "tts_prob": 0.5,
                "pointer_loss_weight": 0.5,
            }
        # stage 3
        return {
            "tts_prob": 0.7,
            "pointer_loss_weight": 0.5,
            "conditioning_dropout_prob": 0.15,
        }


class UCLMTrainer:
    """Trainer for Disentangled UCLM (TTS & VC).

    Supports:
        - Multi-task training (randomly switch between TTS and VC)
        - Classifier-Free Guidance (CFG) dropout
        - Adversarial disentanglement loss
        - Duration prediction loss (TTS legacy mode)
        - Pointer-based TTS training (v3 pointer mode)
        - Monotonic Alignment Search (MAS) for pointer targets
    """
    def __init__(
        self,
        model: DisentangledUCLM,
        optimizer: torch.optim.Optimizer,
        device: str = "cuda",
        tts_prob: float = 0.5,
        tts_mode: str = "pointer",
        pointer_loss_weight: float = 0.5,
        progress_loss_weight: float = 0.2,
        alignment_loss_type: str = "none",
        pointer_target_source: str = "mas",
        legacy_duration_loss_weight: float = 0.0,
        bootstrap_alignment_required: bool = False,
        voice_state_loss_weight: float = 0.0,
        delta_voice_state_loss_weight: float = 0.0,
        conditioning_dropout_prob: float = 0.15,
        curriculum: CurriculumScheduler | None = None,
        prompt_sampling_prob: float = 0.0,
        pointer_aux_alignment_warmup_steps: int = 2000,
        pointer_aux_alignment_anneal_steps: int = 5000,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.tts_prob = tts_prob
        self.tts_mode = tts_mode
        self.pointer_loss_weight = pointer_loss_weight
        self.progress_loss_weight = progress_loss_weight
        self.alignment_loss_type = alignment_loss_type
        self.pointer_target_source = pointer_target_source
        self.legacy_duration_loss_weight = legacy_duration_loss_weight
        self.bootstrap_alignment_required = bootstrap_alignment_required
        self.voice_state_loss_weight = voice_state_loss_weight
        self.delta_voice_state_loss_weight = delta_voice_state_loss_weight
        self.conditioning_dropout_prob = conditioning_dropout_prob
        self.curriculum = curriculum
        self.prompt_sampling_prob = prompt_sampling_prob
        self.pointer_aux_alignment_warmup_steps = pointer_aux_alignment_warmup_steps
        self.pointer_aux_alignment_anneal_steps = pointer_aux_alignment_anneal_steps
        self._global_step = 0
        
    def _generate_pointer_targets(
        self, durations: torch.Tensor, target_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate advance and progress targets from phoneme durations.

        Args:
            durations: [B, L] durations per phoneme.
            target_length: total number of acoustic frames (T).

        Returns:
            advance_targets: [B, T] binary (1 at phoneme boundaries).
            progress_targets: [B, T] fractional progress (0-1).
        """
        B, L = durations.shape
        device = durations.device
        advance_targets = torch.zeros(B, target_length, device=device)
        progress_targets = torch.zeros(B, target_length, device=device)

        for b in range(B):
            curr_durations = durations[b]
            cum_dur = torch.cumsum(curr_durations, dim=0).long()
            
            # Boundary indices (where phoneme changes)
            boundaries = cum_dur - 1
            # Filter valid boundaries within target_length
            valid_boundaries = boundaries[boundaries < target_length]
            advance_targets[b, valid_boundaries] = 1.0
            
            # Calculate progress within each phoneme
            start_idx = 0
            for i, dur in enumerate(curr_durations):
                if dur <= 0:
                    continue
                dur_int = int(dur.item())
                end_idx = min(start_idx + dur_int, target_length)
                actual_dur = end_idx - start_idx
                if actual_dur <= 0:
                    continue
                # progress goes from 1/dur to 1.0
                p = torch.linspace(1.0 / dur_int, 1.0, steps=actual_dur, device=device)
                progress_targets[b, start_idx:end_idx] = p
                start_idx = end_idx
                if start_idx >= target_length:
                    break
                    
        return advance_targets, progress_targets

    def _generate_pointer_targets_from_path(
        self, path: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert MAS binary path [B, L, T] into pointer targets."""
        B, L, T = path.shape
        device = path.device
        
        # durations: [B, L]
        durations = path.sum(dim=2)
        return self._generate_pointer_targets(durations, T)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Apply curriculum overrides for this step
        tts_prob = self.tts_prob
        pointer_loss_weight = self.pointer_loss_weight
        conditioning_dropout_prob = self.conditioning_dropout_prob
        if self.curriculum is not None:
            overrides = self.curriculum.get_config(self._global_step)
            tts_prob = overrides.get("tts_prob", tts_prob)
            pointer_loss_weight = overrides.get("pointer_loss_weight", pointer_loss_weight)
            conditioning_dropout_prob = overrides.get(
                "conditioning_dropout_prob", conditioning_dropout_prob
            )
        
        # Auxiliary alignment weight annealing
        aux_alignment_weight = 0.0
        if self.pointer_target_source == "mas":
            if self._global_step < self.pointer_aux_alignment_warmup_steps:
                aux_alignment_weight = 1.0
            elif self._global_step < (self.pointer_aux_alignment_warmup_steps + self.pointer_aux_alignment_anneal_steps):
                # Linear anneal to 0
                elapsed = self._global_step - self.pointer_aux_alignment_warmup_steps
                aux_alignment_weight = 1.0 - (elapsed / self.pointer_aux_alignment_anneal_steps)
        
        self._global_step += 1

        # Move common tensors to device
        target_a = batch["target_a"].to(self.device)
        target_b = batch["target_b"].to(self.device)
        explicit_state = batch["explicit_state"].to(self.device)
        ssl_state = batch["ssl_state"].to(self.device)
        speaker_embed = batch["speaker_embed"].to(self.device)
        speaker_labels = batch["speaker_id"].to(self.device)
        f0_condition = batch.get("f0_condition")
        if f0_condition is not None:
            f0_condition = f0_condition.to(self.device)

        # Classifier-Free Guidance Dropout
        if random.random() < conditioning_dropout_prob:
            explicit_state = torch.zeros_like(explicit_state)
            ssl_state = torch.zeros_like(ssl_state)
            speaker_embed = torch.zeros_like(speaker_embed)
            if batch.get("dialogue_context") is not None:
                batch["dialogue_context"] = torch.zeros_like(batch["dialogue_context"])
            if batch.get("acting_intent") is not None:
                batch["acting_intent"] = torch.zeros_like(batch["acting_intent"])
            if batch.get("delta_voice_state") is not None:
                batch["delta_voice_state"] = torch.zeros_like(batch["delta_voice_state"])
            if batch.get("prosody_targets") is not None:
                batch["prosody_targets"] = torch.zeros_like(batch["prosody_targets"])

        # Multi-task sampling: decide vc vs tts
        mode = "vc"
        has_phonemes = batch.get("phoneme_ids") is not None
        has_durations = batch.get("durations") is not None
        can_do_tts = has_phonemes and (self.tts_mode == "pointer" or has_durations)
        if can_do_tts and random.random() < tts_prob:
            mode = "tts"

        if mode == "vc":
            source_a_t = batch["source_a_t"].to(self.device)
            source_mask = (target_a[:, 0, :] != -1)
            out = self.model.forward_vc(
                source_a_t=source_a_t,
                target_b=target_b,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                speaker_embed=speaker_embed,
                source_mask=source_mask,
                f0_condition=f0_condition,
            )

            losses = uclm_loss(
                logits_a=out["logits_a"],
                logits_b=out["logits_b"],
                target_a=target_a,
                target_b=target_b,
                vq_loss=out.get("vq_loss"),
                adv_logits=out.get("adv_logits"),
                speaker_labels=speaker_labels
            )

        else:
            phonemes = batch["phoneme_ids"].to(self.device)
            language_ids = batch["language_id"].to(self.device)
            target_length = target_a.shape[-1]

            dialogue_context = batch.get("dialogue_context")
            if dialogue_context is not None:
                dialogue_context = dialogue_context.to(self.device)
            acting_intent = batch.get("acting_intent")
            if acting_intent is not None:
                acting_intent = acting_intent.to(self.device)
            prosody_targets = batch.get("prosody_targets")
            if prosody_targets is not None:
                prosody_targets = prosody_targets.to(self.device)
            delta_voice_state = batch.get("delta_voice_state")
            if delta_voice_state is not None:
                delta_voice_state = delta_voice_state.to(self.device)

            out = self.model.forward_tts_pointer(
                phoneme_ids=phonemes,
                language_ids=language_ids,
                pointer_state=None,
                speaker_embed=speaker_embed,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                target_a=target_a,
                target_b=target_b,
                target_length=target_length,
                f0_condition=f0_condition,
                dialogue_context=dialogue_context,
                acting_intent=acting_intent,
                prosody_latent=prosody_targets,
                delta_voice_state=delta_voice_state,
            )

            # Alignment Target Generation
            if self.pointer_target_source == "mas":
                # Compute log-likelihood for MAS
                # Use phoneme encoder output vs transformer hidden states
                # (Detached hidden states to avoid circular gradients if needed, 
                # but standard MAS uses them together)
                with torch.no_grad():
                    # x_text: [B, L, D]
                    phoneme_lens = (phonemes != 0).sum(dim=1)
                    x_text = self.model.text_encoder(phonemes, language_ids, phoneme_lens).transpose(1, 2)
                    # x_acoustic: [B, T, D]
                    x_acoustic = out["hidden_states"]
                    
                    # Log-probs based on negative distance
                    # [B, L, T]
                    dist = torch.cdist(x_text, x_acoustic)
                    log_probs = -0.5 * (dist ** 2)
                    
                    path = monotonic_alignment_search(log_probs)
                    adv_targets, prog_targets = self._generate_pointer_targets_from_path(path)
            elif self.pointer_target_source == "legacy_duration" and has_durations:
                durations = batch["durations"].to(self.device)
                adv_targets, prog_targets = self._generate_pointer_targets(durations, target_length)
            else:
                # Uniform fallback
                L = phonemes.shape[1]
                dummy_durations = torch.full((phonemes.shape[0], L), target_length // L, device=self.device)
                adv_targets, prog_targets = self._generate_pointer_targets(dummy_durations, target_length)

            context_groups = batch.get("context_groups")
            if context_groups is not None:
                context_groups = context_groups.to(self.device)

            losses = uclm_loss(
                logits_a=out["logits_a"],
                logits_b=out["logits_b"],
                target_a=target_a,
                target_b=target_b,
                adv_logits=out.get("adv_logits"),
                speaker_labels=speaker_labels,
                pointer_logits=out.get("advance_logit"),
                advance_targets=adv_targets,
                progress_delta=out.get("progress_delta"),
                progress_targets=prog_targets,
                hidden_states=out.get("hidden_states"),
                context_groups=context_groups,
                lambda_pointer=pointer_loss_weight,
                lambda_progress=self.progress_loss_weight,
                # Flow-matching loss for ProsodyPredictor
                prosody_predictor=self.model.prosody_predictor,
                phoneme_features=self.model.text_encoder(phonemes, language_ids, (phonemes != 0).sum(dim=1)).transpose(1, 2),
                target_prosody=prosody_targets,
                prosody_dialogue_context=dialogue_context,
                prosody_speaker_embed=speaker_embed,
                lambda_prosody=1.0 if prosody_targets is not None else 0.0,
            )
            
        # Optimization
        losses["loss"].backward()
        self.optimizer.step()
        
        res = {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}
        res["mode"] = 1 if mode == "tts" else 0
        return res

    def train_epoch(self, dataloader: DataLoader):
        total_loss = 0
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            pbar.set_postfix({"loss": metrics["loss"], "mode": "TTS" if metrics["mode"] else "VC"})
        return total_loss / len(dataloader)
