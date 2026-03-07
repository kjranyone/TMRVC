import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from .models import DisentangledUCLM, uclm_loss

class UCLMTrainer:
    """Trainer for Disentangled UCLM (TTS & VC).

    Supports:
        - Multi-task training (randomly switch between TTS and VC)
        - Classifier-Free Guidance (CFG) dropout
        - Adversarial disentanglement loss
        - Duration prediction loss (TTS legacy mode)
        - Pointer-based TTS training (v3 pointer mode)
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
        pointer_target_source: str = "heuristic_bootstrap",
        legacy_duration_loss_weight: float = 0.0,
        bootstrap_alignment_required: bool = False,
        voice_state_loss_weight: float = 0.0,
        delta_voice_state_loss_weight: float = 0.0,
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
            cum_dur = torch.cumsum(curr_durations, dim=0)
            
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
                end_idx = min(start_idx + dur, target_length)
                # progress goes from 1/dur to 1.0
                p = torch.linspace(1.0 / dur, 1.0, steps=dur, device=device)
                progress_targets[b, start_idx:end_idx] = p[:end_idx-start_idx]
                start_idx = end_idx
                if start_idx >= target_length:
                    break
                    
        return advance_targets, progress_targets

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
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
            
        # Classifier-Free Guidance Dropout (15% chance to drop conditions)
        cfg_scale = 1.0
        if random.random() < 0.15:
            explicit_state = torch.zeros_like(explicit_state)
            ssl_state = torch.zeros_like(ssl_state)
            speaker_embed = torch.zeros_like(speaker_embed)
            
        # Multi-task sampling: decide vc vs tts
        mode = "vc"
        has_phonemes = batch.get("phoneme_ids") is not None
        has_durations = batch.get("durations") is not None
        # In pointer mode, TTS only requires phoneme_ids (no durations needed)
        # In legacy_duration mode, TTS requires both phoneme_ids and durations
        can_do_tts = has_phonemes and (
            self.tts_mode == "pointer" or has_durations
        )
        if can_do_tts and random.random() < self.tts_prob:
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
                cfg_scale=cfg_scale
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
            # Validate pointer target source
            if self.tts_mode == "pointer":
                if self.pointer_target_source == "legacy_duration" and not has_durations:
                    raise RuntimeError(
                        "pointer_target_source='legacy_duration' requires durations in batch"
                    )
                if self.pointer_target_source == "heuristic_bootstrap":
                    pass  # uniform fallback allowed
                elif self.pointer_target_source in ("mas", "ctc"):
                    pass  # online alignment (placeholder)

            # v3 pointer-based TTS training (only supported mode)
            phonemes = batch["phoneme_ids"].to(self.device)
            phoneme_lens = batch["phoneme_lens"].to(self.device)
            language_ids = batch["language_id"].to(self.device)
            target_length = target_a.shape[-1]

            # Optional expressive conditioning fields
            dialogue_context = batch.get("dialogue_context")
            if dialogue_context is not None:
                dialogue_context = dialogue_context.to(self.device)
            acting_intent = batch.get("acting_intent")
            if acting_intent is not None:
                acting_intent = acting_intent.to(self.device)
            prosody_targets = batch.get("prosody_targets")
            if prosody_targets is not None:
                prosody_targets = prosody_targets.to(self.device)

            out = self.model.forward_tts_pointer(
                phoneme_ids=phonemes,
                language_ids=language_ids,
                pointer_state=None,
                speaker_embed=speaker_embed,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                target_b=target_b,
                target_length=target_length,
                f0_condition=f0_condition,
                cfg_scale=cfg_scale,
                dialogue_context=dialogue_context,
                acting_intent=acting_intent,
                prosody_latent=prosody_targets,
            )

            # Generate pointer targets based on configured source
            if self.pointer_target_source == "legacy_duration" and has_durations:
                durations = batch["durations"].to(self.device)
                adv_targets, prog_targets = self._generate_pointer_targets(durations, target_length)
            elif self.pointer_target_source == "heuristic_bootstrap":
                L = phonemes.shape[1]
                dummy_durations = torch.full((phonemes.shape[0], L), target_length // L, device=self.device)
                adv_targets, prog_targets = self._generate_pointer_targets(dummy_durations, target_length)
            elif self.pointer_target_source in ("mas", "ctc"):
                # Online alignment - placeholder; falls back to heuristic
                L = phonemes.shape[1]
                dummy_durations = torch.full((phonemes.shape[0], L), target_length // L, device=self.device)
                adv_targets, prog_targets = self._generate_pointer_targets(dummy_durations, target_length)
            else:
                raise RuntimeError(f"Unknown pointer_target_source: {self.pointer_target_source}")

            # Context group IDs for anti-collapse diversity loss
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
                pointer_logits=out.get("pointer_logits"),
                advance_targets=adv_targets,
                progress_delta=out.get("progress_delta"),
                progress_targets=prog_targets,
                hidden_states=out.get("hidden_states"),
                context_groups=context_groups,
                lambda_pointer=self.pointer_loss_weight,
                lambda_progress=self.progress_loss_weight,
            )
            
        # Optimization
        losses["loss"].backward()
        self.optimizer.step()
        
        res = {k: v.item() for k, v in losses.items()}
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
