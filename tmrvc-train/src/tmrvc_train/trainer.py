import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import DisentangledUCLM
from .models.uclm_loss import (
    uclm_loss,
    monotonic_alignment_search,
    voice_state_supervision_loss,
    boundary_confidence_loss,
)
from .models.reference_encoder import ReferenceEncoderFromWaveform

# v4 imports: biological constraints, acting losses, v4 loss composition
from .models.biological_constraints import BiologicalConstraintRegularizer
from .models.acting_latent import ActingLatentEncoder, ActingLatentPredictor
from .models.acting_losses import (
    acting_latent_kl_loss,
    acting_latent_usage_loss,
    disentanglement_loss,
    semantic_alignment_loss,
)
from .v4_loss import V4LossConfig, V4LossResult, compute_v4_total_loss

# Supervision tier weighting (Task 3-1)
TIER_WEIGHTS = {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.1}
# Map from dataset tier string to TIER_WEIGHTS key
_TIER_KEY_MAP = {"tier_a": "A", "tier_b": "B", "tier_c": "C", "tier_d": "D"}


class CurriculumScheduler:
    """3-stage training curriculum for UCLM.

    Stage 1 (Base LM): Next-token codec prediction, VC-focused. Steps 0 to stage2_start.
    Stage 2 (Alignment & Pointer): Text-aligned pointer training. Steps stage2_start to stage3_start.
    Stage 3 (Drama & Dialogue): Expressive finetuning with CFG. Steps stage3_start onwards.

    Anti-forgetting: ``stage3_replay_mix_ratio`` controls the fraction of
    Stage 3 batches that are replaced with Stage 1/2 stability data to
    prevent base-quality regression during drama finetuning.
    """

    def __init__(
        self,
        stage2_start: int = 5000,
        stage3_start: int = 15000,
        stage3_replay_mix_ratio: float = 0.2,
    ):
        self.stage2_start = stage2_start
        self.stage3_start = stage3_start
        if not 0.0 <= stage3_replay_mix_ratio <= 1.0:
            raise ValueError(
                f"stage3_replay_mix_ratio must be in [0, 1], got {stage3_replay_mix_ratio}"
            )
        self.stage3_replay_mix_ratio = stage3_replay_mix_ratio

    def get_stage(self, step: int) -> int:
        if step < self.stage2_start:
            return 1
        if step < self.stage3_start:
            return 2
        return 3

    def should_replay(self, step: int) -> bool:
        """Return True if this step should use stability replay data.

        Only applies during Stage 3.  The caller is responsible for
        substituting a Stage 1/2 batch when this returns True.
        """
        if self.get_stage(step) < 3:
            return False
        return random.random() < self.stage3_replay_mix_ratio

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
        - Pointer-based TTS training
        - Monotonic Alignment Search (MAS) for pointer targets
        - 3-stage training curriculum (Base, Align, Drama)
        - Flow-matching prosody loss
        - Voice state supervision with masks and confidence
        - Anti-collapse diagnostics (pointer stall rate, advance entropy, voice_state variance)
        - Boundary confidence loss
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
        boundary_confidence_loss_weight: float = 0.1,
        alignment_loss_type: str = "none",
        pointer_target_source: str = "heuristic_bootstrap",
        pointer_supervision_mode: str = "latent_only",
        bootstrap_alignment_required: bool = False,
        voice_state_loss_weight: float = 0.0,
        delta_voice_state_loss_weight: float = 0.0,
        voice_state_confidence_floor: float = 0.5,
        conditioning_dropout_prob: float = 0.15,
        prosody_loss_weight: float = 1.0,
        curriculum: CurriculumScheduler | None = None,
        prompt_sampling_prob: float = 0.0,
        pointer_aux_alignment_warmup_steps: int = 2000,
        pointer_aux_alignment_anneal_steps: int = 5000,
        pointer_hardening_start_step: int = 10000,
        pointer_hardening_ramp_steps: int = 5000,
        cfg_distillation_weight: float = 0.0,
        cfg_distillation_scale_range: tuple[float, float] = (1.0, 3.0),
        cfg_distillation_temperature: float = 2.0,
        cfg_scale_default: float = 2.0,
        reference_encoder: ReferenceEncoderFromWaveform | None = None,
        training_stage: str = "base",
        # v4 Phase 3 parameters
        v4_loss_config: V4LossConfig | None = None,
        enable_v4_losses: bool = False,
        bio_constraint_weight: float = 1.0,
        acting_latent_encoder: ActingLatentEncoder | None = None,
        acting_latent_predictor: ActingLatentPredictor | None = None,
        speaker_consistency_weight: float = 0.5,
        prosody_prediction_weight: float = 0.5,
        semantic_alignment_weight: float = 0.5,
        acting_kl_weight: float = 0.01,
        disentanglement_weight: float = 0.1,
        use_enriched_transcript: bool = False,
        enriched_transcript_prob: float = 0.5,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        # Reference encoder for on-the-fly prosody target extraction
        # from ground-truth audio when prosody_targets.npy is not available.
        if reference_encoder is not None:
            self.reference_encoder: ReferenceEncoderFromWaveform | None = reference_encoder.to(device)
        else:
            self.reference_encoder = None
        self.tts_prob = tts_prob
        self.tts_mode = tts_mode
        self.pointer_loss_weight = pointer_loss_weight
        self.progress_loss_weight = progress_loss_weight
        self.boundary_confidence_loss_weight = boundary_confidence_loss_weight
        self.alignment_loss_type = alignment_loss_type
        self.pointer_target_source = pointer_target_source
        self.pointer_supervision_mode = pointer_supervision_mode
        self.bootstrap_alignment_required = bootstrap_alignment_required
        self.voice_state_loss_weight = voice_state_loss_weight
        self.delta_voice_state_loss_weight = delta_voice_state_loss_weight
        self.voice_state_confidence_floor = voice_state_confidence_floor
        self.conditioning_dropout_prob = conditioning_dropout_prob
        self.prosody_loss_weight = prosody_loss_weight
        self.curriculum = curriculum
        self.prompt_sampling_prob = prompt_sampling_prob
        self.pointer_aux_alignment_warmup_steps = pointer_aux_alignment_warmup_steps
        self.pointer_aux_alignment_anneal_steps = pointer_aux_alignment_anneal_steps
        self.pointer_hardening_start_step = pointer_hardening_start_step
        self.pointer_hardening_ramp_steps = pointer_hardening_ramp_steps
        self.cfg_distillation_weight = cfg_distillation_weight
        self.cfg_distillation_scale_range = cfg_distillation_scale_range
        self.cfg_distillation_temperature = cfg_distillation_temperature
        self.cfg_scale_default = cfg_scale_default
        self.training_stage = training_stage

        # --- v4 Phase 3: Loss composition and bio constraints ---
        self.enable_v4_losses = enable_v4_losses
        self.v4_loss_config = v4_loss_config or V4LossConfig()
        self.bio_constraint_weight = bio_constraint_weight
        self.speaker_consistency_weight = speaker_consistency_weight
        self.prosody_prediction_weight = prosody_prediction_weight
        self.semantic_alignment_weight = semantic_alignment_weight
        self.acting_kl_weight = acting_kl_weight
        self.disentanglement_weight = disentanglement_weight
        self.use_enriched_transcript = use_enriched_transcript
        self.enriched_transcript_prob = enriched_transcript_prob

        # Biological constraint regularizer (Task 3-2)
        self.bio_regularizer: BiologicalConstraintRegularizer | None = None
        self.bio_physical_head: nn.Linear | None = None
        if self.enable_v4_losses:
            self.bio_regularizer = BiologicalConstraintRegularizer().to(device)
            # Projection head: hidden_states [B, T, D_MODEL] -> [B, T, 12]
            from tmrvc_core.constants import D_MODEL as _D_MODEL
            self.bio_physical_head = nn.Linear(_D_MODEL, 12).to(device)

        # Acting latent encoder/predictor (Task 3-3)
        self.acting_latent_encoder = acting_latent_encoder
        if self.acting_latent_encoder is not None:
            self.acting_latent_encoder = self.acting_latent_encoder.to(device)
        self.acting_latent_predictor = acting_latent_predictor
        if self.acting_latent_predictor is not None:
            self.acting_latent_predictor = self.acting_latent_predictor.to(device)

        # T3 fix: add v4 module parameters to the optimizer
        v4_params = []
        if self.bio_regularizer is not None:
            v4_params.extend(self.bio_regularizer.parameters())
        if self.bio_physical_head is not None:
            v4_params.extend(self.bio_physical_head.parameters())
        if self.acting_latent_encoder is not None:
            v4_params.extend(self.acting_latent_encoder.parameters())
        if self.acting_latent_predictor is not None:
            v4_params.extend(self.acting_latent_predictor.parameters())
        if v4_params:
            self.optimizer.add_param_group({"params": v4_params})

        self._global_step = 0
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate training configuration at init time."""
        valid_tts_modes = {"pointer"}
        if self.tts_mode not in valid_tts_modes:
            raise ValueError(f"tts_mode must be one of {valid_tts_modes}, got {self.tts_mode!r}")

        valid_sources = {"none", "heuristic_bootstrap", "bootstrap_projection", "mas", "ctc"}
        if self.pointer_target_source not in valid_sources:
            raise ValueError(
                f"pointer_target_source must be one of {valid_sources}, got {self.pointer_target_source!r}"
            )

        valid_alignment_types = {"none", "aux_mas", "aux_ctc"}
        if self.alignment_loss_type not in valid_alignment_types:
            raise ValueError(
                f"alignment_loss_type must be one of {valid_alignment_types}, got {self.alignment_loss_type!r}"
            )

        valid_supervision_modes = {"latent_only", "supervised"}
        if self.pointer_supervision_mode not in valid_supervision_modes:
            raise ValueError(
                f"pointer_supervision_mode must be one of {valid_supervision_modes}, "
                f"got {self.pointer_supervision_mode!r}"
            )

        # Supervised pointer mode requires a valid target source
        if (
            self.pointer_supervision_mode == "supervised"
            and self.pointer_target_source == "none"
        ):
            raise ValueError(
                "pointer_supervision_mode='supervised' requires a valid pointer_target_source "
                "(not 'none'). Use 'latent_only' for alignment-free training."
            )

    def _get_annealing_phase(self, step: int) -> str:
        """Determine the pointer annealing phase for the given step.

        Returns one of: 'hard_bootstrap', 'soft_transition', 'latent_only'.
        """
        warmup = self.pointer_aux_alignment_warmup_steps
        anneal = self.pointer_aux_alignment_anneal_steps
        if step < warmup:
            return "hard_bootstrap"
        elif step < warmup + anneal:
            return "soft_transition"
        else:
            return "latent_only"

    def _get_bootstrap_mix_ratio(self, step: int) -> float:
        """Compute the mix ratio for bootstrap vs model predictions.

        Returns 1.0 during hard bootstrap, linearly anneals to 0.0 during
        soft transition, and 0.0 during latent-only phase.
        """
        warmup = self.pointer_aux_alignment_warmup_steps
        anneal = self.pointer_aux_alignment_anneal_steps
        if step < warmup:
            return 1.0
        elif step < warmup + anneal:
            progress = (step - warmup) / max(anneal, 1)
            return 1.0 - progress
        else:
            return 0.0

    def _generate_pointer_targets(
        self, durations: torch.Tensor, target_length: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate advance, progress, frame_offset, and position_indices.

        Args:
            durations: [B, L] durations per phoneme.
            target_length: total number of acoustic frames (T).

        Returns:
            advance_targets: [B, T] binary.
            progress_targets: [B, T] velocity field.
            frame_offsets: [B, T] temporal offset within current phoneme.
            position_indices: [B, T] semantic phoneme index.
        """
        B, L = durations.shape
        device = durations.device
        advance_targets = torch.zeros(B, target_length, device=device)
        progress_targets = torch.zeros(B, target_length, device=device)
        frame_offsets = torch.zeros(B, target_length, device=device)
        position_indices = torch.zeros(B, target_length, device=device).long()

        for b in range(B):
            curr_durations = durations[b]
            cum_dur = torch.cumsum(curr_durations, dim=0).long()

            # Boundary indices
            boundaries = cum_dur - 1
            valid_boundaries = boundaries[boundaries < target_length]
            advance_targets[b, valid_boundaries] = 1.0

            # Progress and offsets
            start_idx = 0
            for i, dur in enumerate(curr_durations):
                if dur <= 0:
                    continue
                dur_int = int(dur.item())
                end_idx = min(start_idx + dur_int, target_length)
                actual_dur = end_idx - start_idx
                if actual_dur <= 0:
                    continue
                
                # SOTA Theory: Velocity
                velocity = 1.0 / max(1.0, dur.item())
                progress_targets[b, start_idx:end_idx] = velocity
                
                # SOTA: Temporal offsets for Hierarchical RoPE
                offs = torch.arange(actual_dur, device=device)
                frame_offsets[b, start_idx:end_idx] = offs.float()
                
                # Semantic index
                position_indices[b, start_idx:end_idx] = i
                
                start_idx = end_idx
                if start_idx >= target_length:
                    break

        return advance_targets, progress_targets, frame_offsets, position_indices

    def _generate_pointer_targets_from_path(
        self, path: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert MAS binary path [B, L, T] into pointer targets."""
        B, L, T = path.shape
        device = path.device

        # durations: [B, L]
        durations = path.sum(dim=2)
        return self._generate_pointer_targets(durations, T)

    def _generate_boundary_targets_from_advance(
        self, advance_targets: torch.Tensor
    ) -> torch.Tensor:
        """Generate boundary confidence targets from advance targets.

        Boundary confidence should be high near phoneme boundaries.
        Uses a simple Gaussian-smoothed version of advance targets.
        """
        # advance_targets: [B, T], binary
        # Return same shape with values near 1.0 at boundaries, decaying away
        B, T = advance_targets.shape
        boundary_targets = advance_targets.clone()
        # Smooth with a small window to create soft boundary targets
        if T > 3:
            kernel = torch.tensor([0.25, 0.5, 1.0, 0.5, 0.25], device=advance_targets.device)
            kernel = kernel / kernel.sum()
            padded = F.pad(advance_targets.unsqueeze(1), (2, 2), mode="replicate")
            smoothed = F.conv1d(padded, kernel.view(1, 1, -1)).squeeze(1)
            boundary_targets = smoothed.clamp(0.0, 1.0)
        return boundary_targets

    def _compute_pointer_diagnostics(
        self, advance_logit: torch.Tensor, progress_delta: torch.Tensor
    ) -> dict:
        """Compute anti-collapse diagnostics for pointer behavior.

        Returns:
            pointer_stall_rate: fraction of frames where pointer doesn't advance
            advance_entropy: Shannon entropy of advance decisions (higher = more diverse)
            progress_variance: variance of progress predictions across frames
        """
        diagnostics = {}
        with torch.no_grad():
            # Pointer stall rate: fraction of frames with advance prob < 0.5
            advance_prob = torch.sigmoid(advance_logit.squeeze(-1))  # [B, T]
            advances = (advance_prob > 0.5).float()
            stall_rate = 1.0 - advances.mean().item()
            diagnostics["pointer_stall_rate"] = stall_rate

            # Advance entropy: -p*log(p) - (1-p)*log(1-p) averaged across all frames
            p = advance_prob.clamp(1e-7, 1.0 - 1e-7)
            entropy = -(p * p.log() + (1 - p) * (1 - p).log())
            diagnostics["advance_entropy"] = entropy.mean().item()

            # Progress variance across frames
            prog = progress_delta.squeeze(-1)  # [B, T]
            diagnostics["progress_variance"] = prog.var().item()

        return diagnostics

    def _compute_voice_state_diagnostics(
        self, hidden_states: torch.Tensor
    ) -> dict:
        """Compute voice_state variance across utterances for anti-collapse monitoring."""
        diagnostics = {}
        with torch.no_grad():
            # Use first 12 dims as voice state proxy
            vs_proxy = hidden_states[:, :, :12]  # [B, T, 12]
            # Per-utterance mean
            vs_means = vs_proxy.mean(dim=1)  # [B, 12]
            # Variance across utterances
            if vs_means.shape[0] > 1:
                diagnostics["voice_state_between_utt_variance"] = vs_means.var(dim=0).mean().item()
            else:
                diagnostics["voice_state_between_utt_variance"] = 0.0
        return diagnostics

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        current_step = self._global_step
        self._global_step += 1

        # Stage 3 anti-forgetting replay: override drama config with
        # Stage 1/2 stability settings when the scheduler says so.
        is_replay = (
            self.curriculum is not None
            and self.curriculum.should_replay(current_step)
        )

        # Apply curriculum overrides for this step
        tts_prob = self.tts_prob
        pointer_loss_weight = self.pointer_loss_weight
        conditioning_dropout_prob = self.conditioning_dropout_prob
        if self.curriculum is not None:
            if is_replay:
                # Use Stage 2 config to maintain base quality
                overrides = self.curriculum.get_config(self.curriculum.stage2_start)
            else:
                overrides = self.curriculum.get_config(current_step)
            tts_prob = overrides.get("tts_prob", tts_prob)
            pointer_loss_weight = overrides.get("pointer_loss_weight", pointer_loss_weight)
            conditioning_dropout_prob = overrides.get(
                "conditioning_dropout_prob", conditioning_dropout_prob
            )

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

        # --- Few-Shot Prompt Encoding (Worker 02 Task 14) ---
        prompt_kv_cache = None
        prompt_codec_tokens = batch.get("prompt_codec_tokens")
        prompt_vq_loss = None
        if prompt_codec_tokens is not None:
            prompt_codec_tokens = prompt_codec_tokens.to(self.device)
            # Encode prompt audio evidence into KV cache and refined embedding
            _prompt_result = self.model.encode_speaker_prompt(
                prompt_codec_tokens=prompt_codec_tokens,
                speaker_embed=speaker_embed,
            )
            # encode_speaker_prompt returns (refined_embed, summary_tokens, vq_loss, indices)
            speaker_embed = _prompt_result[0]
            prompt_kv_cache = _prompt_result[1]
            prompt_vq_loss = _prompt_result[2]

        # Classifier-Free Guidance Dropout (Worker 01 frozen contract)
        cfg_dropped = random.random() < conditioning_dropout_prob
        if cfg_dropped:
            masked = DisentangledUCLM.apply_cfg_unconditional_mask(
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                speaker_embed=speaker_embed,
                dialogue_context=batch.get("dialogue_context"),
                acting_intent=batch.get("acting_intent"),
                delta_voice_state=batch.get("delta_voice_state"),
                prosody_latent=batch.get("prosody_targets"),
                prompt_kv_cache=prompt_kv_cache,
            )
            explicit_state = masked["explicit_state"]
            ssl_state = masked["ssl_state"]
            speaker_embed = masked["speaker_embed"]
            prompt_kv_cache = masked["prompt_kv_cache"]
            if masked["dialogue_context"] is not None:
                batch["dialogue_context"] = masked["dialogue_context"]
            if masked["acting_intent"] is not None:
                batch["acting_intent"] = masked["acting_intent"]
            if masked["delta_voice_state"] is not None:
                batch["delta_voice_state"] = masked["delta_voice_state"]
            if masked["prosody_latent"] is not None:
                batch["prosody_targets"] = masked["prosody_latent"]

        # Multi-task sampling: decide vc vs tts
        mode = "vc"
        has_phonemes = batch.get("phoneme_ids") is not None
        can_do_tts = has_phonemes
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
            # Add prompt VQ loss if present
            if prompt_vq_loss is not None:
                losses["loss"] = losses["loss"] + prompt_vq_loss
                losses["loss_prompt_vq"] = prompt_vq_loss
        else:
            # TTS path
            phonemes = batch["phoneme_ids"].to(self.device)
            # --- v4: enriched transcript routing ---
            if (self.use_enriched_transcript
                    and batch.get("enriched_phoneme_ids") is not None
                    and isinstance(batch["enriched_phoneme_ids"], torch.Tensor)):
                enriched = batch["enriched_phoneme_ids"].to(self.device)
                use_flags = batch.get("use_enriched", [False] * phonemes.shape[0])
                mask = (torch.tensor(use_flags, dtype=torch.bool, device=self.device)
                        if isinstance(use_flags, list)
                        else use_flags.to(self.device).bool())
                # Pad to same length + per-sample routing
                max_len = max(phonemes.shape[1], enriched.shape[1])
                if phonemes.shape[1] < max_len:
                    phonemes = F.pad(phonemes, (0, max_len - phonemes.shape[1]))
                if enriched.shape[1] < max_len:
                    enriched = F.pad(enriched, (0, max_len - enriched.shape[1]))
                phonemes = torch.where(mask.unsqueeze(1).expand_as(phonemes), enriched, phonemes)
            language_ids = batch["language_id"].to(self.device)
            B = phonemes.shape[0]
            target_length = target_a.shape[-1]

            # --- Replay Mix (Worker 02 Task 15) ---
            if is_replay:
                dialogue_context = None
                acting_intent = None
                prosody_targets = None
                delta_voice_state = None
            else:
                dialogue_context = batch.get("dialogue_context")
                if dialogue_context is not None:
                    dialogue_context = dialogue_context.to(self.device)
                acting_intent = batch.get("acting_intent")
                if acting_intent is not None:
                    acting_intent = acting_intent.to(self.device)
                prosody_targets = batch.get("prosody_targets")
                if prosody_targets is not None:
                    prosody_targets = prosody_targets.to(self.device)
                elif self.reference_encoder is not None and batch.get("audio") is not None:
                    with torch.no_grad():
                        audio = batch["audio"].to(self.device)
                        prosody_targets = self.reference_encoder(audio)
                delta_voice_state = batch.get("delta_voice_state")
                if delta_voice_state is not None:
                    delta_voice_state = delta_voice_state.to(self.device)
            text_suprasegmentals = batch.get("text_suprasegmentals")
            if text_suprasegmentals is not None:
                text_suprasegmentals = text_suprasegmentals.to(self.device)

            losses = self._tts_pointer_step(
                phonemes=phonemes,
                language_ids=language_ids,
                speaker_embed=speaker_embed,
                speaker_labels=speaker_labels,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                target_a=target_a,
                target_b=target_b,
                target_length=target_length,
                f0_condition=f0_condition,
                dialogue_context=dialogue_context,
                acting_intent=acting_intent,
                prosody_targets=prosody_targets,
                delta_voice_state=delta_voice_state,
                text_suprasegmentals=text_suprasegmentals,
                prompt_kv_cache=prompt_kv_cache,
                batch=batch,
                current_step=current_step,
                pointer_loss_weight=pointer_loss_weight,
            )
            # Add prompt VQ loss if present
            if prompt_vq_loss is not None:
                losses["loss"] = losses["loss"] + prompt_vq_loss
                losses["loss_prompt_vq"] = prompt_vq_loss

        # --- v4 Phase 3: Physical supervision with mask (Task 3-1) ---
        # Physical supervision is supervision-dependent, so it must be
        # accumulated BEFORE tier weighting is applied.
        if self.enable_v4_losses:
            physical_targets = batch.get("physical_targets")
            physical_mask = batch.get("physical_observed_mask")
            if physical_targets is not None:
                physical_targets = physical_targets.to(self.device)
                physical_pred = explicit_state[:, :, :12] if explicit_state.shape[-1] >= 12 else explicit_state
                # Align temporal dimensions
                T_pred = physical_pred.shape[1]
                T_tgt = physical_targets.shape[1]
                T_min = min(T_pred, T_tgt)
                physical_pred_aligned = physical_pred[:, :T_min, :]
                physical_targets_aligned = physical_targets[:, :T_min, :]

                # Physical loss with mask: don't treat NaN/masked as zero
                # Replace NaN in targets with 0 to avoid NaN propagation
                physical_targets_safe = torch.nan_to_num(physical_targets_aligned, nan=0.0)
                phys_loss = F.mse_loss(
                    physical_pred_aligned, physical_targets_safe, reduction='none',
                )
                if physical_mask is not None:
                    physical_mask = physical_mask.to(self.device)[:, :T_min, :]
                    # Zero out loss for unobserved dimensions
                    phys_loss = phys_loss * physical_mask.float()
                    denom = physical_mask.float().sum().clamp(min=1.0)
                    phys_loss_scalar = phys_loss.sum() / denom
                else:
                    phys_loss_scalar = phys_loss.mean()

                losses["loss"] = losses["loss"] + self.v4_loss_config.lambda_physical * phys_loss_scalar
                losses["loss_physical_12d"] = phys_loss_scalar

        # --- v4 Phase 3: Tier-aware sample weighting (Task 3-1) ---
        # Apply tier weighting ONLY to supervision-dependent terms (codec loss,
        # physical loss, pointer loss) accumulated above.  Supervision-
        # independent terms (biological constraints, acting latent KL,
        # disentanglement, semantic alignment) are added AFTER this block and
        # remain at full weight regardless of tier.
        if self.enable_v4_losses:
            supervision_tier = batch.get("supervision_tier")
            if supervision_tier is not None:
                # supervision_tier can be a string or list of strings
                if isinstance(supervision_tier, str):
                    tier_key = _TIER_KEY_MAP.get(supervision_tier, "D")
                    sample_weight = TIER_WEIGHTS[tier_key]
                elif isinstance(supervision_tier, (list, tuple)):
                    # Per-sample tier: average the weights across batch
                    sample_weight = sum(
                        TIER_WEIGHTS.get(_TIER_KEY_MAP.get(t, "D"), 0.1)
                        for t in supervision_tier
                    ) / len(supervision_tier)
                else:
                    sample_weight = 1.0
                losses["loss"] = losses["loss"] * sample_weight
                losses["tier_sample_weight"] = torch.tensor(sample_weight)

        # --- v4 Phase 3: Biological constraint regularization (Task 3-2) ---
        # Supervision-independent: NOT tier-weighted.
        if self.enable_v4_losses and self.bio_regularizer is not None:
            # T2 fix: use model's hidden_states projected to 12-D so gradients
            # flow back through the model, instead of using the detached input
            # explicit_state which would make the regularizer a no-op.
            hidden_states = losses.get("_hidden_states")
            bio_losses = None
            if hidden_states is not None and self.bio_physical_head is not None:
                physical_for_bio = self.bio_physical_head(hidden_states)  # [B, T, 12]
                bio_losses = self.bio_regularizer(physical_for_bio)
            elif explicit_state.dim() == 3 and explicit_state.shape[-1] >= 12:
                # Fallback for VC mode where hidden_states may not be available
                physical_for_bio = explicit_state[:, :, :12]
                bio_losses = self.bio_regularizer(physical_for_bio)
            if bio_losses is not None:
                bio_total = (
                    self.v4_loss_config.lambda_bio_covariance * bio_losses["bio_covariance_loss"]
                    + self.v4_loss_config.lambda_bio_transition * bio_losses["bio_transition_loss"]
                    + self.v4_loss_config.lambda_bio_implausibility * bio_losses["bio_implausibility_loss"]
                )
                losses["loss"] = losses["loss"] + self.bio_constraint_weight * bio_total
                losses["loss_bio_covariance"] = bio_losses["bio_covariance_loss"]
                losses["loss_bio_transition"] = bio_losses["bio_transition_loss"]
                losses["loss_bio_implausibility"] = bio_losses["bio_implausibility_loss"]

        # --- v4 Phase 3: Full v4 loss composition (Task 3-3) ---
        # Acting latent KL, disentanglement, semantic alignment are
        # supervision-independent: NOT tier-weighted.
        if self.enable_v4_losses:
            losses = self._compute_v4_additional_losses(
                losses, batch, explicit_state, ssl_state, speaker_embed,
            )

        # Remove internal-only keys before backward
        losses.pop("_hidden_states", None)

        # Optimization
        losses["loss"].backward()
        # SOTA: Strict gradient clipping to prevent catastrophic loss spikes
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        res = {k: v.item() for k, v in losses.items() if isinstance(v, torch.Tensor)}
        res["mode"] = 1 if mode == "tts" else 0
        res["is_replay"] = 1 if is_replay else 0
        res["cfg_dropped"] = 1 if cfg_dropped else 0
        return res

    def _compute_v4_additional_losses(
        self,
        losses: dict,
        batch: dict,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        speaker_embed: torch.Tensor,
    ) -> dict:
        """Compute all 9 v4 loss terms (Task 3-3).

        Loss terms:
        1. codec token prediction - already in losses["loss_a"] from uclm_loss
        2. control token prediction - already in losses["loss_b"] from uclm_loss
        3. pointer progression - already in losses from _tts_pointer_step
        4. explicit physical supervision (12-D) - handled separately in train_step
        5. acting latent regularization (KL) - computed here
        6. disentanglement loss - computed here
        7. speaker consistency loss - computed here
        8. prosody prediction loss - already in losses from _tts_pointer_step
        9. semantic alignment loss - computed here
        """
        device = self.device

        # --- Loss 5: Acting latent regularization (KL) ---
        if self.acting_latent_encoder is not None and ssl_state is not None:
            # ssl_state should be [B, T, d_ssl=128]; ensure correct layout
            ssl_for_acting = ssl_state
            d_input = self.acting_latent_encoder.encoder[0].in_features  # 128
            if ssl_for_acting.dim() == 3 and ssl_for_acting.shape[-1] != d_input:
                if ssl_for_acting.shape[1] == d_input:
                    # [B, d_ssl, T] -> transpose to [B, T, d_ssl]
                    ssl_for_acting = ssl_for_acting.transpose(1, 2)

            latent, mu, logvar = self.acting_latent_encoder(ssl_for_acting)

            # KL regularization
            kl_loss = acting_latent_kl_loss(mu, logvar)
            losses["loss"] = losses["loss"] + self.acting_kl_weight * kl_loss
            losses["loss_acting_kl"] = kl_loss

            # Usage loss (anti-collapse)
            usage_loss = acting_latent_usage_loss(latent)
            losses["loss"] = losses["loss"] + self.v4_loss_config.lambda_acting_usage * usage_loss
            losses["loss_acting_usage"] = usage_loss

            # --- Loss 6: Disentanglement loss ---
            physical_pred = explicit_state
            if physical_pred.dim() == 3 and physical_pred.shape[-1] >= 12:
                physical_for_dis = physical_pred[:, :, :12]
                dis_loss = disentanglement_loss(physical_for_dis, latent)
                losses["loss"] = losses["loss"] + self.disentanglement_weight * dis_loss
                losses["loss_disentanglement"] = dis_loss

            # --- Loss 9: Semantic alignment loss ---
            if self.acting_latent_predictor is not None:
                # Use text features if available, otherwise use a pooled ssl_state proxy
                text_features = batch.get("text_features")
                if text_features is not None:
                    text_features = text_features.to(device)
                else:
                    # Pool ssl_state as proxy for text features
                    text_features = ssl_for_acting.mean(dim=1)  # [B, d_ssl]

                predicted_latent = self.acting_latent_predictor(text_features)
                sem_loss = semantic_alignment_loss(predicted_latent, latent)
                losses["loss"] = losses["loss"] + self.semantic_alignment_weight * sem_loss
                losses["loss_semantic_alignment"] = sem_loss

        # --- Loss 7: Speaker consistency loss ---
        # Encourage consistent speaker embeddings across same-speaker utterances
        if speaker_embed is not None and speaker_embed.shape[0] > 1:
            # Compute pairwise cosine similarity within batch
            spk_norm = F.normalize(speaker_embed, dim=-1)  # [B, d_speaker]
            similarity_matrix = spk_norm @ spk_norm.T  # [B, B]

            # Speaker IDs for contrastive comparison
            speaker_ids = batch.get("speaker_id")
            if speaker_ids is not None and isinstance(speaker_ids, torch.Tensor):
                speaker_ids = speaker_ids.to(device)
                # Build same-speaker mask
                same_speaker = (speaker_ids.unsqueeze(0) == speaker_ids.unsqueeze(1)).float()
                # Exclude diagonal
                mask_diag = 1.0 - torch.eye(speaker_embed.shape[0], device=device)
                same_speaker = same_speaker * mask_diag

                if same_speaker.sum() > 0:
                    # Same-speaker pairs should be similar (high cosine)
                    positive_sim = (similarity_matrix * same_speaker).sum() / same_speaker.sum().clamp(min=1)
                    # Loss: 1 - average positive similarity
                    spk_loss = 1.0 - positive_sim
                    losses["loss"] = losses["loss"] + self.speaker_consistency_weight * spk_loss
                    losses["loss_speaker_consistency"] = spk_loss

        return losses

    def _tts_pointer_step(
        self,
        phonemes: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embed: torch.Tensor,
        speaker_labels: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        target_length: int,
        f0_condition: torch.Tensor | None,
        dialogue_context: torch.Tensor | None,
        acting_intent: torch.Tensor | None,
        prosody_targets: torch.Tensor | None,
        delta_voice_state: torch.Tensor | None,
        text_suprasegmentals: torch.Tensor | None,
        prompt_kv_cache: torch.Tensor | None,
        batch: dict,
        current_step: int,
        pointer_loss_weight: float,
    ) -> dict:
        """Pointer-mode TTS training step with full annealing schedule."""
        B = phonemes.shape[0]
        L = phonemes.shape[1]

        # Determine annealing phase
        annealing_phase = self._get_annealing_phase(current_step)
        is_hard_bootstrap = (annealing_phase == "hard_bootstrap")

        # Stage 2 Annealing: provide bootstrap alignment if in hard phase
        bootstrap_data = None
        pos_indices_init = None
        if is_hard_bootstrap:
            bootstrap_data = batch.get("bootstrap_alignment")
            if bootstrap_data is not None and "phoneme_indices" in bootstrap_data:
                pos_indices_init = bootstrap_data["phoneme_indices"]
            else:
                # Fallback to heuristic durations if curated bootstrap is missing
                # but we are in the hard bootstrap phase.
                indices = torch.arange(target_length, device=self.device).float() * L / target_length
                pos_indices_init = indices.long().clamp(max=L - 1).unsqueeze(0).expand(B, -1)
                bootstrap_data = {"phoneme_indices": pos_indices_init}

        # First pass: use initial pos_indices (if bootstrap) or uniform (if latent)
        # to get hidden states for MAS.
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
            prompt_kv_cache=prompt_kv_cache,
            bootstrap_alignment=bootstrap_data,
            text_suprasegmentals=text_suprasegmentals,
            position_indices=pos_indices_init,
        )

        # --- Alignment Target Generation (Stage 2 Annealing Logic) ---
        if annealing_phase == "hard_bootstrap":
            # Phase 1: Use Bootstrap Alignment as ground truth
            if pos_indices_init is not None:
                indices = pos_indices_init  # [B, T]
                pseudo_durations = torch.zeros(B, L, device=self.device)
                for b in range(B):
                    counts = torch.bincount(indices[b].clamp(min=0, max=L-1), minlength=L)
                    pseudo_durations[b] = counts[:L].float()
                adv_targets, prog_targets, frame_offsets, pos_indices = self._generate_pointer_targets(pseudo_durations, target_length)
            else:
                # Heuristic fallback
                dummy_durations = torch.full((B, L), target_length / max(L, 1), device=self.device)
                adv_targets, prog_targets, frame_offsets, pos_indices = self._generate_pointer_targets(dummy_durations, target_length)
        elif annealing_phase == "soft_transition":
            # Phase 2: Mix bootstrap with MAS-derived targets
            mix_ratio = self._get_bootstrap_mix_ratio(current_step)

            # MAS targets
            with torch.no_grad():
                phoneme_lens = (phonemes != 0).sum(dim=1)
                x_text = F.normalize(self.model.text_encoder(
                    phonemes, language_ids, phoneme_lens,
                    text_suprasegmentals=text_suprasegmentals,
                ).transpose(1, 2), dim=-1)
                x_acoustic = F.normalize(out["hidden_states"], dim=-1)
                dist = torch.cdist(x_text, x_acoustic)
                log_probs = -0.5 * (dist ** 2)
                path = monotonic_alignment_search(log_probs)
                mas_adv, mas_prog, mas_offs, mas_indices = self._generate_pointer_targets_from_path(path)

            if mix_ratio > 0 and pos_indices_init is not None:
                indices = pos_indices_init
                pseudo_durations = torch.zeros(B, L, device=self.device)
                for b in range(B):
                    counts = torch.bincount(indices[b].clamp(min=0, max=L-1), minlength=L)
                    pseudo_durations[b] = counts[:L].float()
                boot_adv, boot_prog, boot_offs, boot_indices = self._generate_pointer_targets(pseudo_durations, target_length)
                # Blend
                adv_targets = mix_ratio * boot_adv + (1 - mix_ratio) * mas_adv
                prog_targets = mix_ratio * boot_prog + (1 - mix_ratio) * mas_prog
                frame_offsets = mix_ratio * boot_offs + (1 - mix_ratio) * mas_offs
                # SOTA: Use bootstrap indices if mix_ratio is high for RoPE stability
                pos_indices = boot_indices if mix_ratio > 0.5 else mas_indices
            else:
                adv_targets, prog_targets, frame_offsets, pos_indices = mas_adv, mas_prog, mas_offs, mas_indices
        else:
            # Phase 3: Pure Latent-Only (MAS from internal attention)
            with torch.no_grad():
                phoneme_lens = (phonemes != 0).sum(dim=1)
                x_text = F.normalize(self.model.text_encoder(
                    phonemes, language_ids, phoneme_lens,
                    text_suprasegmentals=text_suprasegmentals,
                ).transpose(1, 2), dim=-1)
                x_acoustic = F.normalize(out["hidden_states"], dim=-1)
                dist = torch.cdist(x_text, x_acoustic)
                log_probs = -0.5 * (dist ** 2)
                path = monotonic_alignment_search(log_probs)
                adv_targets, prog_targets, frame_offsets, pos_indices = self._generate_pointer_targets_from_path(path)

        # SOTA: Generate Dynamic Cross-Attention Window Mask for training
        # This ensures the model learns to focus only on relevant phonemes.
        # Window: current_phoneme +/- 1.
        with torch.no_grad():
            B, T = pos_indices.shape
            S = phonemes.shape[1]
            # Create mask: [B, 1, T, S]
            ca_mask = torch.full((B, 1, T, S), float("-inf"), device=self.device)
            # Efficiently fill the window using scatter or loop
            # For training simplicity, we can use a loop over B or a vectorized approach
            for b in range(B):
                for t_idx in range(T):
                    p_idx = pos_indices[b, t_idx]
                    w_start = max(0, p_idx - 1)
                    w_end = min(S, p_idx + 2)
                    ca_mask[b, 0, t_idx, w_start:w_end] = 0.0

        # Worker 01/02: Second forward pass with correct position_indices AND frame_offsets for RoPE
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
            prompt_kv_cache=prompt_kv_cache,
            text_suprasegmentals=text_suprasegmentals,
            position_indices=pos_indices,
            frame_offsets=frame_offsets,
            cross_attn_mask=ca_mask,
        )

        # Generate boundary confidence targets
        boundary_targets = self._generate_boundary_targets_from_advance(adv_targets)

        context_groups = batch.get("context_groups")
        if context_groups is not None:
            context_groups = context_groups.to(self.device)

        # Voice state supervision (Worker 01/03 contract)
        vs_targets = batch.get("voice_state_targets")
        vs_observed_mask = batch.get("voice_state_observed_mask")
        vs_confidence = batch.get("voice_state_confidence")
        if vs_targets is not None:
            vs_targets = vs_targets.to(self.device)
            if vs_observed_mask is not None:
                vs_observed_mask = vs_observed_mask.to(self.device)
            else:
                # Default to True if not provided
                vs_observed_mask = torch.ones_like(vs_targets, dtype=torch.bool, device=self.device)
                
            if vs_confidence is not None:
                vs_confidence = vs_confidence.to(self.device)
                # Apply confidence floor (Worker 02 requirement)
                floor_mask = vs_confidence >= self.voice_state_confidence_floor
                # If confidence is [B, T, 1], broadcast to [B, T, D]
                if floor_mask.shape[-1] == 1 and vs_observed_mask.shape[-1] > 1:
                    floor_mask = floor_mask.expand_as(vs_observed_mask)
                vs_observed_mask = vs_observed_mask & floor_mask

        # Compute phoneme features for prosody loss (reuse text encoder)
        phoneme_lens = (phonemes != 0).sum(dim=1)
        phoneme_features = self.model.text_encoder(
            phonemes, language_ids, phoneme_lens,
            text_suprasegmentals=text_suprasegmentals,
        ).transpose(1, 2)

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
            phoneme_features=phoneme_features,
            target_prosody=prosody_targets,
            prosody_dialogue_context=dialogue_context,
            prosody_speaker_embed=speaker_embed,
            lambda_prosody=self.prosody_loss_weight if prosody_targets is not None else 0.0,
        )

        # Boundary confidence loss
        if (
            self.boundary_confidence_loss_weight > 0
            and out.get("boundary_confidence") is not None
        ):
            loss_boundary = boundary_confidence_loss(
                out["boundary_confidence"], boundary_targets
            )
            losses["loss"] = losses["loss"] + self.boundary_confidence_loss_weight * loss_boundary
            losses["loss_boundary_confidence"] = loss_boundary

        # CFG distillation loss (Stage 3 only)
        if (
            self.cfg_distillation_weight > 0
            and self.curriculum is not None
            and self.curriculum.get_stage(current_step) >= 3
        ):
            self._apply_cfg_distillation(
                losses=losses,
                phonemes=phonemes,
                language_ids=language_ids,
                speaker_embed=speaker_embed,
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                target_a=target_a,
                target_b=target_b,
                target_length=target_length,
                f0_condition=f0_condition,
                dialogue_context=dialogue_context,
                acting_intent=acting_intent,
                prosody_targets=prosody_targets,
                delta_voice_state=delta_voice_state,
                text_suprasegmentals=text_suprasegmentals,
                out=out,
                position_indices=pos_indices,
                frame_offsets=frame_offsets,
                cross_attn_mask=ca_mask,
            )

        # Voice state supervision with observed_mask and confidence (Worker 02)
        if (
            self.voice_state_loss_weight > 0
            and vs_targets is not None
            and out.get("hidden_states") is not None
        ):
            # Use hidden_states[:, :, :12] as proxy for voice state prediction
            # (real implementation uses a dedicated head)
            vs_pred = out["hidden_states"][:, :vs_targets.shape[1], :12]
            loss_vs = voice_state_supervision_loss(
                vs_pred,
                vs_targets,
                observed_mask=vs_observed_mask,
                confidence=vs_confidence,
            )
            losses["loss"] = losses["loss"] + self.voice_state_loss_weight * loss_vs
            losses["loss_voice_state"] = loss_vs

        # Anti-collapse diagnostics (logged but not added to loss)
        if out.get("advance_logit") is not None:
            pointer_diag = self._compute_pointer_diagnostics(
                out["advance_logit"], out["progress_delta"]
            )
            for k, v in pointer_diag.items():
                losses[k] = torch.tensor(v)

        if out.get("hidden_states") is not None:
            vs_diag = self._compute_voice_state_diagnostics(out["hidden_states"])
            for k, v in vs_diag.items():
                losses[k] = torch.tensor(v)

        # Pass hidden_states through for bio regularizer (T2 fix: use model output)
        if out.get("hidden_states") is not None:
            losses["_hidden_states"] = out["hidden_states"]

        return losses

    def _apply_cfg_distillation(
        self,
        losses: dict,
        phonemes: torch.Tensor,
        language_ids: torch.Tensor,
        speaker_embed: torch.Tensor,
        explicit_state: torch.Tensor,
        ssl_state: torch.Tensor,
        target_a: torch.Tensor,
        target_b: torch.Tensor,
        target_length: int,
        f0_condition: torch.Tensor | None,
        dialogue_context: torch.Tensor | None,
        acting_intent: torch.Tensor | None,
        prosody_targets: torch.Tensor | None,
        delta_voice_state: torch.Tensor | None,
        text_suprasegmentals: torch.Tensor | None,
        out: dict,
        position_indices: torch.Tensor,
        frame_offsets: torch.Tensor,
        cross_attn_mask: torch.Tensor, # New SOTA field
    ) -> None:
        """Apply CFG self-distillation loss (Stage 3 only)."""
        # Sample a random cfg_scale from the configured range
        cfg_lo, cfg_hi = self.cfg_distillation_scale_range
        sampled_cfg_scale = cfg_lo + random.random() * (cfg_hi - cfg_lo)

        # Teacher: full two-pass guided logits (detached)
        with torch.no_grad():
            # Conditional pass (already computed: out["logits_a"], out["logits_b"])
            cond_a = out["logits_a"]
            cond_b = out["logits_b"]

            # Unconditional pass
            masked_distill = DisentangledUCLM.apply_cfg_unconditional_mask(
                explicit_state=explicit_state,
                ssl_state=ssl_state,
                speaker_embed=speaker_embed,
                dialogue_context=dialogue_context,
                acting_intent=acting_intent,
                prosody_latent=prosody_targets,
                delta_voice_state=delta_voice_state,
            )
            out_uncond = self.model.forward_tts_pointer(
                phoneme_ids=phonemes,
                language_ids=language_ids,
                pointer_state=None,
                speaker_embed=masked_distill["speaker_embed"],
                explicit_state=masked_distill["explicit_state"],
                ssl_state=masked_distill["ssl_state"],
                target_a=target_a,
                target_b=target_b,
                target_length=target_length,
                f0_condition=f0_condition,
                dialogue_context=masked_distill["dialogue_context"],
                acting_intent=masked_distill["acting_intent"],
                prosody_latent=masked_distill["prosody_latent"],
                delta_voice_state=masked_distill["delta_voice_state"],
                position_indices=position_indices, # SOTA consistency
                frame_offsets=frame_offsets,       # SOTA consistency
                cross_attn_mask=cross_attn_mask,   # SOTA consistency
            )
            uncond_a = out_uncond["logits_a"]
            uncond_b = out_uncond["logits_b"]

            # Full CFG blend (teacher targets)
            teacher_a = uncond_a + sampled_cfg_scale * (cond_a - uncond_a)
            teacher_b = uncond_b + sampled_cfg_scale * (cond_b - uncond_b)

        # Student: single-pass with cfg_scale as input (gradients flow)
        student_out = self.model.forward_tts_distilled_cfg(
            phoneme_ids=phonemes,
            language_ids=language_ids,
            speaker_embed=speaker_embed,
            explicit_state=explicit_state,
            ssl_state=ssl_state,
            target_a=target_a,
            target_b=target_b,
            cfg_scale=sampled_cfg_scale,
            target_length=target_length,
            f0_condition=f0_condition,
            dialogue_context=dialogue_context,
            acting_intent=acting_intent,
            prosody_latent=prosody_targets,
            delta_voice_state=delta_voice_state,
            text_suprasegmentals=text_suprasegmentals,
            position_indices=position_indices,
            frame_offsets=frame_offsets,
            cross_attn_mask=cross_attn_mask, # SOTA consistency
        )
        student_a = student_out["logits_a"]
        student_b = student_out["logits_b"]

        loss_cfg_distill = DisentangledUCLM.cfg_distillation_loss(
            teacher_logits_a=teacher_a.detach(),
            teacher_logits_b=teacher_b.detach(),
            student_logits_a=student_a,
            student_logits_b=student_b,
            temperature=self.cfg_distillation_temperature,
        )
        losses["loss"] = losses["loss"] + self.cfg_distillation_weight * loss_cfg_distill
        losses["loss_cfg_distillation"] = loss_cfg_distill

    @torch.no_grad()
    def val_step(self, batch: dict) -> dict:
        """Validation step with SOTA parity."""
        self.model.eval()
        
        # 1. Prepare Inputs
        phonemes = batch["phoneme_ids"].to(self.device)
        language_ids = batch["language_id"].to(self.device)
        speaker_embed = batch["speaker_embed"].to(self.device)
        speaker_labels = batch["speaker_id"].to(self.device)
        explicit_state = batch["voice_state_explicit"].to(self.device)
        ssl_state = batch["ssl_state"].to(self.device)
        target_a = batch["codec_tokens_a"].to(self.device)
        target_b = batch["codec_tokens_b"].to(self.device)
        target_length = target_a.shape[2]
        
        text_suprasegmentals = batch.get("text_suprasegmentals")
        if text_suprasegmentals is not None:
            text_suprasegmentals = text_suprasegmentals.to(self.device)

        # 2. First Pass (Acoustic Context Only)
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
            text_suprasegmentals=text_suprasegmentals,
        )

        # 3. Generate Alignment via MAS (Force MAS for validation)
        phoneme_lens = (phonemes != 0).sum(dim=1)
        x_text = F.normalize(self.model.text_encoder(
            phonemes, language_ids, phoneme_lens,
            text_suprasegmentals=text_suprasegmentals,
        ).transpose(1, 2), dim=-1)
        x_acoustic = F.normalize(out["hidden_states"], dim=-1)
        dist = torch.cdist(x_text, x_acoustic)
        log_probs = -0.5 * (dist ** 2)
        path = monotonic_alignment_search(log_probs)
        adv_targets, prog_targets, frame_offsets, pos_indices = self._generate_pointer_targets_from_path(path)

        # 4. Generate Window Mask
        B, T = pos_indices.shape
        S = phonemes.shape[1]
        ca_mask = torch.full((B, 1, T, S), float("-inf"), device=self.device)
        for b in range(B):
            for t_idx in range(T):
                p_idx = pos_indices[b, t_idx]
                w_start = max(0, p_idx - 1)
                w_end = min(S, p_idx + 2)
                ca_mask[b, 0, t_idx, w_start:w_end] = 0.0

        # 5. Second Pass (Full SOTA Parity)
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
            position_indices=pos_indices,
            frame_offsets=frame_offsets,
            text_suprasegmentals=text_suprasegmentals,
            cross_attn_mask=ca_mask,
        )

        # 6. Compute Losses
        losses = uclm_loss(
            logits_a=out["logits_a"],
            logits_b=out["logits_b"],
            target_a=target_a,
            target_b=target_b,
            speaker_labels=speaker_labels,
            pointer_logits=out.get("advance_logit"),
            advance_targets=adv_targets,
            progress_delta=out.get("progress_delta"),
            progress_targets=prog_targets,
            lambda_pointer=self.pointer_loss_weight,
            lambda_progress=self.progress_loss_weight,
        )
        
        metrics = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
        metrics["mode"] = True # TTS
        return metrics

    def train_epoch(self, dataloader: DataLoader):
        total_loss = 0
        pbar = tqdm(dataloader, desc="Training")
        for batch in pbar:
            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            pbar.set_postfix({"loss": metrics["loss"], "mode": "TTS" if metrics["mode"] else "VC"})
        return total_loss / len(dataloader)
