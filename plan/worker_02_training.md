# Worker 02: Training Path and Losses

## Scope

Replace duration-centric TTS training with pointer-centric TTS training while preserving a legacy fallback branch for comparison.


## Primary Files

- `tmrvc-train/src/tmrvc_train/trainer.py`
- `tmrvc-train/src/tmrvc_train/models/uclm_loss.py`
- `tmrvc-train/src/tmrvc_train/models/duration_predictor.py`
- `tmrvc-train/src/tmrvc_train/cli/train_uclm.py`
- `tmrvc-train/src/tmrvc_train/pipeline.py`
- `configs/train_uclm.yaml.example`
- tests under `tests/train/`


## Required Outcomes

- training works without `durations.npy`
- pointer loss is first-class
- optional auxiliary alignment regularization hook exists for MAS/CTC
- duration loss is legacy-only
- pointer supervision source is explicit and not hand-waved
- MAS/CTC ownership boundary is explicit
- expressive phrasing is trainable from dialogue-conditioned supervision
- acting diversity does not collapse to one reading per sentence
- **Adopt a 3-stage Training Curriculum (Base, Align, Drama)**
- **Define CFG (Classifier-Free Guidance) Training Policy**
- mainline release path includes at least one pointer-training configuration that does not depend on external aligners


## Training Curriculum

Worker 02 must implement and document the following curriculum:

### Stage 1: Base LM Pre-training
- **Goal:** Learn general acoustic/codec distributions and basic speaker/timbre consistency.
- **Data:** Massive raw audio (curated but potentially low-supervision).
- **Task:** Next-token prediction for codec tokens.
- **Note:** VC path is primarily trained here.

### Stage 2: Alignment & Pointer Training
- **Goal:** Learn text-to-audio alignment and monotonic pointer progression.
- **Data:** High-quality transcribed subsets (curated).
- **Task:** Internal pointer learning with optional auxiliary monotonic regularization and optional bootstrap supervision.
- **Note:** Duration-predictor logic is replaced by pointer-loss here.

### Stage 3: Dramatic & Dialogue Finetuning
- **Goal:** Learn drama-grade acting, context-sensitivity, and zero-shot disentanglement.
- **Data:** High-expressivity dialogue datasets (curated).
- **Task:** Dialogue-conditioned TTS with Prosody Predictor and CFG training.
- **Note:** **Classifier-Free Guidance (CFG)** training is introduced here by randomly dropping conditioning (10-20% dropout) to allow inference-time guidance scaling.


## Concrete Tasks

1. Add config flags:
   - `tts_mode: legacy_duration | pointer`
   - `alignment_loss_type: none | mas | ctc`
   - `pointer_target_source: mas | ctc | legacy_duration | heuristic_bootstrap`
   - `pointer_loss_weight`
   - `progress_loss_weight`
   - `legacy_duration_loss_weight`
   - `bootstrap_alignment_required: true | false`
   - **`conditioning_dropout_prob`** (for CFG training)
   - **`prosody_loss_weight`** (for Flow-matching)
2. Refactor `trainer.py`:
   - separate VC step and TTS pointer step
   - remove assumption that TTS batch must have durations
3. Extend `uclm_loss.py`:
   - add pointer classification loss
   - add progress regression loss
   - keep duration loss behind legacy flag
4. Define pointer-target generation contract:
   - pointer loss may run only when one supervision source is available
   - accepted sources:
     - online `mas`
     - online `ctc`
     - temporary `legacy_duration`
     - temporary `heuristic_bootstrap`
   - if none exists, pointer training must fail configuration validation rather than silently training without labels
5. Add alignment-loss implementation policy:
   - `mas` or `ctc` is the intended steady-state path
   - if those are not ready yet, temporary bootstrap alignment is allowed only as an explicit transitional mode
   - no-op alignment is allowed for interface scaffolding only, not for a trainable pointer experiment
   - worker ownership:
     - Worker 02 owns online `mas` / `ctc` target generation inside training
     - Worker 03 may provide bootstrap labels or sidecar supervision artifacts, but not the main online alignment logic
6. Update CLI validation:
   - `--require-tts-supervision` must not mean `durations.npy required`
   - it should mean `text-side supervision required`
   - `tts_mode=pointer` must require a valid `pointer_target_source`
7. Update pipeline quality gate terminology:
   - separate `text supervision coverage`
   - separate `legacy duration coverage`
   - separate `pointer target coverage`
8. **Implement the 3-stage Training Pipeline and logic for Stage 1/2/3 transitions.**
9. **Implement CFG Training logic (conditioning dropout for speaker/style/context).**
10. **Implement Flow-matching Loss for the Prosody Predictor.**
11. Add expressive-training hooks:
    - dialogue-context conditioning batch fields
    - utterance-level style / acting labels when available
    - optional local prosody latent supervision
    - **`style_guidance_embedding`** (for CFG-based acting control)
12. Add anti-collapse training losses or diagnostics:
    - same-text different-context separation metric
    - prosody variance floor or diversity regularizer
13. **Add Zero-Shot/Few-Shot prompt training logic:**
    - dynamic prompt sampling: pick a random 3-10s clip from the same speaker to serve as `prompt_codec_tokens` and `speaker_embed` target.
    - information bottleneck or contrastive learning objectives to enforce timbre-prosody disentanglement (ensure the model doesn't just copy the reference prosody).
    - cross-emotion/cross-context prompting: train with a prompt from context A, while generating text from context B, to force the model to rely on `dialogue_context` for prosody and the prompt only for timbre.


## Alignment Responsibility Split

- Worker 02 owns:
  - online `mas` / `ctc` computation used directly by the trainer
  - pointer-target extraction from online alignment outputs
  - alignment-related losses and training-time diagnostics
- Worker 03 owns:
  - offline bootstrap supervision artifacts only
  - dataset plumbing for loading those artifacts when explicitly configured
- Worker 02 must not assume Worker 03 will solve online alignment.
- Worker 03 must not silently synthesize pseudo-durations and present them as stable mainline supervision.


## Batch Contract

The trainer must accept TTS examples with:

- `phoneme_ids` present
- `durations` optional
- `language_id` present
- required pointer-target source in pointer mode:
  - online MAS/CTC targets, or
  - temporary bootstrap alignment labels
- optional `dialogue_context` as text-context embeddings or pre-encoded context tokens
- optional `acting_intent`
- optional `prosody_targets` (for Flow-matching)
- optional `acoustic_history_teacher` or equivalent teacher-forced codec/control history source


## Training Policy For Pointer Supervision

- steady-state target:
  - pointer supervision comes from online `mas` or `ctc`
- transitional target:
  - pointer supervision may come from `legacy_duration` or `heuristic_bootstrap`
- forbidden state:
  - pointer loss enabled while pointer targets are absent
- release condition:
  - once `mas` or `ctc` is stable, transitional bootstrap modes must be demoted to legacy or ablation-only


## Early Anti-Collapse Metric Definitions

Worker 02 should freeze initial metric formulas early so Worker 06 can reuse them.

- `context_separation_score`
  - compare prosody/style realizations for same text under different context
  - initial proxy:
    - mean pairwise distance between context-conditioned prosody embeddings for identical text
    - normalized by within-context variance
- `control_response_score`
  - measure monotonic change in runtime metrics under controlled `pace`, `hold_bias`, `boundary_bias` sweeps
- `prosody_collapse_score`
  - ratio of between-context variance to total variance for same-text samples

Exact production formulas may evolve, but these names and intents must be fixed early.


## Guardrails

- do not hard-wire MAS implementation before interface stabilizes
- keep all new losses independently switchable for ablation
- do not remove current VC training path
- do not equate low reconstruction loss with expressive success
- do not pretend pointer learning is possible without a concrete alignment or bootstrap target source
- do not push ownership of online MAS/CTC into dataset code


## Handoff Contract

- training path runs in `pointer` mode without `durations.npy`
- legacy duration mode still works for comparison
- checkpoint saves all new pointer-head weights
- **CFG training capability confirmed.**
- **Flow-matching prosody prediction trained.**


## Required Tests

- trainer can step with `phoneme_ids` but no `durations`
- pointer mode config fails fast when no valid pointer target source is configured
- legacy duration mode still computes duration loss
- **CFG training test (ensure conditioning is dropped per configured probability).**
- **Flow-matching loss test (ensure gradient flows to Prosody Predictor).**
- config parsing tests for new flags
- quality gate tests renamed or expanded for new supervision semantics
- test that context-conditioned batches are accepted without breaking non-context batches
- test that diversity diagnostics run on same-text multi-context samples
