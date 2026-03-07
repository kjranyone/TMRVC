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
- runtime-friendly CFG fast paths are planned explicitly rather than assumed away


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
   - `alignment_loss_type: none | aux_mas | aux_ctc`
   - `pointer_supervision_mode: latent_only | supervised`
   - `pointer_target_source: none | heuristic_bootstrap | bootstrap_projection | legacy_duration`
     - `heuristic_bootstrap`: uniform phoneme distribution used as initial scaffolding (no external labels needed)
     - `bootstrap_projection`: ASR-derived timestamps projected onto canonical phoneme indices (requires curated alignment artifacts)
     - `legacy_duration`: MFA-derived duration labels (v2 compatibility only)
   - `pointer_loss_weight`
   - `progress_loss_weight`
   - `legacy_duration_loss_weight`
   - `bootstrap_alignment_required: true | false`
   - `pointer_aux_alignment_warmup_steps`
   - `pointer_aux_alignment_anneal_steps`
   - `stage3_replay_mix_ratio`
   - `stage3_dialogue_mix_ratio`
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
   - mainline pointer mode must support `latent_only` training with no external pointer labels
   - supervised pointer loss may run only when one supervision source is available
   - accepted supervised sources:
     - deterministic `bootstrap_projection`
     - temporary `legacy_duration`
   - if `pointer_supervision_mode=supervised` and no valid source exists, configuration validation must fail rather than silently training without labels
5. Add alignment-loss implementation policy:
   - `aux_mas` or `aux_ctc` may regularize monotonicity or provide diagnostics, but they are not the steady-state release dependency
   - temporary bootstrap alignment is allowed only as an explicit transitional mode
   - no-op alignment is allowed for interface scaffolding only, not for a trainable pointer experiment
   - convergence policy:
     - early pointer training uses non-zero auxiliary alignment weight during a documented warmup window
     - auxiliary alignment weight is annealed toward the alignment-free regime over a documented schedule
     - a pure `latent_only` recipe is not considered valid unless it converges after this scheduled annealing procedure
   - release exit criterion:
     - v3 sign-off requires at least one convergent pointer-training recipe that does not require MAS/CTC to produce frame-level pointer labels
   - worker ownership:
     - Worker 02 owns optional auxiliary `mas` / `ctc` regularizers and diagnostics inside training
     - Worker 03 defines canonical phoneme-space bootstrap projection inputs
     - Worker 10 exports bootstrap supervision artifacts in that canonical format
6. Update CLI validation:
   - `--require-tts-supervision` must not mean `durations.npy required`
   - it should mean `text-side supervision required`
   - `tts_mode=pointer` with `pointer_supervision_mode=supervised` must require a valid `pointer_target_source`
   - release presets must include a documented `latent_only` pointer recipe
7. Update pipeline quality gate terminology:
   - separate `text supervision coverage`
   - separate `legacy duration coverage`
   - separate `pointer target coverage`
8. **Implement the 3-stage Training Pipeline and logic for Stage 1/2/3 transitions.**
   - Stage 2 must document the warmup-to-latent-only schedule explicitly
9. **Implement CFG Training logic (conditioning dropout for speaker/style/context).**
10. Add CFG runtime-acceleration preparation:
    - define whether runtime may use `off | full | lazy | distilled` CFG modes
    - if `distilled` is supported, implement **CFG Self-Distillation** hooks in Stage 3, allowing the model to approximate 2-pass guided outputs in a 1-pass inference step, ensuring high expressive capacity fits within real-time VST budgets.
    - if `lazy` CFG is supported, define the training-time assumption or robustness check for skipped guidance refreshes
11. **Implement Flow-matching Loss for the Prosody Predictor.**
12. Add expressive-training hooks:
    - dialogue-context conditioning batch fields
    - utterance-level style / acting labels when available
    - optional local prosody latent supervision
    - (CFG-based acting control is achieved via `cfg_scale` scalar and conditioning dropout, not a dedicated embedding)
13. Add anti-collapse training losses or diagnostics:
    - same-text different-context separation metric
    - prosody variance floor or diversity regularizer
14. **Add Zero-Shot/Few-Shot prompt training logic:**
    - dynamic prompt sampling: pick a random 3-10s clip from the same speaker to serve as `prompt_codec_tokens` and `speaker_embed` target.
    - information bottleneck or contrastive learning objectives to enforce timbre-prosody disentanglement (ensure the model doesn't just copy the reference prosody).
    - cross-emotion/cross-context prompting: train with a prompt from context A, while generating text from context B, to force the model to rely on `dialogue_context` for prosody and the prompt only for timbre.
15. Add Stage 3 anti-forgetting sampling policy:
    - mix Stage 1/2 stability data with Stage 3 drama data using explicit replay ratios
    - allow dynamic N:M schedule rather than drama-only finetuning
    - track whether generic read-speech quality or base intelligibility regresses during drama finetuning


## Alignment Responsibility Split

- Worker 02 owns:
  - optional auxiliary `mas` / `ctc` computation used directly by the trainer
  - conversion of canonical bootstrap spans into frame-level pointer supervision when supervised mode is enabled
  - alignment-related losses and training-time diagnostics
- Worker 03 owns:
  - canonical text normalization and phoneme-index projection rules for bootstrap supervision artifacts
  - dataset plumbing for loading those artifacts when explicitly configured
- Worker 10 owns:
  - export of `bootstrap_alignment.json` in canonical phoneme-index space with provenance and confidence
- Worker 02 must not assume Worker 03 will solve pointer learning through permanent online aligners.
- Worker 03 must not silently synthesize pseudo-durations and present them as stable mainline supervision.


## Batch Contract

The trainer must accept TTS examples with:

- `phoneme_ids` present
- `durations` optional
- `language_id` present
- required pointer-target source in pointer mode:
  - no external labels when `pointer_supervision_mode=latent_only`, or
  - temporary bootstrap alignment labels already projected onto canonical phoneme indices
- optional `dialogue_context` as raw text-context tokens or deterministically derived embeddings
- optional `acting_intent`
- optional `prosody_targets` (for Flow-matching)
- optional `acoustic_history_teacher` or equivalent teacher-forced codec/control history source

## Alignment Learning Policy (Pointer Target Generation)

To avoid learning a robotic, uniform-duration rhythm, the model must **NOT** use naive uniform temporal distribution (`target_length // L`) for pointer targets during early training.

- **Stage 2 Requirement:** The trainer must use **Monotonic Alignment Search (MAS)** or a dynamic programming equivalent (e.g., Forward-Sum) to align `phoneme_features` with the actual acoustic features (`target_b` / `target_a`) dynamically.
- The pointer head's target (`advance_targets`, `progress_targets`) must be derived from this MAS path, ensuring the model learns the natural variance in phoneme durations.
- Uniform distribution is explicitly forbidden as a training target, as it guarantees prosody collapse.


## Training Policy For Pointer Supervision

- steady-state target:
  - pointer progression is learnable in `latent_only` mode with no external frame-level aligner dependency
- optional auxiliary target:
  - monotonic regularization may use `aux_mas` or `aux_ctc`
- transitional target:
  - supervised pointer loss may come from `bootstrap_projection` or `legacy_duration`
- forbidden state:
  - supervised pointer loss enabled while pointer targets are absent
- release condition:
  - bootstrap and legacy-duration supervision must be demoted to transitional or ablation-only once the alignment-free pointer recipe is validated
- recommended convergence recipe:
  - warm up pointer learning with auxiliary alignment regularization for the first configured step window
  - anneal auxiliary alignment toward zero before declaring the recipe alignment-free


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
- do not turn MAS/CTC into an unexamined permanent dependency of the released v3 path
- do not enable supervised pointer loss without a concrete bootstrap target source
- do not push ownership of canonical bootstrap projection into ad hoc dataset code


## Handoff Contract

- training path runs in `pointer` mode without `durations.npy`
- legacy duration mode still works for comparison
- checkpoint saves all new pointer-head weights
- **CFG training capability confirmed.**
- **CFG fast-path contract (`full | lazy | distilled | off`) documented and aligned with Worker 04.**
- **Flow-matching prosody prediction trained.**
- at least one documented recipe trains pointer mode without requiring external aligners at release time
- Stage 3 replay-mix policy is documented and prevents drama finetuning from collapsing base quality


## Required Tests

- trainer can step with `phoneme_ids` but no `durations`
- pointer supervised mode config fails fast when no valid pointer target source is configured
- legacy duration mode still computes duration loss
- latent-only pointer mode config is accepted without bootstrap labels
- auxiliary-alignment warmup / anneal schedule config test
- **CFG training test (ensure conditioning is dropped per configured probability).**
- CFG fast-path config validation test
- **Flow-matching loss test (ensure gradient flows to Prosody Predictor).**
- config parsing tests for new flags
- quality gate tests renamed or expanded for new supervision semantics
- test that context-conditioned batches are accepted without breaking non-context batches
- test that diversity diagnostics run on same-text multi-context samples
- test that bootstrap alignment projection is monotonic and indexes canonical `phoneme_ids`
- Stage 3 replay-mix sampler test
