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
- 8-D `voice_state` supervision, masks, and confidences are trainable when present and safely ignorable when absent
- suprasegmental text features are trainable and validated for languages that require them


## Training Curriculum

Worker 02 must implement and document the following curriculum:

### Stage 1: Base LM Pre-training
- **Goal:** Learn general acoustic/codec distributions and basic speaker/timbre consistency.
- **Data:** Massive raw audio (curated but potentially low-supervision).
- **Task:** Next-token prediction for codec tokens.
- **Few-shot requirement:** Stage 1 must already learn prompt-conditioned timbre anchoring from prompt-target pairs of the same speaker so `SpeakerPromptEncoder` and `speaker_embed` are part of the base path rather than a Stage 3 add-on.
- **Note:** VC path is primarily trained here.
- **VC training contract:**
  - VC reconstruction loss (source timbre → target timbre codec prediction) must be explicitly defined and independently switchable
  - VC-specific conditioning (source semantic features, optional speaker embed) must remain compatible with the v2 VC path to prevent regression
  - VC quality gates (speaker similarity, intelligibility) must be tracked per-stage so Stage 2/3 regressions are detectable
  - if VC shares the codec LM backbone with TTS, the loss weighting and batch mixing ratio between VC and TTS samples must be documented

### Stage 2: Alignment & Pointer Training
- **Goal:** Learn text-to-audio alignment and monotonic pointer progression.
- **Data:** High-quality transcribed subsets (curated).
- **Task:** Internal pointer learning with optional auxiliary monotonic regularization and optional bootstrap supervision, while preserving prompt-conditioned timbre behavior on the TTS path.
- **Training contract:** `latent_only` training must follow Worker 01's differentiable local-pointer surrogate. The model may use a soft expected mixture over only the current and next text unit during teacher forcing, but the exported/runtime pointer remains hard and discrete.
- **Default transitional artifact requirement:** The default Stage 2 recipe assumes canonical `bootstrap_alignment` is available as a transitional artifact for cold-start protection. A fully bootstrap-free Stage 2 recipe may exist as a research or fallback recipe, but it is not the default worker handoff unless Worker 02 explicitly redefines the curriculum and Worker 06 signs off on convergence.
- **Annealing Schedule (Cold-Start Protection):** To prevent pointer collapse during early training, the trainer MUST follow a strictly defined schedule:
  1. **Hard Bootstrap Phase:** Force the pointer to use external `bootstrap_alignment` (from ASR) as the ground truth for the first N steps.
  2. **Soft Transition Phase:** Mix `bootstrap_alignment` with the model's own `latent` predictions.
  3. **Latent-Only Phase:** Remove external alignment and train through the frozen differentiable local-pointer contract rather than through hidden discrete argmax behavior.
- **Note:** Duration-predictor logic is replaced by pointer-loss here.

### Stage 3: Dramatic & Dialogue Finetuning
- **Goal:** Learn drama-grade acting, context-sensitivity, and zero-shot disentanglement.
- **Data:** High-expressivity dialogue datasets (curated).
- **Task:** Dialogue-conditioned TTS with Prosody Predictor and CFG training.
- **Note:** **Classifier-Free Guidance (CFG)** training is introduced here by randomly dropping conditioning (10-20% dropout) to allow inference-time guidance scaling.
- **Note:** Stage 3 must train the exact pointer-aware CFG contract used at inference, including `advance_logit` / `progress_delta` guidance behavior.
- **Few-shot scope:** Stage 3 strengthens cross-context disentanglement and acting override, but must not be the first stage where prompt conditioning is learned.
- **Quality gate at Stage B completion:** Evaluate whether utterance-global prosody `[B, d_prosody]` is sufficient for drama acting quality. If anti-collapse metrics (`context_separation_score`, `prosody_collapse_score`) indicate insufficient local variation, schedule time-local prosody `[B, T_plan, d_prosody]` upgrade for Stage C.


## Concrete Tasks

1. Add config flags:
   - `tts_mode: legacy_duration | pointer`
   - `alignment_loss_type: none | aux_mas | aux_ctc`
   - `pointer_supervision_mode: latent_only | supervised`
   - `pointer_training_mode: local_expected | local_expected_st_hardening`
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
   - `pointer_hardening_start_step`
   - `pointer_hardening_ramp_steps`
   - `stage3_replay_mix_ratio`
   - `stage3_dialogue_mix_ratio`
   - **`conditioning_dropout_prob`** (for CFG training)
   - `conditioning_dropout_schema_version`
   - **`prosody_loss_weight`** (for Flow-matching)
   - `voice_state_loss_weight`
   - `voice_state_confidence_floor`
   - `voice_state_teacher_mode: none | pseudo_labeled | direct_labeled | mixed`
   - `suprasegmental_required_languages`
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
   - use the exact unconditional mask contract frozen by Worker 01
   - dropping only a subset of conditioning paths is forbidden unless it is an explicit ablation mode
   - trainer diagnostics must separately report acoustic-logit CFG behavior and pointer-output CFG behavior
10. Add CFG runtime-acceleration preparation:
    - define whether runtime may use `off | full | lazy | distilled` CFG modes
    - if `distilled` is supported, implement **CFG Self-Distillation** hooks in Stage 3, allowing the model to approximate 2-pass guided outputs in a 1-pass inference step, ensuring high expressive capacity fits within real-time VST budgets.
    - if `lazy` CFG is supported, define the training-time assumption or robustness check for skipped guidance refreshes
    - `off` and `full` are the v3.0 mainline requirements
    - `lazy` and `distilled` are post-v3.0 optimization tracks unless explicitly promoted into the release checklist
11. **Implement Flow-matching Loss for the Prosody Predictor.**
    - the ground-truth prosody target `[B, d_prosody]` is extracted by Worker 01's `ReferenceEncoder` from the target waveform during training
    - the `ProsodyPredictor` is trained to predict this target from text tokens, dialogue context, and speaker embed via a flow-matching objective
    - the `ReferenceEncoder` is trained jointly; Worker 02 must ensure gradients flow to it through the decoder's prosody-conditioning path
12. Add expressive-training hooks:
    - dialogue-context conditioning batch fields
    - utterance-level style / acting labels when available
    - optional local prosody latent supervision
    - optional `text_suprasegmentals`
    - optional `voice_state_targets`, `voice_state_observed_mask`, and `voice_state_confidence`
    - (CFG-based acting control is achieved via `cfg_scale` scalar and conditioning dropout, not a dedicated embedding)
13. Add anti-collapse training losses or diagnostics:
    - same-text different-context separation metric
    - prosody variance floor or diversity regularizer
14. **Add Zero-Shot/Few-Shot prompt training logic across Stage 1-3:**
    - dynamic prompt sampling: pick a random 3-10s clip from the same speaker to serve as `prompt_codec_tokens` and `speaker_embed` target.
    - Stage 1: train prompt-conditioned timbre anchoring and prompt-cache consumption.
    - Stage 2: retain prompt-conditioned timbre anchoring while pointer alignment is learned.
    - Stage 3: add cross-context / cross-emotion prompting so `dialogue_context` overrides prompt prosody.
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
- optional `text_suprasegmentals` aligned to `phoneme_ids`
- optional `acting_intent`
- optional `prosody_targets` (for Flow-matching)
- optional `acoustic_history_teacher` or equivalent teacher-forced codec/control history source
- optional `voice_state_targets`
- optional `voice_state_observed_mask`
- optional `voice_state_confidence`
- optional `voice_state_target_source`

## Alignment and Pointer Supervision Policy

This section defines both the allowed/forbidden pointer-target types and the supervision lifecycle. The two concerns are tightly coupled and must be read together.

### Allowed and Forbidden Pointer Targets

To avoid learning a robotic, uniform-duration rhythm, the model must **NOT** use naive uniform temporal distribution (`target_length // L`) as a release recipe.

- allowed:
  - `aux_mas`
  - `aux_ctc`
  - deterministic bootstrap projections exported by Worker 10
- forbidden as release truth:
  - uniform `target_length // L` targets
- research-only scaffolding:
  - a temporary heuristic bootstrap may exist for smoke-testing interfaces, but it must be labeled non-release and excluded from final convergence claims

### Supervision Lifecycle

- steady-state target:
  - pointer progression is learnable in `latent_only` mode with no external frame-level aligner dependency
- optional auxiliary target:
  - monotonic regularization may use `aux_mas` or `aux_ctc`
- transitional target:
  - supervised pointer loss may come from `bootstrap_projection` or `legacy_duration`
- alternative alignment-free candidate:
  - a progress-aware positional scheme may provide implicit alignment without explicit pointer targets (see Worker 01 § RoPE / pointer interaction)
- forbidden state:
  - supervised pointer loss enabled while pointer targets are absent
- release condition:
  - bootstrap and legacy-duration supervision must be demoted to transitional or ablation-only once the alignment-free pointer recipe is validated
  - a progress-aware positional alignment recipe is acceptable only if it meets the same quality gates
- recommended convergence recipe:
  - warm up pointer learning with auxiliary alignment regularization for the first configured step window
  - anneal auxiliary alignment toward zero before declaring the recipe alignment-free

### Tension with Alignment-Free Release Criterion

MAS-derived pointer targets are structurally a form of duration supervision. While they are superior to uniform distribution, relying on MAS indefinitely contradicts the plan's release exit criterion of at least one alignment-free recipe. Two resolution paths exist:

1. **Annealing path (current plan):** Use MAS targets during warmup, then anneal toward `latent_only` mode where the pointer head learns to align without external targets. This requires the pointer head to bootstrap alignment from the acoustic-text cross-attention signal alone.
2. **Progress-aware positional path (alternative):** If Worker 01 adopts a progress-aware positional scheme, alignment progress may be encoded directly into position structure and reduce the need for explicit pointer targets. This remains a research-track alternative until validated.

Worker 02 must validate at least one of these paths before release sign-off.

### Latent-Only Convergence Mechanism

In `latent_only` mode (no external frame-level alignment labels), the pointer head learns to advance through the following implicit supervision signals:

0. **Differentiable local-pointer surrogate:** The teacher-forced path must not use a hidden hard argmax. Instead, it uses Worker 01's local two-state pointer surrogate over the current and next text unit. This creates a defined gradient path from acoustic reconstruction into `advance_logit` / `progress_delta` without violating the hard runtime pointer contract.

1. **Reconstruction gradient:** When the local pointer mixture references the wrong region, the conditioned acoustic prediction degrades. The gradient from the acoustic reconstruction loss (`logits_a`, `logits_b`) propagates backward through the local text-conditioning mixture and cross-attention path, creating an implicit alignment signal for `advance_logit` and `progress_delta`.

2. **Monotonicity inductive bias:** The pointer is architecturally constrained to be monotonically non-decreasing with at most one text-unit advance per frame, and the differentiable surrogate is local to `{i_t, i_t+1}`. Combined with `progress_value` accumulation, the model only needs to learn *when* to advance, not arbitrary many-to-many text-audio alignment. This dramatically reduces the search space relative to unconstrained attention.

3. **Cross-attention entropy regularizer (optional, lightweight):** An optional low-weight regularizer may encourage peaky (low-entropy) cross-attention distributions during the annealing window. Unlike full MAS or CTC, this does not produce frame-level alignment labels; it only biases the attention pattern toward monotonic sharpness. This is distinct from the auxiliary alignment modes (`aux_mas`, `aux_ctc`) and does not constitute an external aligner dependency.

4. **Boundary confidence co-training:** The `boundary_confidence` head is trained jointly with acoustic prediction, creating an auxiliary signal that reinforces pointer advancement at natural phoneme/word boundaries detectable from the acoustic-text relationship.

5. **Scheduled hardening:** Before release sign-off, Stage 2 must transition from fully soft expected updates toward straight-through or sampled hard pointer updates so teacher-forced training does not drift arbitrarily far from runtime semantics.

**Convergence hypothesis:** The combination of the explicit differentiable local-pointer surrogate, acoustic reconstruction gradient through cross-attention, monotonic structural constraints, and scheduled hardening is expected to be sufficient for pointer convergence after auxiliary alignment is annealed to zero. This hypothesis must be validated empirically before release sign-off. If convergence fails, the annealing schedule must be extended or the progress-aware positional alternative must be evaluated.

### Suprasegmental Supervision Policy

- `phoneme_ids` remain the canonical text-unit ids
- `text_suprasegmentals` are companion per-unit features and must remain index-aligned with `phoneme_ids`
- Japanese / tonal-language recipes must report explicit coverage for these features
- if a sample belongs to a language/backend that declares suprasegmental support but the features are absent, training must either:
  - drop the sample from the mainline recipe, or
  - mark it as fallback-mode and exclude it from language-naturalness claims


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
- do not let CFG training and inference disagree on which conditioning fields are dropped
- do not treat low-confidence or missing `voice_state` targets as dense zero targets
- do not enable supervised pointer loss without a concrete bootstrap target source
- do not push ownership of canonical bootstrap projection into ad hoc dataset code
- do not silently train Japanese/tonal recipes from `phoneme_ids` alone while claiming suprasegmental support


## Handoff Contract

- training path runs in `pointer` mode without `durations.npy`
- legacy duration mode still works for comparison
- checkpoint saves all new pointer-head weights
- **CFG training capability confirmed.**
- **CFG v3.0 modes (`off`, `full`) trained and validated; post-v3.0 acceleration modes (`lazy`, `distilled`) deferred and documented as upgrade path, aligned with Worker 04.**
- **Flow-matching prosody prediction trained.**
- `voice_state` supervision path is documented and masked correctly when pseudo-label confidence is partial
- at least one documented recipe trains pointer mode without requiring external aligners at release time
- Stage 3 replay-mix policy is documented and prevents drama finetuning from collapsing base quality


## Required Tests

- trainer can step with `phoneme_ids` but no `durations`
- pointer supervised mode config fails fast when no valid pointer target source is configured
- legacy duration mode still computes duration loss
- latent-only pointer mode config is accepted without bootstrap labels
- auxiliary-alignment warmup / anneal schedule config test
- **CFG training test (ensure conditioning is dropped per configured probability).**
- CFG conditioning-schema parity test against Worker 01 contract
- CFG fast-path config validation test
- **Flow-matching loss test (ensure gradient flows to Prosody Predictor).**
- `voice_state` mask/confidence loss test
- config parsing tests for new flags
- quality gate tests renamed or expanded for new supervision semantics
- test that context-conditioned batches are accepted without breaking non-context batches
- test that diversity diagnostics run on same-text multi-context samples
- test that bootstrap alignment projection is monotonic and indexes canonical `phoneme_ids`
- Stage 3 replay-mix sampler test
