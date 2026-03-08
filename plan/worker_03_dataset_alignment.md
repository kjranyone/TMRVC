# Worker 03: Dataset Contract, Text Supervision, and Metrics

## Scope

Rewrite the dataset and cache assumptions so v3 training no longer depends on MFA-derived duration artifacts.


## Primary Files

- `tmrvc-data/src/tmrvc_data/uclm_dataset.py`
- `tmrvc-data/src/tmrvc_data/tts_dataset.py`
- `tmrvc-train/src/tmrvc_train/cli/train_uclm.py`
- `tmrvc-data/src/tmrvc_data/g2p.py`
- `tmrvc-data/src/tmrvc_data/alignment.py`
- `scripts/annotate/run_forced_alignment.py`
- tests under `tests/data/`, `tests/train/`, `tests/scripts/`


## Required Outcomes

- v3 dataset path treats `durations.npy` as optional
- unknown-phone coverage is measurable per dataset
- phone inventory mismatch is visible before training
- phone inventory migration policy is explicit
- MFA tooling is clearly legacy-only
- expressive supervision coverage is measurable before training starts
- local-model pseudo-annotation pipeline is defined for raw short-utterance corpora
- multilingual and code-switch readiness is measurable before training starts
- bootstrap alignment projection from ASR timestamps into canonical phoneme space is deterministic and measurable
- 8-D `voice_state` supervision readiness is measurable before training starts


## Concrete Tasks

1. Refactor dataset loading:
   - `phoneme_ids.npy` remains the canonical initial v3 trainable text-unit artifact whenever text supervision exists
   - `text_suprasegmentals.npy` is the canonical companion artifact for accent / tone / phrase-boundary cues when the text frontend provides them
   - transcript-only or alternate text-token records are not train-ready until they are normalized into the canonical text-unit contract
   - `durations.npy` never required in v3 mode
2. Add dataset-level supervision report utilities:
   - text coverage
   - phone unknown ratio
   - alignment artifact coverage
3. Add analysis CLI or script output for:
   - `unknown_phone_ratio`
   - per-dataset phone inventory
   - missing supervision counts
   - alias-hit statistics
   - unmapped-phone dump
4. Mark `run_forced_alignment.py` and `tmrvc_data.alignment` as `legacy v2 tooling` in code comments and docs if still retained.
5. If feasible, normalize current phone inventory to reduce false `UNK` inflation in existing data.
6. Freeze phone inventory migration policy:
   - v3 adopts the existing v2-compatible canonical phone id space
   - new annotation sources must map into that canonical inventory through normalization/alias tables
   - document how future phone additions remain append-only and do not renumber legacy ids
   - document checkpoint/cache migration consequences and why a fresh v3-only inventory is rejected for now
7. Add expressive-data readiness report:
   - dialogue-turn metadata availability
   - same-text multi-take coverage
   - emotion / style label availability
   - pause / breath / non-verbal event coverage
   - cross-lingual speaker coverage
   - code-switch span coverage
   - **Pitch Accent / Tone coverage (Critical for Japanese/Chinese naturalness)**

8. **Japanese G2P Policy Requirement:** The `tmrvc_data/g2p.py` backend must parse and retain **Pitch Accent information** (e.g., Upstep, Downstep) from the OpenJTalk fullcontext labels. Stripping accent data to raw phonemes (`a, i, u`) is explicitly forbidden for the v3 mainline, as it destroys the ability to learn natural Japanese prosody.
   - these cues must be exported into the canonical `text_suprasegmentals.npy` artifact aligned 1:1 with canonical text units rather than remaining trapped inside backend-specific fullcontext strings
9. Define curated-asset consumption contract for the data loader:
   - consume `text`, `language`, and `phoneme_ids` from exported cache
   - consume `text_suprasegmentals` when the exported text frontend declares suprasegmental support
   - consume `speaker_cluster` and `speaker_embedding` from metadata
   - consume optional `voice_state_targets`, `voice_state_observed_mask`, and `voice_state_confidence` from curated export
   - consume raw conversation graph / context text as the canonical dialogue-context source
   - allow optional derived context caches only when versioned by encoder/checkpoint hash
   - consume `quality_score` for dynamic batch filtering
10. Define quality filtering policy for the data loader:
   - allow runtime thresholding based on `quality_score`
   - support filtering by `provenance_class` or `legality_gate`
11. Define canonical phone alias mapping artifacts:
   - source phone symbol
   - normalized source symbol
   - canonical target symbol
   - source backend or provider
   - language scope
   - confidence or rule provenance
   - explicit `drop` / `map_to_unk` / `map_to_canonical` action
12. Define dataset reporting outputs for phone normalization quality:
   - `active_phone_inventory`
   - `alias_hit_ratio`
   - `direct_hit_ratio`
   - `unk_ratio`
   - `unmapped_phone_counts`
   - `top_unmapped_examples`
   - per-language breakdown for all of the above
13. Define multilingual / code-switch metadata artifacts:
   - utterance-level `language_id`
   - token-span or segment-level `language_spans`
   - prompt-language metadata for few-shot speaker prompts
   - cross-lingual train/eval split tags
14. Define long-horizon G2P fallback policy for multilingual robustness:
   - preserve normalized text alongside canonical phoneme ids
   - allow byte-level or grapheme-level fallback artifacts when G2P confidence is too low
   - mark fallback mode explicitly in metadata so training/validation can stratify on it
15. Define canonical bootstrap-alignment projection artifacts (Default transitional artifact for Stage 2; not a permanent mainline dependency):
   - preserve raw ASR token/word timestamps for provenance
   - deterministically project those timestamps onto canonical `phoneme_ids` where possible, or allow use of robust pre-trained aligners (e.g., Wav2Vec2) if they avoid MFA's heavy dependency footprint.
   - the projection algorithm must be frozen as one deterministic, versioned recipe rather than an implementation-defined heuristic
   - the recipe must specify feature extraction, parameter defaults, tie-break rules, low-confidence behavior, and exact fallback behavior when acoustic cues are unusable
   - export `bootstrap_alignment.json` with `text_unit_index`, `start_frame`, `end_frame`, `confidence`, and projection provenance
   - fail validation if projection skips, reorders, or ambiguously duplicates canonical text units
16. Define canonical `voice_state` supervision artifacts:
   - `voice_state.npy`
     - canonical shape: `[T_frames, 8]`
   - `voice_state_observed_mask.npy`
     - canonical shape: `[T_frames, 8]`
   - `voice_state_confidence.npy`
     - canonical shape: `[T_frames, 8]` or `[T_frames, 1]`
   - `voice_state_meta.json`
     - estimator identity
     - calibration version
     - provenance
     - whether labels are direct, pseudo-labeled, or absent
17. Define canonical few-shot prompt-pairing artifacts:
   - each record must either expose deterministic prompt eligibility metadata or explicit ineligibility
   - minimum required metadata:
     - canonical `speaker_id` or speaker-cluster id
     - `prompt_candidate_record_ids`
     - `prompt_language`
     - `prompt_duration_sec`
     - `speaker_purity_estimate`
     - `prompt_target_overlap_forbidden`
     - same-file / same-conversation policy flag
     - `prompt_selection_policy_version`
   - training and evaluation must be able to choose prompt clips without hidden trainer-only heuristics


## Important Design Decision

For v3, the dataset should distinguish:

- `text supervision available`
- `canonical text units available`
- `legacy duration supervision available`
- `no text supervision`

For drama-grade TTS, the dataset should also distinguish:

- `dialogue context available`
- `same-text multi-context coverage available`
- `acting labels available`
- `prosodic event supervision available`
- `curated provenance available`
- `code_switch_metadata_available`
- `cross_lingual_prompt_coverage_available`
- `voice_state_supervision_available`
- `voice_state_supervision_density`
- `voice_state_supervision_source`

Those are different states and must not be conflated.


## Phone Inventory Policy

Chosen policy: `v2-compatible canonical inventory`.

Rationale:

- the current codebase already has a unified multilingual `PHONE2ID` inventory in `tmrvc_data.g2p`
- existing caches already store `phoneme_ids.npy` against that inventory
- existing checkpoints and tests assume stable phoneme embedding semantics
- replacing the canonical inventory in v3 would create a broad checkpoint/cache migration burden without a demonstrated quality win

Implementation rules:

- preserve existing phone ids as the canonical semantic space for v3
- add normalization and alias mapping from new G2P / ASR / annotation sources into that canonical space
- if future phones must be added, add them append-only and never renumber existing ids
- keep `phoneme_vocab_size` capacity management separate from the active canonical symbol count
- treat the active symbol inventory defined in `tmrvc_data.g2p` as the source of truth for actual ids; overprovisioned embedding capacity is not a reason to redefine ids

Capacity expansion trigger:

- `configs/constants.yaml` defines `phoneme_vocab_size` with a minimum headroom of 20 symbols above the active canonical symbol count
- if `active_phone_inventory` count exceeds `(phoneme_vocab_size - 20)`, Worker 03 must file a capacity-expansion request before new languages or annotation backends are added
- capacity expansion is append-only and requires checkpoint migration documentation

Non-goal for initial v3:

- do not introduce a fresh v3-only phone inventory unless the existing canonical space is proven insufficient by measured coverage failures


## Canonical Text-Unit Policy

Chosen policy: `phoneme_ids` remain the initial v3 mainline text-unit ids, paired with optional-but-canonical companion suprasegmental features.

Rationale:

- Worker 01 and Worker 02 require a stable, causal, serializable text-unit contract
- the current codebase and checkpoints are built around canonical phoneme ids
- allowing multiple interchangeable mainline text-unit contracts at this stage would create avoidable worker drift

Rules:

- if transcript text exists, the dataset must either:
  - convert it into canonical `phoneme_ids` plus `text_suprasegmentals` when the language/backend supports them, or
  - mark the record as not yet train-ready for mainline pointer TTS
- alternate text-token backends may exist only as explicit ablation or future-extension paths
- dataset reports must distinguish `text_supervision_coverage` from `canonical_text_unit_coverage`
- dataset reports must separately expose `suprasegmental_coverage` for languages/backends that declare support
- retain normalized text so later fallback or dual-input experiments remain possible without rebuilding the whole dataset

### Canonical Suprasegmental Artifact Policy

- required artifact name:
  - `text_suprasegmentals.npy`
- canonical shape:
  - `[L, d_supra]`
- canonical alignment:
  - row `i` must align exactly to `phoneme_ids[i]`
- canonical metadata:
  - feature schema version
  - language/backend identity
  - whether features are direct, projected, or absent
- example dimensions:
  - Japanese:
    - `accent_upstep`
    - `accent_downstep`
    - `accent_phrase_break`
  - tonal languages:
    - normalized lexical tone id or equivalent tone feature
- forbidden behavior:
  - storing accent/tone only inside backend-specific strings while exporting bare `phoneme_ids`
  - losing unit alignment during cache export/import


## G2P Fallback Policy

Worker 03 must define a safety path for multilingual and code-switch instability.

- preferred path:
  - normalized text -> canonical `phoneme_ids` + `text_suprasegmentals` when available
- fallback path when G2P is weak or ambiguous:
  - normalized text + explicit fallback marker
  - optional byte/grapheme-level side input artifact for later-stage experiments
  - runtime policy must be explicit and shared across Python/Rust:
    - for the initial v3 mainline TTS contract, if canonical `phoneme_ids` cannot be produced with sufficient confidence, the request/record is downgraded or rejected rather than silently synthesized into a divergent runtime-only text representation
    - any accepted fallback mode must carry an explicit `fallback_mode` tag that survives dataset export, training, serving, and validation
    - Rust/VST must follow the same canonical accept/reject/downgrade rule as Python serve; a Rust-only text frontend contract is forbidden
- forbidden behavior:
  - silently emitting large `UNK` spans without preserving enough text to recover later

This fallback is a Stage B-or-later safety valve, not permission to abandon the canonical phoneme contract in initial mainline training.


## Bootstrap Alignment Projection Policy

Bootstrap alignment is transitional supervision, not a hidden replacement for the pointer model.

- canonical source:
  - normalized text
  - canonical `phoneme_ids`
  - ASR token/word timestamps retained for provenance
- required derived artifact:
  - `bootstrap_alignment.json` already indexed in canonical phoneme space
  - projection must use one frozen deterministic algorithm version rather than vague "acoustic-aware heuristics"
  - the algorithm spec must include:
    - algorithm id
    - parameter set and defaults
    - feature extraction recipe
    - tie-break rules
    - behavior on low-confidence spans
    - fallback rule when acoustic cues are unusable
  - any change to these items requires a version bump in exported provenance
  - frame convention is frozen:
    - `sample_rate = 24000`
    - `hop_length = 240`
    - `start_frame` is inclusive
    - `end_frame` is exclusive
    - utterance frame count must satisfy `T = ceil(num_samples / 240)`
    - all exported frame indices must match `tmrvc-core` frame-alignment tests exactly
- ownership split:
  - Worker 03 defines deterministic projection and validation rules
  - Worker 10 exports the validated artifact
  - Worker 02 consumes it and may densify it to frame-level loss targets
- forbidden behavior:
  - handing Worker 02 raw ASR timestamps and leaving phoneme projection implicit
  - exporting labels produced by an implementation-defined heuristic with no algorithm/version identity

## Few-Shot Prompt Eligibility Policy

Worker 03 must define a canonical dataset-level contract for promptable speaker evidence.

- each train/eval record must either:
  - declare one or more valid prompt candidates, or
  - declare explicit absence / ineligibility
- the canonical policy must freeze:
  - whether prompt and target may come from the same source file
  - whether prompt and target may come from the same conversation window
  - minimum and maximum prompt duration
  - minimum speaker-purity threshold
  - cross-lingual prompt eligibility tagging
  - deterministic ordering / sampling key so repeated runs can reproduce prompt choice
- forbidden behavior:
  - ad hoc "pick another clip from the same speaker" logic living only in the trainer
  - evaluation-time prompt sampling that can accidentally choose the target segment or a leakage-equivalent near-duplicate


## Alias Mapping Specification

Worker 03 must define a deterministic alias-mapping layer between external annotation outputs and canonical `PHONE2ID`.

Minimum artifact format:

- table file:
  - recommended path:
    - `configs/phoneme_aliases.yaml`
- per-entry fields:
  - `source_symbol`
  - `normalized_symbol`
  - `canonical_symbol`
  - `language`
  - `source_backend`
  - `action`
    - `map`
    - `drop`
    - `unk`
  - `note` or provenance field

Normalization stages:

1. Unicode normalization
2. backend-specific cleanup
3. alias-table lookup
4. canonical-symbol validation
5. fallback to `UNK_ID` only if no explicit safe mapping exists

Rules:

- alias mapping must be deterministic and testable
- backend-specific phones must never bypass canonical validation
- `UNK` fallback must be measurable and reported, not silent
- dropping phones is allowed only for explicit noise/prosody artifacts that are not part of the canonical semantic inventory


## Dataset Report Specification

Worker 03 must produce a machine-readable supervision and normalization report per dataset.

Minimum report fields:

- `dataset_name`
- `split`
- `num_utterances`
- `text_supervision_coverage`
- `canonical_text_unit_coverage`
- `legacy_duration_coverage`
- `unknown_phone_ratio`
- `direct_hit_ratio`
- `alias_hit_ratio`
- `active_phone_inventory`
- `unmapped_phone_counts`
- `top_unmapped_examples`
- `per_language_stats`
- `dialogue_context_coverage`
- `same_text_multi_context_coverage`
- `code_switch_coverage`
- `cross_lingual_prompt_coverage`
 - `prompt_pairing_coverage`
 - `prompt_leakage_risk_count`
- `g2p_fallback_coverage`
- `voice_state_supervision_coverage`
- `voice_state_observed_ratio`
- `voice_state_confidence_summary`

Interpretation requirements:

- `direct_hit_ratio`
  - fraction of phones already in canonical inventory before alias mapping
- `alias_hit_ratio`
  - fraction resolved by alias rules
- `unknown_phone_ratio`
  - fraction still unresolved after normalization and alias mapping
- `unmapped_phone_counts`
  - exact counts by raw source symbol so inventory gaps can be fixed intentionally
- `canonical_text_unit_coverage`
  - fraction of records that already satisfy the canonical `phoneme_ids` contract required by Worker 02
- `code_switch_coverage`
  - fraction of records with valid token-span or segment-level language annotations
- `cross_lingual_prompt_coverage`
  - fraction of speakers or records usable for prompt-language != target-language evaluation
- `prompt_pairing_coverage`
  - fraction of records with at least one valid prompt candidate under the frozen few-shot policy
- `prompt_leakage_risk_count`
  - count of records whose only available prompt candidates violate same-target or leakage guardrails
- `g2p_fallback_coverage`
  - fraction of records that required normalized-text or byte/grapheme fallback because canonical G2P was too weak

Fail / warn policy:

- hard fail:
  - if canonical validation is skipped
  - if report generation is missing
- warning:
  - if `unknown_phone_ratio` exceeds project threshold
  - if one language dominates unmapped_phone_counts
  - if alias-hit ratio is unexpectedly high, implying backend drift or poor normalization


## Guardrails

- do not implement the curation pipeline in this worker; consume its exported assets
- do not silently fallback to heuristic durations in v3 mode
- do not let legacy alignment files leak into required batch fields
- keep legacy analysis tools loadable for comparison
- do not call a dataset drama-ready if it only has plain read speech
- do not call a dataset physical-control-ready unless `voice_state` supervision density and confidence are reported
- do not trust ASR output blindly; assume the curation system (Worker 11) has already validated it
- do not conflate per-file diarization labels with dataset-global speaker identity
- do not leave v2 versus v3 phone inventory migration unspecified
- do not leave transcript-supervised but non-normalized records ambiguous; they are not mainline-trainable until canonical text units exist
- do not claim multilingual readiness from utterance-level language labels alone if code-switching is in scope
- do not silently hide G2P instability behind `UNK`; preserve normalized text and fallback markers
- do not make model-coupled pre-encoded dialogue embeddings part of the canonical dataset contract
- do not export bootstrap alignment that is not already projected into canonical phoneme-index space


## Handoff Contract

- worker 02 can train pointer mode from dataset output using assets exported by Worker 10
- worker 05 can expose correct supervision status in `dev.py`
- worker 06 can assert phone-coverage metrics in tests


## Required Tests

- dataset returns valid TTS sample without `durations.npy` using curated cache
- dataset report distinguishes `text_supervision_coverage` from `canonical_text_unit_coverage`
- unknown-phone metric test
- supervision coverage summary test
- phone inventory migration-policy test or fixture
- alias-mapping determinism test
- dataset report field-completeness test
- legacy forced-alignment script still works in isolated legacy path
- expressive-readiness report test
- quality-score based filtering test
- cross-file speaker clustering field presence test
- multilingual/code-switch report field-completeness test
- G2P fallback reporting test
- bootstrap-alignment projection determinism test
- bootstrap-alignment frame-convention parity test against `tmrvc-core`
- `voice_state` artifact shape/reporting test
- few-shot prompt-eligibility policy determinism test
- few-shot prompt leakage-guard test
- canonical context-graph fields present even when no derived context cache is exported
