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


## Concrete Tasks

1. Refactor dataset loading:
   - `phoneme_ids.npy` remains the canonical initial v3 trainable text-unit artifact whenever text supervision exists
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
8. Define the loading contract for curated subsets:
   - consume `text`, `language`, and `phoneme_ids` from exported cache
   - consume `speaker_cluster` and `speaker_embedding` from metadata
   - consume raw conversation graph / context text as the canonical dialogue-context source
   - allow optional derived context caches only when versioned by encoder/checkpoint hash
   - consume `quality_score` for dynamic batch filtering
9. Define quality filtering policy for the data loader:
   - allow runtime thresholding based on `quality_score`
   - support filtering by `provenance_class` or `legality_gate`
10. Define canonical phone alias mapping artifacts:
   - source phone symbol
   - normalized source symbol
   - canonical target symbol
   - source backend or provider
   - language scope
   - confidence or rule provenance
   - explicit `drop` / `map_to_unk` / `map_to_canonical` action
11. Define dataset reporting outputs for phone normalization quality:
   - `active_phone_inventory`
   - `alias_hit_ratio`
   - `direct_hit_ratio`
   - `unk_ratio`
   - `unmapped_phone_counts`
   - `top_unmapped_examples`
   - per-language breakdown for all of the above
12. Define multilingual / code-switch metadata artifacts:
   - utterance-level `language_id`
   - token-span or segment-level `language_spans`
   - prompt-language metadata for few-shot speaker prompts
   - cross-lingual train/eval split tags
13. Define long-horizon G2P fallback policy for multilingual robustness:
   - preserve normalized text alongside canonical phoneme ids
   - allow byte-level or grapheme-level fallback artifacts when G2P confidence is too low
   - mark fallback mode explicitly in metadata so training/validation can stratify on it
14. Define canonical bootstrap-alignment projection artifacts:
   - preserve raw ASR token/word timestamps for provenance
   - deterministically project those timestamps onto canonical `phoneme_ids`
   - export `bootstrap_alignment.json` with `text_unit_index`, `start_frame`, `end_frame`, `confidence`, and projection provenance
   - fail validation if projection skips, reorders, or ambiguously duplicates canonical text units


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

Non-goal for initial v3:

- do not introduce a fresh v3-only phone inventory unless the existing canonical space is proven insufficient by measured coverage failures


## Canonical Text-Unit Policy

Chosen policy: `phoneme_ids` remain the initial v3 mainline text-unit interface.

Rationale:

- Worker 01 and Worker 02 require a stable, causal, serializable text-unit contract
- the current codebase and checkpoints are built around canonical phoneme ids
- allowing multiple interchangeable mainline text-unit contracts at this stage would create avoidable worker drift

Rules:

- if transcript text exists, the dataset must either:
  - convert it into canonical `phoneme_ids`, or
  - mark the record as not yet train-ready for mainline pointer TTS
- alternate text-token backends may exist only as explicit ablation or future-extension paths
- dataset reports must distinguish `text_supervision_coverage` from `canonical_text_unit_coverage`
- retain normalized text so later fallback or dual-input experiments remain possible without rebuilding the whole dataset


## G2P Fallback Policy

Worker 03 must define a safety path for multilingual and code-switch instability.

- preferred path:
  - normalized text -> canonical phoneme ids
- fallback path when G2P is weak or ambiguous:
  - normalized text + explicit fallback marker
  - optional byte/grapheme-level side input artifact for later-stage experiments
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
  - projection must use **Acoustic-Aware Phoneme Boundary Heuristics** (e.g., energy flux or spectral changes) rather than naive uniform time-splitting, to provide sharper initial targets for pointer learning.
- ownership split:
  - Worker 03 defines deterministic projection and validation rules
  - Worker 10 exports the validated artifact
  - Worker 02 consumes it and may densify it to frame-level loss targets
- forbidden behavior:
  - handing Worker 02 raw ASR timestamps and leaving phoneme projection implicit


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
- `g2p_fallback_coverage`

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
- canonical context-graph fields present even when no derived context cache is exported
