# Worker 10: Export, Cache Integration, and Tooling

## Purpose

Make curated outputs directly consumable by TMRVC training and operations.

## Intent

Curation is not finished when labels exist. It is finished when the promoted subset can enter the training stack without ad hoc conversion and without losing provenance.

## Primary Files

- export module under `tmrvc-data`
- cache writers
- `dev.py`
- training-side dataset contracts where needed

## Required Outcomes

- promoted curation subsets can export to cache-compatible assets
- provenance survives export
- `dev.py` can launch curation and export flows
- mainline training can distinguish curated provenance classes
- bootstrap alignment export is already projected into canonical phoneme space
- dialogue-context export remains model-agnostic by default
- export and packaging flows required by humans are triggerable from the WebUI without shell access
- `voice_state` supervision artifacts survive export with masks, confidences, and provenance

## Export Targets

### Target A: Curation manifest

For further refinement and audits.

### Target B: TMRVC cache-ready subset

For direct use in:

- pointer TTS
- VC prior
- expressive prior

### Target C: Evaluation subset package

For reproducible held-out tests.

## Required Export Fields

- transcript
- language
- text units
- speaker metadata
- conversation metadata
- source legality
- quality score
- `score_components`
- provider provenance
- provider decision provenance:
  - `provider_id`
  - `provider_revision`
  - `calibration_version`
  - optional `fallback_class`
- promotion bucket
- `voice_state` supervision status
- few-shot prompt eligibility metadata

Optional:

- pause events
- breath events
- style embedding
- same-text cluster
- `voice_state` targets when not required by the destination bucket

## Concrete Tasks

1. Define export format from manifest to cache.
2. Define bucket-specific export behavior:
   - TTS mainline requires text units
   - VC prior may not require transcript certainty at the same threshold
3. Define how exported metadata is embedded in `meta.json`.
4. Export dialogue graph fields required for context-conditioned batching:
   - `conversation_id`
   - `turn_index`
   - `prev_record_id`
   - `next_record_id`
   - `context_window_ids`
5. Request `dev.py` entrypoints from Worker 05:
   - Worker 05 owns `dev.py` implementation; Worker 10 defines the required export-side commands and their arguments
   - required commands: run curation, resume curation, export promoted subset, show curation summary
   - these commands must call through the authoritative backend API defined by Worker 04 / Worker 07
6. Add WebUI-facing export actions:
   - materialize promoted subset
   - package holdout evaluation bundle
   - download or register exported artifacts for downstream training/eval
   - return structured payloads with artifact ids, download urls, and provenance summary
7. Define how later training quality gates read curation provenance and legality.
   - export must preserve the exact provider/calibration fields used by Worker 09 promotion policy so release bundles remain reproducible
8. Export dialogue context in a model-agnostic way:
   - raw context text, turn graph, and canonical text units are the default export
   - optional derived context embeddings may be materialized only as checkpoint-hashed caches and must be invalidatable
9. Export ASR-derived alignment for bootstrap:
   - preserve token-level or word-level timestamps from Stage 4/5 for provenance
   - export `bootstrap_alignment.json` already projected onto canonical `phoneme_ids` with `text_unit_index`, `start_frame`, `end_frame`, `confidence`, and projection provenance.
   - include `projection_algorithm_id`, `projection_algorithm_version`, and a parameter/config fingerprint so the projection is replayable.
   - ensure these labels are available to Worker 02 as a supervised `pointer_target_source`
   - freeze frame convention:
     - `sample_rate = 24000`
     - `hop_length = 240`
     - `start_frame` inclusive
     - `end_frame` exclusive
     - `T = ceil(num_samples / 240)`
10. Export physical-control supervision:
   - `voice_state.npy`
   - `voice_state_observed_mask.npy`
   - `voice_state_confidence.npy`
   - `voice_state_meta.json`
   - preserve estimator identity, calibration version, and target-source provenance
11. Export canonical few-shot prompt metadata:
   - promptable records must carry:
     - `speaker_id`
     - `prompt_candidate_record_ids`
     - `prompt_policy_version`
     - `prompt_duration_sec` summary
     - `prompt_language`
     - `speaker_purity_estimate`
     - leakage-policy flags
   - holdout bundles must freeze the exact prompt-target pairings used in evaluation so external-baseline comparisons are reproducible
12. Define artifact package contract:
   - every exported package must include:
     - `artifact_id`
     - `artifact_type`
     - `created_at`
     - `source_dataset_ids`
     - `manifest_snapshot_id`
     - `policy_version`
     - `provenance_summary`
     - `retention_class`
   - package formats must be explicit for:
     - cache-ready training bundle
     - holdout evaluation bundle
     - pinned workshop take bundle
13. Define artifact lifecycle and cleanup policy:
   - `ephemeral`
   - `durable`
   - `release_candidate`
   - who may delete each class
   - whether download URLs are time-limited
14. Define export failure / retry semantics:
   - partial package cleanup rules
   - idempotent retry behavior keyed by manifest snapshot and export intent
   - WebUI-visible failure payload with actionable remediation
15. Define browser-safe artifact handoff:
   - download for human operators
   - server-side registration for training/eval jobs
   - checksum display and verification status in WebUI
16. Post-v3.0 only: user voice adaptor export
   - if adaptor/LoRA export remains in scope, move it behind the post-v3.0 Training Cockpit / production-export workstream
   - do not let adaptor-merging or ONNX baking block the mainline curated-data export contract

## Guardrails

- do not drop provenance during export
- do not export review items into train buckets
- do not make export depend on legacy MFA artifacts
- do not drop conversation graph fields needed for dialogue-conditioned training
- do not make model-dependent context embeddings the canonical export contract
- do not export bootstrap alignment that still requires downstream phoneme projection guesswork
- do not export `voice_state` supervision without masks and provenance
- do not export provider-driven scores or decisions without the pinned provider/calibration provenance that produced them
- do not make shell access a prerequisite for exporting, packaging, or downloading curated outputs
- do not produce opaque artifact directories that the WebUI cannot describe or audit

## Handoff Contract

- training workers can ingest curated subsets cleanly
- validation workers can trace evaluation failures back to curation history
- worker 12 can present export/download state without inventing artifact metadata

## Required Tests

- export preserves bootstrap-alignment projection provenance fields
- export preserves few-shot prompt eligibility metadata
- holdout bundle freezes prompt-target pairings reproducibly
- export fails closed when projection provenance is missing
