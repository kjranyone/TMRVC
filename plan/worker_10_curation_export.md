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
- provider provenance
- promotion bucket

Optional:

- pause events
- breath events
- style embedding
- same-text cluster

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
5. Add `dev.py` operations:
   - run curation
   - resume curation
   - export promoted subset
   - show curation summary
6. Define how later training quality gates read curation provenance and legality.
7. Implement pre-encoding of dialogue context:
   - optionally run the Text Encoder on preceding turns and export embeddings (`.npy`) to avoid runtime bottlenecks during training.
8. Export ASR-derived alignment for bootstrap:
   - export token-level or word-level timestamps from Stage 4/5 as a `bootstrap_alignment.json` or equivalent.
   - ensure these labels are available to Worker 02 as a `pointer_target_source`.

## Guardrails

- do not drop provenance during export
- do not export review items into train buckets
- do not make export depend on legacy MFA artifacts
- do not drop conversation graph fields needed for dialogue-conditioned training

## Handoff Contract

- training workers can ingest curated subsets cleanly
- validation workers can trace evaluation failures back to curation history
