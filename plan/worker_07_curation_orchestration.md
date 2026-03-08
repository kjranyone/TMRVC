# Worker 07: Curation Orchestration and Manifest Contract

## Purpose

Define the execution backbone for the AI curation system.

## Intent

This worker makes the curation system resumable, parallelizable, and auditable. It does not chase model quality directly. It defines how every other curation worker plugs into one stable manifest and stage contract.

## Primary Files

- new curation orchestration module under `tmrvc-data`
- new curation CLI entrypoint
- `dev.py`
- `plan/ai_curation_system.md`
- `docs/design/auth-spec.md` (Optimistic Locking & Audit Foundation)
- `docs/design/curation-contract.md` (Manifest / Promotion / Export contract consolidation; this document must remain consistent with Worker 07-11 plan files)

## Required Outcomes

- stable `manifest.jsonl` schema
- resumable stage execution
- pass index tracking
- per-stage retry support
- provider provenance written for every stage
- human action provenance written for every audit-critical state change
- **Concurrency Control:** Implement optimistic locking via record-level versioning to support multi-user WebUI workflows.
- one authoritative API contract between multi-user WebUI and curation state; direct manifest file reads are dev-only
- ownership boundary is explicit:
  - Worker 07 owns curation storage, record versioning, and mutation semantics
  - Worker 04 owns the HTTP `/ui/*` and `/admin/*` transport and middleware that expose those semantics
  - Worker 12 is an API consumer, not a second backend
- if post-v3.0 Personal Voice Training is retained, Worker 07 also owns the lightweight enrollment-preparation workflow that converts user-uploaded audio into canonical trainable artifacts before any fine-tune job may start

## Manifest Contract

The system handles tens of millions of records (100k+ hours). To ensure performance:

- **Storage Strategy:** `manifest.jsonl` is the canonical **interchange and snapshot format**. However, the authoritative curation service must use **SQLite** as the operational DB for live operations, audit trails, and concurrent WebUI access. SQLite is chosen for its high performance with multi-million record datasets, single-file portability, and zero-config deployment. **WAL mode must be enabled** to support concurrent reads during multi-user WebUI workflows; connection pooling or serialized-write queue strategy must be documented before multi-user deployment.
- Every record must include:
  - source path
  - utterance or segment id
  - segment bounds
  - duration
  - current curation status
  - **`metadata_version` (incrementing integer for optimistic locking, see `docs/design/auth-spec.md`)**
  - current quality score
  - `score_components` for auditable sub-scores and reject/review diagnostics
  - canonical provider decision fields needed for reproducibility:
    - `provider_id`
    - `provider_revision`
    - `calibration_version`
    - `fallback_class` when a downgrade path was used
  - stage outputs
  - stage confidences
  - stage provenance
  - conversation graph links
  - source legality state
  - pass history
  - promotion / rejection reasons
  - human action history (actor ID, timestamp, rationale)

## Concrete Tasks

1. Define top-level commands:
   - ingest
   - run-stage
   - resume
   - score
   - promote
   - export
2. **Implement SQLite Storage Layer:**
   - develop a `CurationDataService` that manages the transition between `manifest.jsonl` snapshots and the operational **SQLite DB**.
   - ensure high-speed indexing for `record_id`, `status`, `speaker_cluster`, and `quality_score`.
   - implement batch-update capabilities to handle large-scale provider outputs efficiently.
3. **Implement Optimistic Locking:**
   - define `metadata_version` in the manifest schema and SQLite table
   - implement version-aware `update_record` API in the curation service
   - expose version-check results to Worker 04 as typed domain errors rather than raw storage exceptions
   - ensure Gradio UI (Worker 12) sends the current version during submission
4. Define stage-addressable execution so later workers can rerun only:
   - separation
   - diarization
   - ASR
   - refinement
   - prosody / event extraction
   - lightweight enrollment preparation (`VAD -> ASR/transcript check -> G2P -> boundary/alignment refinement`) for user-uploaded personal-voice data
5. Define retry behavior:
   - transient failure
   - provider unavailable
   - low-confidence output
6. Define pass lifecycle:
   - pass 0 = ingest
   - pass N = refinement pass
7. Define immutable split and export gating fields:
   - train / holdout membership
   - legality gate
   - bucket eligibility
8. Define audit-trail fields for UI and policy enforcement:
   - actor role
   - actor id
   - timestamp
   - action type
   - before / after state summary
   - rationale or note
9. Wire `dev.py` eventually to call this system as a first-class operation.
10. Publish the authoritative service boundary:
   - `tmrvc-serve` hosts the typed `/ui/*` and `/admin/*` network APIs
   - those APIs must call Worker 07-owned storage/mutation services for curation data rather than reimplementing record semantics in route handlers
   - filesystem-direct manifest access is allowed only for local debugging, never as the mainline multi-user contract
11. Define post-v3.0 enrollment-to-training handoff contract:
   - uploaded personal-voice audio must first become canonical curation records
   - the preparation job must emit the same train-ready artifacts used by Worker 02 / Worker 03 (`normalized text`, `phoneme_ids`, optional `text_suprasegmentals`, bootstrap/boundary artifact as required by the selected recipe)
   - low-confidence transcript/G2P/alignment items must enter a review queue rather than flowing silently into training
   - Worker 12 may trigger this flow, but it must not bypass Worker 07 storage/provenance/versioning

## Guardrails

- do not hardcode one provider as permanent truth
- do not let providers write incompatible ad hoc sidecars
- do not use implicit state hidden outside the manifest
- do not leave promotion/export decisions without durable human-action provenance when they are human-driven
- **Do not** allow record updates that bypass the `metadata_version` check.
- do not let Worker 12 build a second authoritative data plane around direct manifest reads
- do not let post-v3.0 personal-voice training consume raw wav uploads directly; training inputs must come through the same canonical preparation/provenance path as other trainable data

## Handoff Contract

- worker 08 can plug providers into a stable schema
- worker 09 can score and promote records without guessing fields
- worker 10 can export promoted subsets without reverse engineering
