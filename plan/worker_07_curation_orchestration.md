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

## Required Outcomes

- stable `manifest.jsonl` schema
- resumable stage execution
- pass index tracking
- per-stage retry support
- provider provenance written for every stage
- human action provenance written for every audit-critical state change
- **Concurrency Control:** Implement optimistic locking via record-level versioning to support multi-user WebUI workflows.
- one authoritative API contract between multi-user WebUI and curation state; direct manifest file reads are dev-only

## Manifest Contract

Each record must include:

- source path
- utterance or segment id
- segment bounds
- duration
- current curation status
- **`metadata_version` (incrementing integer for optimistic locking, see `docs/design/auth-spec.md`)**
- current quality score
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
2. **Implement Optimistic Locking:**
   - define `metadata_version` in the manifest schema
   - implement version-aware `update_record` API in the curation service
   - ensure Gradio UI (Worker 12) sends the current version during submission
3. Define stage-addressable execution so later workers can rerun only:
   - separation
   - diarization
   - ASR
   - refinement
   - prosody / event extraction
4. Define retry behavior:
   - transient failure
   - provider unavailable
   - low-confidence output
5. Define pass lifecycle:
   - pass 0 = ingest
   - pass N = refinement pass
6. Define immutable split and export gating fields:
   - train / holdout membership
   - legality gate
   - bucket eligibility
7. Define audit-trail fields for UI and policy enforcement:
   - actor role
   - actor id
   - timestamp
   - action type
   - before / after state summary
   - rationale or note
8. Wire `dev.py` eventually to call this system as a first-class operation.
9. Publish the authoritative service boundary:
   - `tmrvc-serve` hosts the typed `/ui/*` and `/admin/*` network APIs
   - filesystem-direct manifest access is allowed only for local debugging, never as the mainline multi-user contract

## Guardrails

- do not hardcode one provider as permanent truth
- do not let providers write incompatible ad hoc sidecars
- do not use implicit state hidden outside the manifest
- do not leave promotion/export decisions without durable human-action provenance when they are human-driven
- **Do not** allow record updates that bypass the `metadata_version` check.
- do not let Worker 12 build a second authoritative data plane around direct manifest reads

## Handoff Contract

- worker 08 can plug providers into a stable schema
- worker 09 can score and promote records without guessing fields
- worker 10 can export promoted subsets without reverse engineering
