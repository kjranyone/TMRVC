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

## Required Outcomes

- stable `manifest.jsonl` schema
- resumable stage execution
- pass index tracking
- per-stage retry support
- provider provenance written for every stage
- human action provenance written for every audit-critical state change

## Manifest Contract

Each record must include:

- source path
- utterance or segment id
- segment bounds
- duration
- current curation status
- current quality score
- stage outputs
- stage confidences
- stage provenance
- conversation graph links
- source legality state
- pass history
- promotion / rejection reasons
- human action history

## Concrete Tasks

1. Define top-level commands:
   - ingest
   - run-stage
   - resume
   - score
   - promote
   - export
2. Define stage-addressable execution so later workers can rerun only:
   - separation
   - diarization
   - ASR
   - refinement
   - prosody / event extraction
3. Define retry behavior:
   - transient failure
   - provider unavailable
   - low-confidence output
4. Define pass lifecycle:
   - pass 0 = ingest
   - pass N = refinement pass
5. Define immutable split and export gating fields:
   - train / holdout membership
   - legality gate
   - bucket eligibility
6. Define audit-trail fields for UI and policy enforcement:
   - actor role
   - actor id
   - timestamp
   - action type
   - before / after state summary
   - rationale or note
7. Wire `dev.py` eventually to call this system as a first-class operation.

## Guardrails

- do not hardcode one provider as permanent truth
- do not let providers write incompatible ad hoc sidecars
- do not use implicit state hidden outside the manifest
- do not leave promotion/export decisions without durable human-action provenance when they are human-driven

## Handoff Contract

- worker 08 can plug providers into a stable schema
- worker 09 can score and promote records without guessing fields
- worker 10 can export promoted subsets without reverse engineering
