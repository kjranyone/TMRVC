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
- every curation operation needed by humans is invocable through a WebUI-safe API, not only through CLI entrypoints

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

### JSONL Record Format

Each line of `manifest.jsonl` is a JSON object with the following canonical field names (matching `ai_curation_system.md` minimum manifest fields):

```json
{
  "record_id": "uuid-or-deterministic-hash",
  "source_path": "/data/raw/file001.wav",
  "audio_hash": "sha256:...",
  "segment_start_sec": 0.0,
  "segment_end_sec": 3.5,
  "duration_sec": 3.5,
  "language": "ja",
  "transcript": "...",
  "transcript_confidence": 0.95,
  "speaker_cluster": "spk_042",
  "diarization_confidence": 0.88,
  "conversation_id": "conv_001",
  "turn_index": 3,
  "prev_record_id": "...",
  "next_record_id": "...",
  "context_window_ids": ["...", "..."],
  "quality_score": 0.91,
  "status": "promoted",
  "promotion_bucket": "tts_mainline",
  "rejection_reasons": [],
  "review_reasons": [],
  "providers": {"asr": "VibeVoice-ASR/v1.2", "diarization": "pyannote/3.1"},
  "pass_index": 1,
  "source_legality": "licensed",
  "human_actions": [
    {"actor_role": "auditor", "actor_id": "user_a", "timestamp": "...", "action": "approve", "note": "..."}
  ]
}
```

All downstream workers (08-12) must produce and consume records conforming to this schema. Fields may be `null` for stages not yet completed, but the field keys must always be present.

## Concrete Tasks

1. Define top-level commands:
   - ingest
   - run-stage
   - resume
   - score
   - promote
   - export
   - each command must also map to a stable WebUI action / API contract
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
8. Define WebUI-safe orchestration endpoints for:
   - dataset upload / registration
   - stage start / stop / resume
   - score / promote / export
   - progress polling and failure inspection
9. Define orchestration payload schemas for the WebUI:
   - upload job status
   - curation run status
   - stage progress event
   - failure payload with actionable retry information
   - manifest query filter payload
10. Define canonical job-state machine shared with Worker 12:
   - `queued`
   - `running`
   - `blocked_human`
   - `failed_retryable`
   - `failed_terminal`
   - `completed`
   - `canceled`
11. Define record concurrency semantics:
   - soft lock acquisition on review open
   - lock timeout / heartbeat policy
   - forced takeover with admin-only audit note
   - optimistic `object_version` on save / promote / reject
12. Define idempotency semantics for WebUI-triggered operations:
   - dataset registration
   - run creation
   - export launch
   - review action submission
13. Define assignment and ownership fields:
   - current owner role / id
   - assignment timestamp
   - next required human role
   - blocking reason
14. Define manifest query and saved-filter contract:
   - deterministic filter serialization
   - sort key semantics
   - cursor or page token behavior
15. Define event payload contract for resumable UI progress streams:
   - `event_id`
   - `job_id`
   - `event_type`
   - `stage_name`
   - `progress_percent`
   - `message`
   - `timestamp`

## Guardrails

- do not hardcode one provider as permanent truth
- do not let providers write incompatible ad hoc sidecars
- do not use implicit state hidden outside the manifest
- do not leave promotion/export decisions without durable human-action provenance when they are human-driven
- do not make CLI the only way for human operators to ingest, resume, promote, or export datasets
- do not let multiple reviewers silently overwrite each other on the same manifest record
- do not make progress tracking depend on transient in-memory state only

## Handoff Contract

- worker 08 can plug providers into a stable schema
- worker 09 can score and promote records without guessing fields
- worker 10 can export promoted subsets without reverse engineering
- worker 12 can build conflict-safe review and job-monitoring UI without inventing orchestration semantics
