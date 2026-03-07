# Worker 12: Gradio Control Plane & HITL Interface

## Purpose

Build a unified WebUI using Gradio to serve as the control plane for UCLM v3. This moves the project from CLI-only operations to an interactive, human-in-the-loop (HITL) system for auditing, evaluation, and dramatic prototyping.

## Intent

The WebUI is the "cockpit" where humans make the final calls on acting quality and data promotion. It should wrap the capabilities of all other workers into a visual, accessible interface.

## Primary Files

- `tmrvc-gui/src/tmrvc_gui/app.py` (Gradio entrypoint)
- `tmrvc-gui/src/tmrvc_gui/components/` (Reusable Gradio blocks)
- `tmrvc-serve/src/tmrvc_serve/routes/admin.py` (UI-specific management APIs)
- `plan/worker_12_gradio_control_plane.md` (authoritative UI contract)

## Required Outcomes

- **Drama Workshop:** Interactive sliders for `pace`, `hold_bias`, `boundary_bias`, and `context` injection with real-time audio comparison.
- **Voice Cloning / Casting Gallery:** Upload reference audio, extract speaker prompts, and **save them to a persistent "Casting Gallery" for later use in TTS API or VST.**
- **Physical Control Panel:** First-class 8-D `voice_state` controls with reproducible presets and sweep logging.
- **Curation Auditor:** A visual browser for the curation manifest allowing one-click `Promote/Reject` decisions and text refinement.
- **Evaluation Arena:** A blind A/B testing interface for human preference scoring (MOS/Preference).
- **Dataset Manager:** A dashboard for asset ingest, provenance assignment, and visual health metrics (phoneme coverage, speaker distribution).
- **Human Workflow Layer:** Explicit role-based workflows from ingest to final sign-off.
- **Training Cockpit (Deferred):** Only after runtime/admin and export contracts are stable.

## Human Roles

Worker 12 must make the following human responsibilities explicit in the UI:

- `annotator`
  - fixes transcripts, language spans, and speaker assignments
- `auditor`
  - reviews AI-generated labels and dataset eligibility
- `director`
  - judges acting quality, controllability, and casting fitness
- `rater`
  - performs blinded subjective evaluation
- `admin`
  - manages models, datasets, legality metadata, and final export controls

The UI must not assume one super-user does every step.

## Concrete Tasks

### 1. Interactive Drama Workshop (Inference UI)
- **Voice Cloning / Actor Casting:** Add a file upload component for `reference_audio` and a **"Casting Gallery"** to save/load/export speaker profiles (Speaker Prompt + Embedding).
- **Session Persistence:** Implement "Save Project" to store all parameters, context history, and assigned actors for the workshop.
- Implement sliders for new v3 pacing controls.
- Implement explicit 8-D `voice_state` controls and preset recall.
- Implement dialogue context input as text-side history/context injection.
- Add "Compare" mode to hear differences between parameter sets or model versions side-by-side.
- Visualize pointer state (active text index) during playback if possible.

### 2. Curation Auditor (Data UI)
- Build a manifest browser that displays waveform, transcript, and AI confidence scores.
- **Data Access:** Read the `manifest.jsonl` directly from the file system or via a dedicated local data service (not `tmrvc-serve`).
- Implement "Quick Review" mode for rapid manual promotion/rejection.
- Implement a simple text editor to fix ASR errors, feeding back into the refinement stage.
- Visualize speaker clusters and diarization timelines.
- Support manual segment-boundary correction, speaker merge/split, and language-span correction.
- Show per-record approval history, current owner, and blocking reasons.

### 3. Dataset Management (Operational UI)
- **Asset Ingest & Scan:** Implement a UI for scanning raw audio directories and bulk assigning legality/provenance metadata.
- **Health Dashboard:** Visual charts for phoneme coverage, speaker stats, and duration (based on Worker 03 metrics).
- **Pipeline Orchestrator:** Controls to start/resume/stop curation stages (Ingest -> ASR -> Refinement) with progress bars.
- **Export Trigger:** One-click materialization of promoted subsets into TMRVC-cache for training.
- **Approval Queue:** Role-specific inboxes for `annotator`, `auditor`, and `admin`.
- **Split Manager:** UI to lock holdout/train assignments and inspect leakage risks.

### 4. Evaluation Arena (Subjective Quality UI)
- Build a blind A/B test module for pairwise comparison between `v2 legacy` and `v3 pointer`.
- Build a blind A/B test module for comparison against the fixed external baseline used by Worker 06.
- Implement MOS (Mean Opinion Score) collection for naturalness and dramatic appropriateness.
- Record the exact reference-audio length and baseline model/version used in each few-shot speaker-adaptation trial.
- Auto-save evaluation results to a JSON/SQL backend for Worker 06 analysis.
- Implement rater assignment, sample randomization, duplicate-sample quality checks, and basic rater-QC flags.
- Separate `director` qualitative notes from blinded `rater` scoring so subjective evaluation is auditable.

### 5. System Admin & Telemetry
- Expose model loading/unloading controls.
- Display server health, latency metrics, and VRAM usage.
- Show the active runtime contract/version so UI inputs cannot drift from Worker 01 / Worker 04 semantics.
- One-click "Export for Production" for models that pass human audit.
- Expose approval-policy settings:
  - who may override promotion decisions
  - whether double approval is required for `tts_mainline`
  - which actions are audit-critical


## Screen Map

The initial WebUI must be split into the following top-level screens:

1. `Upload`
   - browser upload of raw audio assets
   - server-side dataset registration
   - legality / provenance assignment
   - upload-job and ingest-job status
2. `Datasets`
   - dataset list
   - health metrics
   - split overview
   - curation-run launch / resume / stop controls
3. `Curation Queue`
   - manifest browser
   - review queue
   - transcript / language / speaker correction
   - promotion / rejection actions with audit notes
4. `Drama Workshop`
   - TTS generation
   - take generation and comparison
   - speaker casting gallery
   - control sweeps and saveable sessions
5. `Evaluation Arena`
   - blind A/B session launch
   - rater assignment
   - rating form
   - QC monitoring and result export
6. `Admin`
   - model load / unload
   - runtime contract inspection
   - policy settings
   - telemetry and audit viewer


## Role Access Matrix

Minimum page-level access policy:

| Role | Upload | Datasets | Curation Queue | Drama Workshop | Evaluation Arena | Admin |
|---|---|---|---|---|---|---|
| `annotator` | view | view | edit assigned records | no | no | no |
| `auditor` | view | view | review / promote / reject by policy | view | no | no |
| `director` | no | view | view | full | view own review sessions | no |
| `rater` | no | no | no | no | assigned blind sessions only | no |
| `admin` | full | full | full | full | full | full |

Worker 12 must also define record-level permission checks so page access alone is not the only gate.


## WebUI API Contract

Worker 12 must not guess backend routes. The following API groups must exist and be documented.

### Dataset / Upload APIs

- `POST /ui/datasets/upload`
  - multipart upload of raw audio or archive
- `POST /ui/datasets/register`
  - register existing server-side path
- `GET /ui/datasets`
  - list datasets and ingest status
- `GET /ui/datasets/{dataset_id}`
  - dataset metadata and health summary

### Curation Orchestration APIs

- `POST /ui/curation/runs`
  - create or start curation run
- `POST /ui/curation/runs/{run_id}/resume`
- `POST /ui/curation/runs/{run_id}/stop`
- `GET /ui/curation/runs/{run_id}`
  - run status, stage progress, failures
- `GET /ui/curation/records`
  - filtered manifest query
- `POST /ui/curation/records/{record_id}/action`
  - review / promote / reject / edit with audit note

### Workshop / Generation APIs

- `POST /ui/workshop/generate`
  - create one or many takes
- `POST /ui/workshop/takes/{take_id}/pin`
- `POST /ui/workshop/takes/{take_id}/export`
- `POST /ui/workshop/sessions`
  - save session state
- `GET /ui/workshop/sessions/{session_id}`
- `POST /ui/workshop/casting_gallery`
  - save speaker profile

### Evaluation APIs

- `POST /ui/eval/sessions`
  - create blinded evaluation session
- `GET /ui/eval/assignments/{assignment_id}`
- `POST /ui/eval/assignments/{assignment_id}/submit`
- `GET /ui/eval/sessions/{session_id}/results`

### Admin / Policy APIs

- `GET /admin/health`
- `GET /admin/telemetry`
- `POST /admin/load_model`
- `GET /admin/models`
- `GET /admin/runtime_contract`
- `GET /admin/audit`
- `POST /admin/policy`


## Frontend Module Layout

The Gradio app should be split into explicit modules so page logic does not collapse into one monolith:

- `tmrvc_gui/app.py`
  - bootstraps auth/session wiring, shared API client, and top-level navigation
- `tmrvc_gui/pages/upload.py`
  - upload form, legality metadata editor, upload-job progress
- `tmrvc_gui/pages/datasets.py`
  - dataset table, health dashboards, split manager, curation run controls
- `tmrvc_gui/pages/curation_queue.py`
  - review queue, manifest editor, waveform/transcript inspector
- `tmrvc_gui/pages/drama_workshop.py`
  - generation form, take board, compare player, casting gallery
- `tmrvc_gui/pages/evaluation_arena.py`
  - blind assignment player, scoring form, rater QC, export actions
- `tmrvc_gui/pages/admin.py`
  - model admin, runtime contract viewer, telemetry, audit browser
- `tmrvc_gui/components/`
  - reusable `audio_player`, `waveform_view`, `record_editor`, `take_card`, `audit_timeline`, `status_badge`
- `tmrvc_gui/state/`
  - lightweight session cache, optimistic-lock tokens, saved filter helpers
- `tmrvc_gui/api_client.py`
  - typed wrapper for all `/ui/*` and `/admin/*` routes

Worker 12 must define page modules early so future UI work does not re-encode business rules in ad hoc callbacks.


## Screen Interaction Contracts

Each top-level screen must publish its minimal operator flow, primary objects, and blocking conditions.

### `Upload`

- primary objects:
  - `upload_job`
  - `dataset`
- operator flow:
  1. choose files or archive
  2. assign legality / provenance / owner metadata
  3. submit upload
  4. monitor checksum, unpack, and registration progress
  5. confirm resulting dataset record
- blocking conditions:
  - unknown legality state
  - duplicate dataset fingerprint without admin override
  - failed checksum / archive validation

### `Datasets`

- primary objects:
  - `dataset`
  - `curation_run`
  - `split_assignment`
- operator flow:
  1. inspect health summary
  2. fix missing metadata or split-gate issues
  3. launch / resume / stop curation
  4. inspect run stage progress and failures
  5. export approved subsets
- blocking conditions:
  - unresolved legality issue
  - holdout leakage warning not explicitly acknowledged
  - curation run already owned by another active operator when manual intervention is required

### `Curation Queue`

- primary objects:
  - `manifest_record`
  - `review_action`
  - `audit_event`
- operator flow:
  1. load assigned queue or saved filter
  2. inspect audio, transcript, diarization, context
  3. edit transcript / language / speaker fields
  4. approve, reject, or send back for refinement with note
  5. release lock and move to next record
- blocking conditions:
  - stale edit token
  - missing rationale for audit-critical action
  - conflicting active editor unless admin override is recorded

### `Drama Workshop`

- primary objects:
  - `workshop_session`
  - `speaker_profile`
  - `take`
- operator flow:
  1. select text, actor, and context
  2. choose control preset or manual 8-D values
  3. generate one or many takes
  4. compare, rank, pin, annotate
  5. export winning take or save session for later
- blocking conditions:
  - runtime contract mismatch with saved session version
  - requested model unavailable
  - generation quota or GPU admission denied

### `Evaluation Arena`

- primary objects:
  - `eval_session`
  - `eval_assignment`
  - `rating_submission`
- operator flow:
  1. admin creates blinded session
  2. rater receives assignment
  3. rater listens, scores, and submits
  4. system performs QC checks
  5. admin exports locked result bundle
- blocking conditions:
  - assignment expired or already submitted
  - duplicate QC item unanswered
  - baseline registry mismatch with Worker 06 pinned artifact set

### `Admin`

- primary objects:
  - `runtime_contract`
  - `model_slot`
  - `policy_snapshot`
  - `audit_query`
- operator flow:
  1. inspect health and telemetry
  2. load / unload models
  3. view runtime contract and policy version
  4. inspect audit trail and stuck jobs
  5. apply policy changes with rationale
- blocking conditions:
  - dirty runtime state incompatible with hot-swap
  - policy change missing rationale
  - model artifact missing validation signature


## State Model

The WebUI must persist at least the following entities:

- `dataset`
  - source, legality, upload status, health summary
- `upload_job`
  - file count, bytes, progress, failure reason
- `curation_run`
  - stage, progress, retry count, operator
- `manifest_filter_state`
  - saved filters for reviewers
- `speaker_profile`
  - prompt asset, speaker embedding, metadata, owner
- `workshop_session`
  - text, controls, context, selected actor, active model
- `take`
  - waveform artifact, seed, cfg, pacing controls, ranking, export status
- `eval_session`
  - protocol version, baseline id, prompt set, assignment policy
- `eval_assignment`
  - rater, sample order, duplicate items, completion state
- `audit_event`
  - actor, action, object, before, after, rationale


## Long-Running Job Contract

The WebUI must treat upload, curation, export, and evaluation materialization as first-class jobs rather than one request/response callbacks.

- canonical job states:
  - `queued`
  - `running`
  - `blocked_human`
  - `failed_retryable`
  - `failed_terminal`
  - `completed`
  - `canceled`
- every job payload must include:
  - `job_id`
  - `job_type`
  - `state`
  - `progress_percent`
  - `stage_name`
  - `owner_role`
  - `owner_id`
  - `started_at`
  - `updated_at`
  - `retry_count`
  - `retryable`
  - `failure_code`
  - `failure_message`
  - `next_human_action`
- Worker 12 must support:
  - poll mode via `GET`
  - event mode via SSE or WebSocket
  - page restore after browser refresh using persisted `job_id`

The UI must never lose visibility into a job because a browser tab was closed.


## Concurrency and Locking Policy

WebUI state must stay correct under multiple simultaneous operators.

- reviewable objects (`manifest_record`, `policy_snapshot`, `eval_assignment`) require optimistic locking
- edit forms must submit:
  - `object_version`
  - `edit_session_id`
  - `last_seen_at`
- server must return explicit lock outcomes:
  - `accepted`
  - `stale_version`
  - `locked_by_other`
  - `policy_forbidden`
- `manifest_record` editing policy:
  - soft lock on open
  - idle timeout releases soft lock
  - admin may force-take with audit note
- `eval_assignment` policy:
  - single active rater session
  - no reassignment after first scored item unless session is voided with reason
- `workshop_session` policy:
  - collaborative viewing is allowed
  - write operations create a new session revision rather than mutating another user's pinned run in place


## Notifications and Failure Recovery

The WebUI must expose actionable failure handling instead of raw tracebacks.

- user-visible notification classes:
  - `info`
  - `success`
  - `warning`
  - `action_required`
  - `error`
- every `failed_retryable` job must surface:
  - human-readable cause
  - retry button if policy allows
  - last successful stage
  - recommended owner role for resolution
- every `failed_terminal` job must surface:
  - immutable failure snapshot
  - linked audit event
  - escalation path to admin
- browser refresh or reconnect must restore:
  - current page filters
  - selected record or session
  - active audio compare selection
  - in-flight job cards


## Artifact Lifecycle

Worker 12 must distinguish temporary audition assets from durable release artifacts.

- `take` retention classes:
  - `ephemeral`
  - `pinned`
  - `exported`
- `ephemeral` takes may be GC'd after TTL
- `pinned` takes require owner and note
- `exported` takes require provenance snapshot:
  - model id
  - runtime contract version
  - seed
  - guidance mode
  - voice-state controls
  - speaker profile id or prompt hash
- dataset export bundles and evaluation result bundles must show:
  - artifact id
  - creation time
  - source dataset / eval session
  - retention policy
  - download status


## Audit Event Contract

Every UI-originated critical action must persist this minimum schema:

```json
{
  "event_id": "uuid",
  "actor_role": "auditor",
  "actor_id": "user_123",
  "action": "promote_record",
  "object_type": "manifest_record",
  "object_id": "record_456",
  "before_state": {...},
  "after_state": {...},
  "reason_note": "approved after transcript fix",
  "session_id": "ui_session_789",
  "timestamp": "2026-03-07T12:34:56Z"
}
```

This schema must be shared with Worker 07 / Worker 11 so audit reconstruction is lossless.


## Definition of Done

Worker 12 is not complete when screens merely render. It is complete only when:

1. a human can upload raw assets and register a dataset without shell access
2. a reviewer can edit and promote records with conflict-safe locking and durable audit events
3. a director can generate, compare, and export takes entirely from the browser
4. a rater can finish a blinded assignment without seeing system or baseline identity
5. an admin can reconstruct who changed what and why for every audit-critical object
6. browser refresh or reconnect does not lose job visibility or corrupt critical state


## Delivery Slices

Worker 12 should ship in the following slices:

### Slice A: Operations MVP

- `Upload`
- `Datasets`
- `Curation Queue`
- minimal `Admin`

Goal:
- dataset upload / registration
- curation run control
- review / promote / reject
- export trigger

### Slice B: Workshop MVP

- `Drama Workshop`
- `Casting Gallery`
- take generation / compare / pin

Goal:
- browser-only audition workflow for directors

### Slice C: Evaluation MVP

- `Evaluation Arena`
- rater assignment
- QC and result export

Goal:
- release-signoff blind evaluation without CLI

## Human Workflow Contract

The UI must cover the full human loop:

1. ingest and legality assignment
2. annotation correction and speaker/language cleanup
3. audit and promotion/rejection review
4. holdout / train split confirmation
5. model audition and dramatic tuning
6. blinded subjective evaluation
7. final export approval

Each step must record:

- actor role
- actor identity
- timestamp
- before/after state
- reason or note

## Guardrails

- **Do not** replicate heavy logic in the UI; use `tmrvc-serve` APIs for all heavy lifting.
- **Do not** block the UI thread with long-running inference; use Gradio's async/generator support.
- **Do not** bypass the legality/split gates defined in Worker 09.
- **Do not** invent audio-context injection if Worker 01 freezes text-side `dialogue_context` for the initial mainline.
- **Do not** let the UI become the first implementation of a control path that is not already stable in Worker 04.
- **Do not** treat human approvals as ephemeral UI state; all critical actions must be auditable.
- **Prefer** simplicity over complex JS/CSS; stick to idiomatic Gradio components where possible.

## Handoff Contract

- A non-technical director can evaluate model quality without a terminal.
- Human audit results are stored in a format Worker 11 can use for the final sign-off.
- The "Drama Workshop" becomes the primary tool for tuning the v3 inference path.

## Required Tests

- UI smoke test (Gradio launches and connects to `tmrvc-serve`).
- End-to-end "Generate -> Audition -> Rate" flow test.
- Concurrent user test for the Evaluation Arena (simple multi-user check).
- audit-trail persistence test for approval and rating actions
- role-gating smoke test for `annotator` / `auditor` / `director` / `admin`
- upload-job resume test after browser reconnect
- optimistic-lock conflict test for manifest record editing
- workshop-session revision test for concurrent pin / save actions
- artifact-retention test for `ephemeral` vs `pinned` vs `exported` takes
- failure-recovery test that retryable jobs surface actionable UI state
