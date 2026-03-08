# Worker 12: Gradio Control Plane & HITL Interface

## Purpose

Build a Gradio/WebUI control plane for UCLM v3. This is the mainline HITL surface because the project needs browser-based access, multi-user evaluation, auditable human actions, and role-separated workflows that are awkward to satisfy with a desktop-only Qt application.

## Intent

The WebUI is the "cockpit" where humans make the final calls on acting quality and data promotion. It wraps the capabilities of all other workers into a browser-accessible interface suitable for annotators, auditors, directors, raters, and admins.

## Primary Files

- `tmrvc-gui/src/tmrvc_gui/gradio_app.py` (all tabs: Drama Workshop, Realtime VC, Curation Auditor, Dataset Manager, Evaluation Arena, Speaker Enrollment, Batch Script, ONNX Export, Server Control, System Admin; **Training Monitor is post-v3.0 scope**)
- `tmrvc-gui/src/tmrvc_gui/gradio_state.py`
- `tmrvc-serve/src/tmrvc_serve/routes/admin.py` (UI-specific management APIs)
- `docs/design/gui-design.md`
- `docs/design/auth-spec.md` (Infrastructure Foundation)

## Required Outcomes

- **Control Plane Infrastructure Integration:** Integrate with backend-owned authentication, RBAC, audit logging, and optimistic locking as defined in `docs/design/auth-spec.md`. Worker 12 consumes these capabilities; it does not become a second implementation owner for them.
- **Drama Workshop:** Interactive sliders for `pace`, `hold_bias`, `boundary_bias`, and `context` injection with real-time audio comparison.
- **Voice Cloning / Casting Gallery:** Upload reference audio, extract speaker prompts, and **save them to a persistent "Casting Gallery" using the canonical `SpeakerProfile` contract defined in `docs/design/speaker-profile-spec.md`.**
- **Physical Control Panel:** First-class 8-D `voice_state` controls with reproducible presets and sweep logging.
- **Curation Auditor:** A visual browser for the curation manifest allowing one-click `Promote/Reject` decisions and text refinement.
- **Evaluation Arena:** A blind A/B testing interface for human preference scoring (MOS/Preference).
- **Dataset Manager:** A dashboard for asset ingest, provenance assignment, and visual health metrics (phoneme coverage, speaker distribution).
- **Human Workflow Layer:** Explicit role-based workflows from ingest to final sign-off.
- **No-CLI Human Operation:** A human can upload dataset material, curate, export, audition, and rate entirely from the browser.
- **Multi-device Support:** Gradio app must handle multiple concurrent sessions with independent states.
- **Audit Logs:** All state-changing actions (Casting Gallery saves, Curation promotions) must be logged with actor ID and rationale.
- **Conflict Handling:** Use canonical `metadata_version` tags to prevent race conditions during collaborative curation or casting.
- **Training Cockpit (Post-v3.0):** Only after runtime/admin and export contracts are stable; it is not part of the initial v3.0 proof obligations.

## v3.0 Scope Prioritization

The full feature set above is the target end state. To prevent Worker 12 scope from blocking the v3.0 core proof obligations, the following priority tiers apply:

### Tier 1: v3.0 Minimum (must ship for core proof)

- Drama Workshop: basic inference with pacing controls, `voice_state` sliders, and `speaker_profile_id` selection
- Evaluation Arena: blind A/B comparison and MOS collection with rater assignment and result persistence
- Casting Gallery: upload reference audio, encode `SpeakerProfile`, save/load profiles
- System Admin: model load/unload and basic health display

### Tier 2: v3.0 Extended (required for full human workflow claim)

- Curation Auditor with optimistic locking and manifest browsing
- Dataset Manager: upload, register, curation launch/monitor/resume
- Take Management: multi-take generation, audition, pin/rank
- Export Trigger and artifact download
- Full RBAC enforcement and audit trail persistence
- Approval queues and role-specific inboxes

### Tier 3: Post-v3.0

- Training Cockpit (already declared post-v3.0)
- Advanced rater QC analytics
- Split Manager leak detection dashboard

Tier 1 items must not be blocked by Tier 2 development. If Tier 2 items slip, Tier 1 features must be independently deployable and sufficient for Worker 06 sign-off.

## Human Roles

Worker 12 must make the following human responsibilities explicit in the UI (as defined in `auth-spec.md`):

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
- **Voice Cloning / Actor Casting:** Add a file upload component for `reference_audio` and a **"Casting Gallery"** to save/load/export **`SpeakerProfile`** objects.
  - **Latency Mitigation:** The UI must encourage users to perform reference-audio encoding in the Casting Gallery *before* starting the Workshop session.
  - When a new `reference_audio` is uploaded during a session, show a "Encoding Speaker Profile..." progress indicator and handle the `wait_for_prompt` API behavior.
- `SpeakerProfile` is a backend-owned persistent object. The UI must consume the Worker 01 / Worker 04 schema and must not invent a divergent frontend-only shape.
- **Session Persistence:** Implement "Save Project" to store all parameters, context history, and assigned actors for the workshop, subject to RBAC and audit.
- Implement sliders for new v3 pacing controls.
- Implement explicit 8-D `voice_state` controls and preset recall.
- Implement dialogue context input as text-side history/context injection.
- Add "Compare" mode to hear differences between parameter sets or model versions side-by-side.
- **Take Management (Multi-Take Audition & Ranking System)**:
  - generate multi-take variations by seed / `cfg_scale` / pacing sweeps in parallel
  - persist take lineage, notes, and selected-best take (saving parameters like Seed, Control-curve, Pacing as reproducible metadata)
  - support quick audition, blind A/B compare, pin/rank, and export of chosen takes
- **Personal Voice Training (Few-Shot Fine-tuning):** Post-v3.0 scope only.
  - if retained in the plan, it must be explicitly gated behind the post-v3.0 Training Cockpit workstream
  - it must not become part of the initial v3.0 proof obligations or block mainline sign-off
  - raw uploaded audio is not directly trainable for pointer-TTS. Any retained Personal Voice Training flow must first run a lightweight curation/preparation sub-pipeline that materializes the canonical training artifacts required by Worker 02:
    - VAD / segmentation cleanup
    - fast ASR or user-provided transcript verification
    - text normalization + G2P
    - canonical `phoneme_ids` and optional `text_suprasegmentals`
    - bootstrap alignment / boundary refinement sufficient for the selected training recipe
  - the UI must present this as an explicit staged job (`prepare -> review -> train`), not as "upload wav -> train LoRA" magic
  - if transcript/G2P/bootstrap preparation fails or falls back, the job must block or downgrade visibly; silent best-effort training on raw wav is forbidden
- Visualize pointer state (active text index) during playback if possible.

### 2. Curation Auditor (Data UI)
- Build a manifest browser that displays waveform, transcript, and AI confidence scores.
- **Optimistic Locking:** Surface backend `metadata_version` conflicts clearly when submitting transcript fixes or promotion decisions to prevent lost updates.
- **Data Access:** Use the authoritative typed `/ui/*` API exposed by `tmrvc-serve`. Any local data service must sit behind the same typed contract and auth/concurrency rules. Direct filesystem reads are dev-only and are forbidden for the mainline multi-user path.
- Implement "Quick Review" mode for rapid manual promotion/rejection.
- Implement a simple text editor to fix ASR errors, feeding back into the refinement stage.
- Visualize speaker clusters and diarization timelines.
- Show provider provenance and trust state for each reviewable record:
  - active `provider_id`
  - `provider_revision`
  - `calibration_version`
  - `fallback_class` / downgrade reason when present
  - `score_components` breakdown so auditors can see why a sample landed in `review` or `reject`
- Support manual segment-boundary correction, speaker merge/split, and language-span correction.
- Show per-record approval history, current owner, and blocking reasons.

### 3. Dataset Management (Operational UI)
- **Asset Ingest & Upload:** Implement UI flows for:
  - uploading local dataset material from the browser
  - registering existing server-side directories
  - bulk assigning legality/provenance metadata
- **Health Dashboard:** Visual charts for phoneme coverage, speaker stats, and duration (based on Worker 03 metrics).
- **Provider Health View:** Surface provider calibration/downgrade state at dataset/run level:
  - how many records used fallback providers
  - which provider/calibration versions are active
  - how many records are blocked by unsupported-language or missing-calibration policy
- **Pipeline Orchestrator:** Controls to start/resume/stop curation stages (Ingest -> ASR -> Refinement) with progress bars.
- **Enrollment-to-Training Preparation:** If post-v3.0 Personal Voice Training is enabled, Dataset Manager / Training Cockpit must expose a lightweight preparation flow for user-uploaded adaptation audio:
  - upload short personal-voice clips
  - run the Worker 07-owned preparation job (`VAD -> ASR/transcript check -> G2P -> boundary/alignment refinement`)
  - surface low-confidence items for human correction before any training job is allowed to start
  - materialize the same canonical artifacts used by the mainline training pipeline rather than inventing a GUI-only fine-tune format
- **Export Trigger:** One-click materialization of promoted subsets into TMRVC-cache for training.
- **Artifact Download / Handoff:** Enable browser-side download or registration of exported evaluation/training bundles.
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

## Human Workflow Contract

The UI must cover the full human loop:

1. dataset upload or dataset registration and legality assignment
2. annotation correction and speaker/language cleanup
3. audit and promotion/rejection review (requiring **auditor** role)
4. holdout / train split confirmation (requiring **admin** role)
5. export / package creation
6. model audition and dramatic tuning
7. blinded subjective evaluation (requiring **rater** role)
8. final export approval (requiring **admin** role)

Each step must record:

- actor role
- actor identity
- timestamp
- **`metadata_version`** (for concurrency check)
- before/after state
- reason or note

## Why Gradio Mainline

- browser access is required for directors, auditors, and raters who should not need local Qt installs
- blind A/B evaluation and queue-based review are naturally multi-user web workflows
- audit trails and approval queues are easier to centralize in a web control plane
- dataset upload and ingest must be operable by humans without shell access
- PySide6/Qt has been fully deprecated and removed; the Gradio WebUI is the sole UI surface


## Guardrails

- **Do not** replicate heavy logic in the UI; use `tmrvc-serve` APIs or the curation data service for heavy lifting.
- **Do not** reimplement auth, audit, concurrency, or curation mutation semantics in frontend-only state.
- **Do not** block request handling with long-running inference; use Gradio queue / generator patterns and background job dispatch.
- **Do not** bypass the legality/split gates defined in Worker 09.
- **Do not** invent audio-context injection if Worker 01 freezes text-side `dialogue_context` for the initial mainline.
- **Do not** let the UI become the first implementation of a control path that is not already stable in Worker 04.
- **Do not** treat human approvals as ephemeral UI state; all critical actions must be auditable.
- **Do not** require humans to use CLI or desktop apps; all workflows are browser-based.
- **Do not** require humans to switch to CLI for dataset upload, curation execution, export, or evaluation setup.
- **Do not** advertise or implement "upload audio -> LoRA train" without the explicit preparation stages that produce canonical text/frontend/alignment artifacts. Personal voice training without this staging is non-functional by construction.

## Handoff Contract

- A non-technical director can evaluate model quality from a browser without local setup.
- A non-technical operator can ingest dataset material and launch curation from a browser without shell access.
- Human audit results are stored in a format Worker 11 can use for the final sign-off.
- The "Drama Workshop" becomes the primary tool for tuning the v3 inference path.

## Required Tests

- UI smoke test (Gradio app launches and connects to `tmrvc-serve`).
- dataset upload / registration flow test
- End-to-end "Generate -> Audition -> Rate" flow test.
- take-generation and take-selection persistence test
- concurrent-user test for the Evaluation Arena
- audit-trail persistence test for approval and rating actions
- role-gating smoke test for `annotator` / `auditor` / `director` / `admin`
