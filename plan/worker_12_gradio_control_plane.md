# Worker 12: Gradio Control Plane & HITL Interface

## Purpose

Build a unified WebUI using Gradio to serve as the control plane for UCLM v3. This moves the project from CLI-only operations to an interactive, human-in-the-loop (HITL) system for auditing, evaluation, and dramatic prototyping.

## Intent

The WebUI is the "cockpit" where humans make the final calls on acting quality and data promotion. It should wrap the capabilities of all other workers into a visual, accessible interface.

## Primary Files

- `tmrvc-gui/src/tmrvc_gui/app.py` (Gradio entrypoint)
- `tmrvc-gui/src/tmrvc_gui/components/` (Reusable Gradio blocks)
- `tmrvc-serve/src/tmrvc_serve/routes/admin.py` (UI-specific management APIs)
- `docs/design/gui-design.md`

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
