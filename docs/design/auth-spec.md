# Control Plane Infrastructure Specification (Auth, Audit, & Concurrency)

## Purpose

Define the foundational infrastructure for the TMRVC Gradio-based control plane. This ensures that the high-level human workflows described in `worker_12_gradio_control_plane.md` are supported by a secure, auditable, and multi-user capable backend.

## 1. Authentication & Session Management

### Identity Provider
- **Default:** Simple internal credential store (username/hashed password).
- **Session:** JWT (JSON Web Token) or encrypted session cookie stored in the browser.
- **Expiry:** Configurable (default 24 hours).
- **Multi-device:** Support multiple active sessions per user.

### Role-Based Access Control (RBAC)
Five mandatory roles as defined in Worker 12:
- `annotator`: Access to Curation Auditor for text/speaker correction.
- `auditor`: Access to Curation Auditor for Promote/Reject actions.
- `director`: Access to Drama Workshop and Casting Gallery.
- `rater`: Access to Evaluation Arena (blinded view).
- `admin`: Full access to system health, model loading, and legality settings.

## 2. Audit Logging

Every state-changing action in the control plane must be logged to a persistent audit store (e.g., `audit_log.jsonl` or a database).

### Log Entry Schema
- `timestamp`: ISO 8601 UTC.
- `actor_id`: Username or UUID.
- `role`: Role active at the time of action.
- `action_type`: e.g., `PROMOTE_RECORD`, `FIX_TRANSCRIPT`, `EXPORT_MODEL`, `SAVE_TAKE`.
- `target_type`: e.g., `CURATION_RECORD`, `SPEAKER_PROFILE`, `MODEL_CHECKPOINT`.
- `target_id`: ID of the object being modified.
- `before_state`: Optional snapshot of the state before modification (for undo/traceability).
- `after_state`: Optional snapshot of the state after modification.
- `rationale`: Human-provided text note explaining the action.

## 3. Concurrency Control (Optimistic Locking)

To prevent "Lost Update" anomalies where multiple users modify the same record (e.g., two annotators fixing the same transcript), the system uses **Optimistic Locking**.

### Versioning Mechanism
- Every mutable record (Curation Record, Speaker Profile) must have a `version_tag` (e.g., an incrementing integer or a UUID/Hash of the content).
- **Update Request:** The UI must send the `version_tag` it originally read.
- **Server Validation:** The backend compares the incoming `version_tag` with the current one in the store.
- **Conflict Handling:**
  - If tags match: Update the record and increment/refresh the `version_tag`.
  - If tags differ: Reject the update with a `409 Conflict` error and prompt the user to refresh and merge.

## 4. Manifest Integrity

The `manifest.jsonl` used for curation is the primary target for concurrency control.
- Each record entry includes a `metadata_version` field.
- The `tmrvc-data` curation service must enforce version checks on all `POST/PATCH` operations.

## 5. Security Guardrails

- **No Shared Passwords:** Each human operator must have a unique identity.
- **Blinding Integrity:** The Evaluation Arena must strip all `actor_id`, `model_id`, and `provenance` metadata from the `rater`'s view.
- **Export Gating:** Only an `admin` can trigger the final production-ready model export.
- **Audit Immutability:** Audit logs must be append-only and protected from unauthorized modification.
