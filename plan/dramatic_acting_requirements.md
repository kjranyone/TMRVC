# Drama-Grade Acting Remaining Requirements

This file is not a full design manifesto.
It records only the remaining blockers for a credible drama-grade `v4` claim.

## Still Required

### 1. Raw-audio bootstrap that preserves acting signal

Drama-grade claims are blocked until the system can start from unlabeled raw audio corpora and still produce:

- stable pseudo speaker identities
- usable transcripts
- acting-relevant semantic annotations
- usable physical-control supervision

Good outputs from a manually curated dataset are not enough for the `v4` thesis.

### 2. Hybrid acting control

Drama-grade acting requires both:

- editable physical control
- non-physical acting texture capture

Prompt-only control is insufficient.
Physical-only control is also insufficient.

### 3. Deterministic acting artifacts

Drama-grade claim is incomplete until the system can:

- compile creator intent into canonical controls
- persist acting as a versioned `TrajectoryRecord`
- replay, patch, and transfer that acting deterministically

Good first-take samples alone are insufficient.

### 4. Trajectory-first control surface

The main UI artifact must be the compiled or replayed trajectory, not only a prompt textbox or a slider snapshot.

### 5. Replay / locality / transfer evidence

The claim is blocked until Worker 06 reports:

- replay fidelity
- edit locality
- transfer quality, if transfer is part of the public product capability

### 6. Runtime-class claim discipline

- Python serve may carry the full drama-grade claim once validated
- Rust / ONNX / VST paths may claim parity and latency only until they pass the same `v4` artifact-aware validation

## Allowed Claims Before Closure

- prompt-conditioned expressive TTS experiments
- physical control experiments
- context-sensitive acting experiments

## Blocked Claims Before Closure

- programmable expressive speech
- deterministic acting replay/edit
- cross-speaker acting transfer as a validated product capability
- broad "beats Fish Audio S2" language
