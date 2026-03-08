# Worker 08: Provider Integration and Model Stack

## Purpose

Define which OSS systems are used for each curation stage and how their outputs are normalized.

## Intent

The curation system must not depend on one model family. It needs a provider layer so stronger OSS can replace older choices without rewriting the whole pipeline.

## Primary Files

- provider adapters under `tmrvc-data`
- curation config files
- `plan/ai_curation_system.md`
- `docs/design/provider-registry.md`

## Required Outcomes

- normalized provider interfaces
- confidence semantics per provider
- fallback order per stage
- artifact translation into the manifest contract
- explicit transcript-refinement engine contract
- explicit `voice_state` pseudo-label estimator contract
- provider artifact/version/runtime pinning contract shared with Worker 11 and Worker 09

## Recommended Provider Stack

### Separation / Enhancement

Primary:

- `sam-audio` for promptable separation in mixed long-form audio

Fallback:

- speech enhancement / separation implementation from `SpeechBrain` or `ESPnet`

### Diarization

Primary:

- `pyannote/speaker-diarization-community-1`

Fallbacks:

- `VibeVoice-ASR` joint structure recovery when pinned and validated
- `Reverb diarization`
- older `pyannote.audio` baselines only as explicit fallback entries in the provider registry

Additional required stage:

- cross-file speaker embedding extraction and global clustering
- recommended tools:
  - `wespeaker`
  - `SpeechBrain` speaker embeddings

### ASR

Primary:

- `Qwen3-ASR-1.7B`

Fallbacks:

- `Qwen3` family throughput/downscaled variants only when separately pinned
- `Reverb ASR`
- `faster-whisper`

### Bootstrap Alignment

Primary:

- `Qwen3-ForcedAligner-0.6B`

Fallback:

- none for unsupported languages; export explicit absence and route the sample to fallback / review policy rather than synthesizing pseudo-truth

### Transcript Refinement

Primary strategy:

- multi-ASR fusion
- audio-aware transcript refinement
- normalization pass

Deliverables:

- refinement engine module
- per-record disagreement summary
- refined transcript confidence
- correction provenance

### Prosody / Event Extraction

Use local extractors and classifiers for:

- pause spans
- breath events
- speech rate
- pitch / energy statistics
- style embeddings
- 8-D `voice_state` estimation
- per-dimension confidence calibration

## Concrete Tasks

1. Define provider interface fields:
   - output payload
   - confidence
   - warnings
   - provenance
   - `provider_id`
   - `artifact_id`
   - `provider_revision`
   - `runtime_backend`
   - `calibration_version`
2. Normalize speaker identifiers across providers.
3. Define cross-file speaker clustering contract:
   - extract speaker embeddings from diarized segments
   - assign persistent global speaker cluster ids across files
   - record clustering confidence and provenance
4. Normalize timestamps and segment boundaries.
5. Normalize transcript confidence semantics.
   - raw confidences from different providers must not be treated as one shared numeric scale
   - every provider confidence used by Worker 09 must reference a Worker 11-issued calibration version
6. Define provider comparison metrics:
   - ASR uplift
   - diarization uplift
   - cross-file speaker-clustering purity
   - separation artifact rate
   - speaker / timbre preservation after separation
   - waveform artifact score for any separated waveform proposed as a teacher signal
7. Define transcript-refinement policy:
   - when to fuse multiple ASR hypotheses
   - when to accept one provider as dominant
   - how to compute disagreement and refinement confidence
   - how to preserve correction provenance
8. Define fallback policy:
   - provider missing
   - provider low-confidence
   - provider disagreement
   - provider unsupported for the target language
   - throughput fallback versus quality-mainline fallback must be distinguishable in provenance
9. Define `voice_state` estimator contract:
   - canonical output tensor shape: `[T_frames, 8]`
   - observed mask shape: `[T_frames, 8]`
   - confidence shape: `[T_frames, 8]` or `[T_frames, 1]`
   - estimator/version provenance
   - calibration semantics and missing-dimension behavior
10. Define provider comparison metrics for `voice_state` labels:
   - agreement across estimators where available
   - calibration quality
   - downstream controllability uplift correlation
11. Freeze the provider registry contract in `docs/design/provider-registry.md`:
   - one pinned entry per mainline provider
   - no `latest`, `main`, or implicit default provider-revision references
   - declare language support, license/gated-access status, runtime backend, and fallback policy per provider
12. Define mainline provider policy:
   - diarization mainline = `pyannote/speaker-diarization-community-1`
   - ASR mainline = `Qwen3-ASR-1.7B`
   - bootstrap-alignment mainline = `Qwen3-ForcedAligner-0.6B`
   - throughput fallback = `faster-whisper`
   - any deviation must be recorded as a provider-registry version bump or a per-run downgrade with provenance

## Guardrails

- do not mix provider-specific semantics directly into the core manifest
- do not adopt a provider solely on demo quality
- do not ignore disagreement between providers
- do not leave transcript refinement as an implied step without a concrete engine contract
- do not confuse file-local diarization labels with persistent dataset-global speaker ids
- do not emit dense zero `voice_state` labels when the estimator is uncertain; use masks and confidences
- do not mix unsupported-language outputs into mainline thresholds as if the provider had native support
- do not allow uncalibrated provider scores into promotion policy

## Handoff Contract

- worker 07 receives normalized stage outputs
- worker 09 receives confidence-calibrated fields
- worker 11 can compare providers on a common basis
- worker 10 receives canonical `voice_state` artifacts without provider-specific reinterpretation
