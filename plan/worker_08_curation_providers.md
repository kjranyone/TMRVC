# Worker 08: Provider Integration and Model Stack

## Purpose

Define which OSS systems are used for each curation stage and how their outputs are normalized.

## Intent

The curation system must not depend on one model family. It needs a provider layer so stronger OSS can replace older choices without rewriting the whole pipeline.

## Primary Files

- provider adapters under `tmrvc-data`
- curation config files
- `plan/ai_curation_system.md`

## Required Outcomes

- normalized provider interfaces
- confidence semantics per provider
- fallback order per stage
- artifact translation into the manifest contract
- explicit transcript-refinement engine contract
- explicit `voice_state` pseudo-label estimator contract

## Recommended Provider Stack

### Separation / Enhancement

Primary:

- `sam-audio` for promptable separation in mixed long-form audio

Fallback:

- speech enhancement / separation implementation from `SpeechBrain` or `ESPnet`

### Diarization

Primary:

- `VibeVoice-ASR` if joint outputs are reliable enough

Fallbacks:

- `Reverb diarization`
- `pyannote.audio`

Additional required stage:

- cross-file speaker embedding extraction and global clustering
- recommended tools:
  - `wespeaker`
  - `SpeechBrain` speaker embeddings

### ASR

Primary:

- `VibeVoice-ASR` for long-form structure-rich audio

Fallbacks:

- `Reverb ASR`
- `faster-whisper`

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
2. Normalize speaker identifiers across providers.
3. Define cross-file speaker clustering contract:
   - extract speaker embeddings from diarized segments
   - assign persistent global speaker cluster ids across files
   - record clustering confidence and provenance
4. Normalize timestamps and segment boundaries.
5. Normalize transcript confidence semantics.
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

## Guardrails

- do not mix provider-specific semantics directly into the core manifest
- do not adopt a provider solely on demo quality
- do not ignore disagreement between providers
- do not leave transcript refinement as an implied step without a concrete engine contract
- do not confuse file-local diarization labels with persistent dataset-global speaker ids
- do not emit dense zero `voice_state` labels when the estimator is uncertain; use masks and confidences

## Handoff Contract

- worker 07 receives normalized stage outputs
- worker 09 receives confidence-calibrated fields
- worker 11 can compare providers on a common basis
- worker 10 receives canonical `voice_state` artifacts without provider-specific reinterpretation
