# AI Curation System Plan

## Purpose

Build a multi-stage, iterative, selective AI curation system that turns large unlabeled wav-only corpora into trainable assets for TMRVC without relying on manual full-corpus annotation.

## Intent

The goal is not to "label everything once". The goal is to:

1. recover as much reliable structure as possible from raw audio
2. reject or demote low-trust audio automatically
3. iteratively improve pseudo-labels with stronger models and cross-checks
4. promote only high-quality subsets into the mainline training path
5. preserve enough provenance that every promoted sample can be traced back to the evidence that justified it

This system is a data-quality machine, not a convenience script.

## Non-Negotiable Principles

1. Raw unlabeled audio must not flow directly into mainline training without curation.
2. Every stage must write machine-readable artifacts with provenance.
3. Every automatic label must carry confidence and source metadata.
4. Low-confidence outputs must be downgraded or rejected, not silently accepted.
5. Separation/enhancement outputs are annotation aids first, waveform teachers second.
6. Long-form audio must preserve conversation structure whenever recoverable.
7. The system must be resumable, restartable, and stage-addressable.
8. Rights or source-legality status must be explicit at ingest time and must gate export.

## Core Claim

High-quality training from wav-only corpora is achievable if the system performs:

- structure recovery
- cross-model agreement checks
- iterative pseudo-label refinement
- quality-gated subset promotion

The system must therefore be designed as a curation loop, not a one-shot pipeline.

## Inputs

Supported source types:

1. short isolated utterances
2. long-form dialogue audio
3. mixed audio with BGM / SFX
4. single-speaker or multi-speaker raw corpora

Each source must also declare a provenance class:

- `owned`
- `licensed`
- `research-restricted`
- `unknown`

`unknown` sources must never auto-promote into mainline export buckets.

## Recommended OSS Stack

### Long-form structure recovery

Primary candidates:

- `VibeVoice-ASR`
- `Reverb`

Reason:

- they are designed for long-form transcription and speaker-aware structure recovery

### Diarization fallback

- `pyannote.audio`

Reason:

- strong OSS baseline for speaker diarization and overlap detection

### Separation / enhancement

Primary candidate:

- `sam-audio`

Fallbacks:

- `SpeechBrain`
- `ESPnet`

Reason:

- mixed drama or movie audio needs a separation-capable front-end

### ASR fallback / throughput path

- `faster-whisper`

Reason:

- practical fallback when larger long-form stacks are unavailable

## Outputs

The curation system must produce:

1. `manifest.jsonl`
2. `summary.json`
3. stage artifacts with provenance
4. promoted subset list
5. review subset list
6. rejected subset list
7. exported cache-ready artifacts for TMRVC training

## Stage Intent

### Stage-to-Worker Mapping

| Stage | Stage Name | Owning Worker |
|-------|-----------|---------------|
| 0 | Ingest | Worker 07 |
| 1 | Cleanup | Worker 07 |
| 2 | Separation / Enhancement | Worker 08 |
| 3 | Speaker Structure Recovery | Worker 08 |
| 4 | Transcript Recovery | Worker 08 |
| 5 | Transcript Refinement | Worker 08 |
| 6 | Prosody / Event Recovery | Worker 08 |
| 7 | Quality Scoring | Worker 09 |
| 8 | Promotion / Review / Rejection | Worker 09 |
| 9 | Export | Worker 10 |

Worker 11 validates all stages. Worker 12 provides the HITL interface for stages 7-9.

### Ingest

Intent:

- identify assets
- prevent duplicate work
- establish canonical record IDs

### Cleanup

Intent:

- remove obviously unusable audio before expensive inference

### Separation / enhancement

Intent:

- improve annotation quality on mixed audio
- not to auto-promote separated waveforms as trusted clean teachers

### Speaker structure recovery

Intent:

- recover turn-taking information needed for dialogue-sensitive modeling

### Transcript recovery

Intent:

- create text-side supervision for pointer TTS

### Transcript refinement

Intent:

- convert one-pass pseudo-labels into stable pseudo-supervision

### Prosody / event recovery

Intent:

- recover acting-relevant signals beyond transcript text

### Quality scoring

Intent:

- decide how each sample can be used without contaminating mainline training

## Stage Graph

### Stage 0: Ingest

Recover file-level facts:

- path
- duration
- sample rate
- loudness
- channel count
- hash
- source legality / provenance class
- source collection id

### Stage 1: Cleanup

Recover speech-focused segments:

- VAD
- clipping detection
- corruption detection
- optional normalization

### Stage 2: Separation / Enhancement

For mixed long-form audio only.

Potential providers:

- `sam-audio`
- enhancement / separation fallback

Outputs:

- cleaned speech candidate
- separation confidence
- artifact notes

### Stage 3: Speaker Structure Recovery

Recover:

- speaker turns
- overlap flags
- speaker clusters
- cluster purity estimates
- conversation id candidates
- turn adjacency
- turn index within conversation

Potential providers:

- `VibeVoice-ASR` joint output
- `Reverb diarization`
- `pyannote.audio`

### Stage 4: Transcript Recovery

Recover:

- transcript
- token or word timestamps
- language
- ASR confidence

Potential providers:

- `VibeVoice-ASR`
- `Reverb ASR`
- `faster-whisper`

### Stage 5: Transcript Refinement

Improve transcripts through agreement and correction.

Methods:

- multi-ASR fusion
- audio-aware correction
- normalization pass
- language consistency checks

Outputs:

- refined transcript
- refinement confidence
- disagreement metrics

### Stage 6: Prosody / Event Recovery

Recover:

- pause spans
- breath events
- laughter / sigh / non-verbal cues
- speech rate
- pitch / energy statistics
- style embedding

### Stage 7: Quality Scoring

Produce per-sample scores from:

- ASR confidence
- ASR cross-model agreement
- diarization confidence
- overlap rate
- separation damage estimate
- language consistency
- duration sanity
- style / event extraction success

### Stage 8: Promotion / Review / Rejection

Each sample must be assigned one of:

- `promoted`
- `review`
- `rejected`

Promotion must require passing both:

1. hard safety constraints
2. quality-score threshold

### Stage 9: Export

Promoted samples become cache-ready assets:

- text
- language
- text units
- speaker metadata
- style/event metadata
- provenance metadata

## Parallel Execution Model

The system must support parallel work across:

1. file-level workers for ingest and cleanup
2. provider-level workers for heavy model inference
3. scoring workers for post-processing
4. export workers for final subset materialization

The orchestration layer must make provider stages independently restartable.

## Canonical Record Lifecycle

Each sample moves through:

1. `ingested`
2. `annotating`
3. `scored`
4. `promoted` or `review` or `rejected`
5. optionally `reprocessed`
6. optionally `exported`

No sample may jump directly from ingest to train-ready without passing the scored state.

## Iterative Loop

This system must support at least three loop modes.

### Loop A: Re-run weak stages on review items

Examples:

- stronger ASR
- different diarizer
- separation enabled only for hard items

### Loop B: Consensus refinement

Use multiple providers, merge outputs, re-score.

### Loop C: Self-training

Train internal quality or style predictors on promoted subsets, then reapply to review subsets.

## Mandatory Provenance Fields

Every record must track:

- source file
- segment bounds
- stage outputs and confidences
- provider names and versions
- curation pass index
- promotion/rejection reasons
- source legality / provenance status

## Minimum Manifest Fields

At minimum, every record must store:

- `record_id`
- `source_path`
- `audio_hash`
- `segment_start_sec`
- `segment_end_sec`
- `duration_sec`
- `language`
- `transcript`
- `transcript_confidence`
- `speaker_cluster`
- `diarization_confidence`
- `conversation_id`
- `turn_index`
- `prev_record_id`
- `next_record_id`
- `context_window_ids`
- `quality_score`
- `status`
- `promotion_bucket`
- `rejection_reasons`
- `review_reasons`
- `providers`
- `pass_index`
- `source_legality`

## Quality Policy

### Hard Reject

Reject immediately if:

- corrupted audio
- no usable speech after cleanup
- extreme overlap
- language mismatch
- transcript empty after refinement
- separation damage above threshold

### Review

Route to review if:

- provider disagreement is high
- transcript confidence is marginal
- cluster purity is unclear
- style/event extraction partially failed

### Promote

Promote only when:

- transcript is stable
- speaker structure is trustworthy enough
- prosody metadata exists or gracefully degrades
- quality score exceeds threshold
- source legality permits the target export bucket

## Default Promotion Buckets

### `tts_mainline`

Requires:

- reliable transcript
- reliable language
- usable text units
- acceptable speaker trust
- context graph preserved when the sample is dialogue-derived

### `vc_prior`

Requires:

- usable speech quality
- transcript can be weaker than `tts_mainline`

### `expressive_prior`

Requires:

- useful prosody / event signal
- may tolerate weaker transcript quality if not used as primary text supervision

### `holdout_eval`

Requires:

- high confidence
- diversity
- strict train/eval separation

## Baseline Threshold Table

These values are initial defaults and must be refined by worker 11, but downstream workers must code against them now.

### `tts_mainline`

- transcript confidence: `>= 0.90`
- cross-ASR agreement: `>= 0.85`
- diarization confidence: `>= 0.80` when multi-speaker
- overlap ratio: `< 0.10`
- quality score: `>= 0.85`
- legality: `owned` or `licensed`

### `vc_prior`

- transcript confidence: `>= 0.60`
- speech quality acceptable
- quality score: `>= 0.70`
- legality: `owned`, `licensed`, or explicitly approved research bucket

### `expressive_prior`

- transcript confidence: `>= 0.50`
- prosody/event extraction success required
- quality score: `>= 0.75`
- legality: `owned`, `licensed`, or explicitly approved research bucket

### `holdout_eval`

- transcript confidence: `>= 0.90`
- quality score: `>= 0.90`
- legality: same as target training policy
- explicit no-leak split assignment required

## Mainline Export Contract

Promoted subsets must export enough to support:

- pointer-based TTS
- self-supervised VC
- expressive style learning

Minimum export fields:

- transcript
- language
- text units
- speaker metadata
- conversation metadata
- quality score
- provenance

Optional export fields:

- pause events
- breath events
- style embedding
- same-text cluster id

## Failure Modes To Design For

1. ASR says something fluent but wrong
2. diarization splits one speaker into many fragments
3. diarization merges two speakers
4. separation improves ASR but damages timbre
5. long-form audio contains subtitle drift or off-screen speech
6. language detection flips on short interjections
7. low-quality repeated clips swamp the promoted subset

Each of these must map to explicit scoring, review, or rejection logic.

## What This System Must Not Do

1. pretend noisy pseudo-labels are ground truth
2. use separation output as unquestioned clean speech
3. collapse all data into one undifferentiated "good enough" bucket
4. hide provider failures
5. conflate "usable for VC prior" with "usable for TTS mainline"
6. auto-promote sources with unknown rights status

## Success Criteria

1. raw wav-only data can be transformed into promoted / review / rejected subsets automatically
2. promoted subsets are traceable and reproducible
3. promoted subsets measurably improve TTS/VC over naive raw-data ingestion
4. review subsets are small enough to inspect if needed
5. the system can absorb stronger OSS providers without redesign

## Definition of Completion

This plan is complete only when implementers can answer, without guessing:

1. which stage owns which decision
2. which providers are primary and fallback
3. how confidence enters scoring
4. how review differs from reject
5. how curated subsets map into TMRVC training paths
