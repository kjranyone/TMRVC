# Dramatic Acting Requirements

## Purpose

This document defines what must be true before TMRVC can credibly claim drama-grade conversational TTS rather than high-quality neutral reading.

## 1. What "dramatic acting" means here

A compliant system must be able to change delivery based on:

- preceding dialogue context
- scene intent
- speaker relationship and tension
- interruption pressure
- phrase-final release or restraint
- hesitation, breath, and pause timing

The same sentence under different context must produce meaningfully different pacing and phrasing without collapsing into noise or random instability.

## 2. Architecture requirements

These are mandatory.

1. Pointer-based online text progression
2. Explicit dialogue-context conditioning path
3. **High-fidelity Zero-Shot / Few-Shot speaker adaptation (in-context prompting)**
4. Local prosody latent or equivalent phrase-planning channel
5. Classifier-free guidance or an equally explicit sampling-control mechanism
6. Explicit waveform-decoder / vocoder quality plan
7. Multilingual and code-switch conditioning contract if SOTA claims extend beyond monolingual read speech
8. Runtime pacing controls:
   - `pace`
   - `hold_bias`
   - `boundary_bias`
9. Event/control path capable of representing breath, pause, release, and non-verbal timing

Pointer alone is insufficient. A purely pointer-based model without context/prosody planning and high-fidelity cloning capacity is non-compliant.

### Prosody granularity escalation policy

Initial v3 uses utterance-global prosody `[B, d_prosody]`. This is sufficient for utterance-level style (e.g., "say this angrily") but insufficient for fine-grained dramatic delivery (mid-sentence emotion shifts, selective emphasis, trailing-off). SOTA systems (DiFlow-TTS, Flamed-TTS, Vevo) operate at token-level or segment-level granularity.

If drama acting quality gates (§ 6.1 context-sensitivity score, § 6.2 same-line different-context appropriateness) are not met with the utterance-global form at Stage B completion, the upgrade to time-local `[B, T_plan, d_prosody]` must be scheduled for Stage C. This is not optional if the drama-grade claim is to be maintained.

### Waveform quality escalation policy

If end-to-end TTS quality comparisons against the external baseline show deficits attributable to fine-grained acoustic detail (rather than prosody or alignment), the 2-stage acoustic refinement path (flow matching / DiT over coarse AR tokens) defined in Worker 01 § Acoustic Refinement Roadmap must be activated. Waveform decoder quality is a drama requirement, not only a fidelity requirement.

## 3. Data requirements

These are mandatory to make the claim believable.

1. Dialogue-like data, not only isolated read speech
2. Dataset-level language purity
3. Enough expressive variation to observe:
   - hesitation
   - urgency
   - restraint
   - overlap pressure
   - line-final decay or punch
4. If possible, same-text or near-paraphrase multi-take coverage under different contexts
5. Metadata or derived labels for:
   - scene
   - speaker turn
   - emotion/style
   - pause/breath/non-verbal events
6. Conversation linkage so preceding and following turns can be reconstructed during training

Without this, the model may still produce clean speech but cannot be said to have learned drama-grade acting.

## 3.1 If only raw short wav files exist

That corpus is still valuable, but it must go through high-quality local pseudo-annotation before it is trusted.

Minimum pipeline:

1. VAD cleanup
2. speaker clustering or diarization
3. optional promptable source separation for mixed long-form audio
4. high-quality ASR
5. text normalization post-edit
6. G2P / text-unit generation
7. pause / breath / non-verbal pseudo-label extraction
8. style embedding extraction
9. confidence-based filtering

The full raw corpus should not automatically become the final training set. A filtered high-quality subset should become the mainline training subset.
If source separation is used, its primary purpose is annotation uplift. It must not automatically become the trusted waveform teacher without audit.

## 4. Training requirements

1. TTS training must accept context-conditioned batches
2. Alignment training must remain internal and causal-safe
3. Prosody/context channels must have explicit loss support or diagnostics
4. Collapse checks must exist:
   - same text, different context should not converge to nearly identical pacing
   - control changes should measurably affect output
5. Legacy duration supervision must not dominate the expressive path
6. Mainline training/eval must include a few-shot speaker adaptation path using short reference audio
7. CFG-compatible condition dropout and guidance validation must be explicit if guidance is part of the serving claim

## 5. Runtime requirements

1. TTS inference must expose expressive controls through API and engine
2. Pointer state and pacing telemetry must be inspectable for debugging
3. Runtime must remain causal and low-latency
4. The engine must not depend on pre-expanded duration plans
5. Cache synchronization under moving pointers must be explicitly testable
6. Waveform-decoder quality must be measurable, not assumed from token quality

## 6. Proof obligations

No quality claim is accepted unless it is backed by reproducible evidence.

### 6.1 Automatic checks

- context-sensitivity score
- control-responsiveness score
- pause / boundary realism metrics
- latency and causality checks
- pseudo-annotation confidence and audit metrics
- few-shot speaker adaptation similarity / intelligibility metrics
- guidance stability metrics
- waveform artifact metrics
- multilingual / code-switch intelligibility metrics when applicable

### 6.2 Human evaluation

At minimum:

1. paired preference test against `v2 legacy`
2. **Blind A/B preference test against the fixed pinned external baseline artifact**
3. same-line different-context appropriateness test
4. actor-control responsiveness test for:
   - `pace`
   - `hold_bias`
   - `boundary_bias`
5. **Zero-shot similarity and disentanglement audit:**
   - measure speaker similarity against reference audio
   - ensure neutral reference prosody does not flatten dramatic output
6. annotation audit of sampled pseudo-labeled utterances

### 6.3 Failure cases that block the claim

- all contexts sound like the same reading
- controls change duration but not acting quality
- expressive settings produce noise or instability
- context changes are overwhelmed by speaker/style defaults
- **Few-shot similarity is significantly lower than the fixed pinned external baseline**

## 7. Acceptance bar

TMRVC may claim drama-grade conversational TTS only if all of the following hold:

1. mainline v3 train/serve path works without MFA artifacts
2. held-out dialogue evaluation shows context-sensitive phrasing changes
3. runtime controls produce consistent directional changes
4. human raters prefer dramatic outputs over neutral baseline on a meaningful subset
5. latency remains within the project budget
6. few-shot speaker adaptation is competitive with the fixed external baseline on the accepted test protocol

## 8. Worker mapping

- architecture: `worker_01_architecture.md`
- training: `worker_02_training.md`
- dataset and supervision: `worker_03_dataset_alignment.md`
- serving/runtime: `worker_04_serving.md`
- validation/proof: `worker_06_validation.md`
