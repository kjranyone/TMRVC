# Fish Audio S2 Competitive Strategy

## Purpose

This note records the design response to the public March 2026 release of Fish Audio S2.

It is not a benchmark report.
It exists to answer one practical question:

- how TMRVC should structure `v4` so that it competes on a winnable axis rather than chasing Fish on Fish's preferred axis

## Situation As Of 2026-03-17

Fish Audio S2 publicly positions itself as a strong expressive TTS stack with:

- natural-language prompting and inline expressive cues
- few-shot / zero-shot speaker cloning
- strong product-facing expressivity
- public self-hosting and public technical materials
- high-capacity model positioning

Implication:

- TMRVC should assume that promptable expressive TTS is already crowded
- TMRVC should not define success as merely matching emotional speech from natural-language prompts

## Competitive Thesis

This strategy does not guarantee broad SOTA.
It is only a falsifiable bet on a narrower outcome:

- TMRVC can beat Fish Audio S2 on programmable expressive-speech axes if, and only if, those axes are frozen and measured head-to-head

TMRVC should not compete as:

- another prompt-only expressive TTS model

TMRVC should compete as:

- a programmable expressive speech engine
- a shared TTS + VC acting system
- a deterministic edit / replay / transfer system for speech performance
- a system that preserves editable physical controls while also supporting open-ended acting prompts

The core bet is that controllability, editability, transferability, and hard runtime contracts are more defensible than prompt-only expressiveness.

## Strategic Direction

### 1. Start from raw audio, not only curated speaker-separated corpora

Fish-scale competitiveness requires a larger data frontier.
`v4` must therefore assume that the starting point can be:

- unlabeled raw audio
- speaker-mixed corpora
- noisy or weakly annotated archives

The answer is not to train directly on raw files.
The answer is to build a strong bootstrap pipeline:

- VAD
- overlap rejection
- diarization / clustering
- pseudo speaker assignment
- Whisper transcription
- DSP / SSL physical extraction
- LLM semantic annotation

### 2. Put an Intent Compiler above the acoustic model

Natural-language acting instructions and reference audio should not remain the final control interface.

They should compile into explicit, serializable controls:

- physical control targets
- acting texture latent prior
- pacing controls
- dialogue or scene context

Why this is the winning move:

- Fish-style prompting is strong for usability
- TMRVC can absorb that usability without giving up explicit control contracts
- reproducibility and post-editing become first-class rather than accidental

### 3. Use a hybrid acting space, not prompt-only or physical-only control

The `v4` acting space should contain:

- explicit physical controls
- acting texture latent

Prompt-only systems are easy to use but hard to edit deterministically.
Physical-only systems are editable but can miss nuanced acting residue.

The hybrid is the differentiated path.

### 4. Treat acting as a trajectory, not a label

The mainline representation must be a time-varying trajectory synchronized to pointer progression.

Not acceptable as the primary abstraction:

- utterance-level emotion class only
- opaque style embedding only
- absolute wall-clock schedule only

Preferred form:

- pointer-synchronous physical and acting trajectories over text-unit neighborhoods and `10 ms` runtime steps

### 5. Keep TTS and VC in one acting space

This remains one of the strongest differentiators.

The same acting control space should support:

- expressive TTS from text
- expressive VC from source speech
- acting transfer from one speaker to another
- trajectory extraction from speech and replay into TTS

If TMRVC degrades into a TTS-only competition, it loses one of its clearest strategic edges.

### 6. Separate creative UX from deterministic runtime

The system should expose two layers:

1. a creator-facing layer that accepts natural-language instructions, tags, references, and scene context
2. a deterministic layer that consumes compiled control trajectories directly

The creator-facing layer may be exploratory.
The deterministic layer must be stable and replayable.

### 7. Win on editability and transfer, not only MOS

TMRVC should define public success on metrics that reward explicit control:

- same-prompt reproducibility
- edit locality
- control-response monotonicity
- physical calibration
- trajectory transfer quality across speakers
- TTS/VC shared-control parity
- hard streaming stability under the `10 ms` causal contract

If evaluation collapses to MOS-only ranking, TMRVC forfeits much of its architectural advantage.

### 8. Do not use broad SOTA language as a crutch

Acceptable outcomes are only:

- TMRVC beats Fish Audio S2 on frozen programmable axes and says so narrowly
- TMRVC matches or exceeds Fish Audio S2 on both programmable and first-take axes and then may broaden the claim
- TMRVC fails, narrows the claim, and does not use SOTA language

## Required `v4` Additions

### A. Raw-audio bootstrap pipeline

New mandatory front-end to data preparation.

Required output:

- pseudo speakerized utterances
- confidence-bearing transcripts
- physical supervision bundle
- semantic acting annotations

### B. Intent Compiler

New component above the acoustic model.

Outputs:

- physical control targets
- acting latent prior
- pacing controls
- provenance and warnings

### C. Trajectory recorder / editor contract

New serializable artifact required.

Minimum contents:

- pointer trace
- realized physical trajectory
- realized acting state
- compiled pacing controls
- provenance of each control segment

### D. Shared acting transfer layer

Introduce a shared transformation path that can:

- infer trajectory from source speech
- normalize speaker-specific residue
- recondition target TTS or VC generation with the recovered acting trajectory

## Worker Impact

### Worker 01

- freeze the `v4` physical registry
- freeze acting texture latent contract
- freeze `IntentCompilerOutput`
- freeze `TrajectoryRecord`

### Worker 04

- expose creator-facing and deterministic inference APIs separately
- cut over serve/export/runtime to the `v4` contract
- support TTS and VC consumption of the same acting trajectory schema

### Worker 06

- define bootstrap QC gates
- define physical calibration, replay fidelity, edit locality, and transfer metrics
- freeze Fish S2 claim rules

### Worker 12

- replace the old `8-D` workshop with the `v4` physical-plus-latent control plane
- expose compile / patch / replay / transfer in the UI
