# arXiv Speech Survey (2026-03-07)

## Purpose

This note records the research survey used to justify the current `UCLM v3` plan as of **March 7, 2026**.

It is not a leaderboard dump. It exists to answer three practical questions:

1. which ideas are strong enough to influence the mainline plan
2. which ideas are promising but should remain research-track
3. which gaps still require TMRVC-specific validation rather than citation by analogy


## Scope

This survey focuses on:

- zero-shot / few-shot TTS
- controllable prosody generation
- low-latency or streamable speech generation / conversion
- disentanglement of timbre vs prosody / style
- implications for `UCLM v3` pointer-based causal serving


## Papers Reviewed

### 1. `E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS`

- arXiv: `2406.18009`
- date: `2024-06-26`
- link: <https://arxiv.org/abs/2406.18009>

Relevance:

- strong evidence that flow matching can remove explicit duration and MAS components in non-causal zero-shot TTS
- strong evidence that simple conditioning contracts can work at scale

Implication for TMRVC:

- supports keeping flow matching in the design space
- does **not** remove the need for a causal pointer contract, because E2 TTS is not a proof of 10 ms causal streaming parity


### 2. `F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching`

- arXiv: `2410.06885`
- date: `2024-10-09`
- link: <https://arxiv.org/abs/2410.06885>

Relevance:

- reinforces flow matching as a strong path for zero-shot naturalness, code-switching, and speed control
- shows that practical speed improvements matter as much as architecture choice

Implication for TMRVC:

- justifies keeping a flow-matching prosody or refinement path in the roadmap
- does **not** justify replacing the causal pointer mainline with a fully non-autoregressive stack


### 3. `ProsodyFM: Unsupervised Phrasing and Intonation Control for Intelligible Speech Synthesis`

- arXiv: `2412.11795`
- date: `2024-12-16`
- link: <https://arxiv.org/abs/2412.11795>

Relevance:

- strong evidence that prosody control should be modeled explicitly
- phrasing and terminal intonation remain critical even without manual labels

Implication for TMRVC:

- supports a dedicated prosody interface instead of pretending pointer pacing alone is enough
- also warns that phrasing control is richer than a single utterance-global latent, so time-local prosody remains a legitimate upgrade path


### 4. `DiffCSS: Diverse and Expressive Conversational Speech Synthesis with Diffusion Models`

- arXiv: `2502.19924`
- date: `2025-02-27`
- link: <https://arxiv.org/abs/2502.19924>

Relevance:

- strong evidence that dialogue context and sampled prosody are both necessary for conversational expressiveness
- diversity and context-appropriateness should be evaluated jointly

Implication for TMRVC:

- supports the requirement that `dialogue_context` and `local_prosody_latent` stay explicit
- supports the plan's anti-collapse metrics for same-text, different-context evaluation


### 5. `DiFlow-TTS: Discrete Flow Matching with Factorized Speech Tokens for Low-Latency Zero-Shot Text-To-Speech`

- arXiv: `2509.09631`
- date: `2025-09-11`
- link: <https://arxiv.org/abs/2509.09631>

Relevance:

- relevant because it explicitly factorizes prosody and acoustic detail while staying latency-aware
- suggests that factorized heads for prosody are compatible with efficient generation

Implication for TMRVC:

- supports keeping factorized prosody control in scope
- still does **not** supersede the need for TMRVC-specific parity tests because the deployment contract here is different: pointer-driven, 10 ms, multi-runtime parity


### 6. `StreamVoice: Streamable Context-Aware Language Modeling for Real-time Zero-Shot Voice Conversion`

- arXiv: `2401.11053`
- date: `2024-01-19`
- link: <https://arxiv.org/abs/2401.11053>

Relevance:

- direct evidence that streamable context-aware LM-based VC is feasible without future look-ahead
- important for keeping the VC path causal rather than letting TTS improvements force VC offline

Implication for TMRVC:

- supports the plan decision that VC should keep its own causal semantic-context path
- argues against forcing VC onto the TTS pointer loop unless there is measured benefit


### 7. `StyleStream: Real-Time Zero-Shot Voice Style Conversion`

- arXiv: `2602.20113`
- date: `2026-02-23`
- link: <https://arxiv.org/abs/2602.20113>

Relevance:

- very recent evidence for streamable style conversion with a strong information bottleneck
- especially relevant to timbre/style disentanglement and real-time deployment

Implication for TMRVC:

- strengthens the requirement that prompt conditioning must not silently carry prosody through an unrestricted side path
- supports the plan decision to require masks, confidences, and disentanglement checks rather than trusting prompt-based cloning by default


### 8. `Takin-VC: Expressive Zero-Shot Voice Conversion via Adaptive Hybrid Content Encoding and Enhanced Timbre Modeling`

- arXiv: `2410.01350`
- date: `2024-10-02`
- link: <https://arxiv.org/abs/2410.01350>

Relevance:

- useful evidence that expressive VC benefits from stronger timbre modeling and memory/context mechanisms
- also reinforces that breathing and paralinguistic cues matter

Implication for TMRVC:

- supports keeping VC semantic context explicit
- supports preserving breath / non-verbal metadata in the curation pipeline


## Synthesis

As of **March 7, 2026**, the literature supports the following:

- flow matching is a serious quality path for zero-shot TTS and prosody modeling
- dialogue context must remain explicit for conversational expressiveness
- streamable prompt-conditioned VC/TTS-like systems are feasible, but only with carefully bounded causality and disentanglement
- disentanglement is now a first-class design problem, not a cosmetic improvement

The literature does **not** yet remove the need for TMRVC-specific proof on:

- pointer-state numerical parity across Python / Rust / ONNX / VST
- exact 10 ms runtime behavior under force-advance and cache re-indexing
- 8-D physical-control supervision utility
- unified UI/backend auditability under multi-user workflows


## Decisions Taken for the Plan

### Tier 1 Mainline

Keep in the release-critical path:

- causal pointer-driven TTS
- shared serializable runtime contract
- explicit 8-D `voice_state` supervision path
- bounded dialogue context
- few-shot speaker adaptation with disentanglement checks

### Tier 2 Research

Keep behind a research boundary until parity and latency proof exists:

- backbone modernization swaps
- CFG acceleration variants such as `lazy` and `distilled`
- flow-matching upgrades beyond the minimum stable prosody path
- alternate vocoders or second-stage acoustic refinement


## Known Research Gaps

- No surveyed paper proves the exact `UCLM v3` combination of:
  - pointer-based causal alignment
  - 10 ms streaming
  - few-shot prompting
  - first-class 8-D physical control
  - multi-runtime parity

- Therefore, whenever this plan claims a design is "supported by the literature", that is an informed architectural direction, not a substitute for the repository's parity and evaluation gates.
