# Worker 01: Architecture and Model Contract

## Scope

Define the new v3 model contract so every other worker can build against stable interfaces.


## Primary Files

- `tmrvc-train/src/tmrvc_train/models/uclm_model.py`
- `tmrvc-train/src/tmrvc_train/models/uclm_transformer.py` (Modernized)
- `tmrvc-train/src/tmrvc_train/models/text_encoder.py`
- `tmrvc-train/src/tmrvc_train/models/reference_encoder.py` (New — prosody target extraction)
- `tmrvc-train/src/tmrvc_train/models/__init__.py`
- `tmrvc-core/src/tmrvc_core/types.py`
- `tmrvc-core/src/tmrvc_core/dialogue_types.py`
- `configs/constants.yaml`
- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/onnx-contract.md`
- `docs/design/rust-engine-design.md`


## Required Outcomes

- define pointer-state representation
- define pointer-head outputs
- define legacy duration path boundary
- define checkpoint compatibility policy
- define canonical `voice_state` / `delta_state` conditioning contract
- define how dialogue context and local prosody planning enter the model
- define required acoustic autoregressive inputs for TTS decoding
- define TTS teacher-forced versus autoregressive state-transition contract
- define whether VC consumes, bypasses, or emulates pointer state
- freeze the canonical unconditional-conditioning contract used by CFG
- define trainable 8-D `voice_state` target, mask, confidence, and provenance interfaces at the model boundary
- define the differentiable teacher-forced surrogate used to train the pointer without changing the hard runtime contract
- keep backbone modernization (`RoPE`, `GQA`, `SwiGLU`, `RMSNorm`, `FlashAttention2`) behind the README-defined `quality amplifier` gate rather than making pointer-core proof depend on it
- keep flow-matching prosody as the preferred expressive path, but not a blocker for the initial pointer-core proof
- define the single source of truth for serializable pointer / `voice_state` / cache-state schemas in `tmrvc-core`


## New Concepts To Introduce

### Shared Contract Ownership

Worker 01 must freeze the canonical serializable state schema in one place.

- **SpeakerProfile (Voice Identity Management):** The `SpeakerProfile` schema in `tmrvc-core` is the single source of truth for a voice in v3.0. The mainline contract is prompt-based only:
  1. **Prompt-based (Zero-Shot / Few-Shot In-Context):** Reference audio tokens, optional prompt cache, and global speaker embeddings.
- **Post-v3.0 extension boundary:** Weight-space adaptation (LoRA/adaptor deltas, merged fine-tuned checkpoints, per-actor ONNX variants) is not part of the v3.0 `SpeakerProfile` schema. If introduced later, it must ship as a separate artifact-management contract and must not change the prompt-based v3.0 serialization fields.
- `tmrvc-core` owns:
  - pointer-state field names and types
  - `voice_state` / `delta_state` external contract
  - dialogue-context record types
  - `SpeakerProfile` schema covering prompt evidence, speaker embedding metadata, ownership, and reproducibility fields
  - cache-state serialization contract needed by Python / Rust / ONNX / VST
- `configs/constants.yaml` remains the single source of truth for constants that affect tensor layout or stepping semantics
- model wrappers in train / serve / export may adapt this contract, but must not redefine it
- exported docs must point back to the same `tmrvc-core` schema rather than prose-only duplication

### Modern Transformer Backbone

The items in this section are the preferred `v3.0 quality amplifier` backbone, not the minimum pointer-core proof. Worker 01 must preserve a fallback path so Worker 06 can determine whether a failure came from the pointer architecture or from the modernization choice.

- **Positional Embedding:** **RoPE (Rotary Positional Embeddings)** for better extrapolation to long dialogues.
- **Attention Mechanism:** **GQA (Grouped Query Attention)** to reduce KV-cache memory footprint. The KV head count is frozen as `n_kv_heads` in `configs/constants.yaml` (initial value: 2, yielding 4x reduction with `uclm_n_heads=8`). Changing `n_kv_heads` requires updating all runtime parity tests.
- **Activation Function:** **SwiGLU** for better representational capacity.
- **Normalization:** **RMSNorm** with pre-norm configuration for training stability.
- **Training Acceleration:** Native **Flash Attention 2** support.
- **Cross-Attention Layer:** To support global prosody planning and non-local text context, the decoder block must include a Cross-Attention layer against the full `phoneme_features` sequence, replacing or augmenting the naive uniform frame-level interpolation.
- **Acoustic Autoregression:** The model must explicitly consume past emitted acoustic tokens (`a_ctx` from Stream A) in addition to control tokens (`b_ctx`), ensuring phase continuity and preventing waveform glitches during autoregressive generation.

`phoneme_features` must be defined as a canonical text-side feature bundle rather than an informal placeholder:

- required base field:
  - `phoneme_ids`
- required companion field when the language/backend provides suprasegmentals:
  - `text_suprasegmentals`
- canonical semantics of `text_suprasegmentals`:
  - language-scoped per-text-unit features for accent / tone / phrase-boundary cues
  - examples:
    - Japanese: `accent_upstep`, `accent_downstep`, `accent_phrase_break`
    - tonal languages: lexical tone id or equivalent normalized tone feature
- canonical shape:
  - `phoneme_ids`: `[B, L]`
  - `text_suprasegmentals`: `[B, L, d_supra]` (where `d_supra = d_suprasegmental` from `configs/constants.yaml`)
- frozen initial `d_supra` value: `4`
  - dim 0: `accent_upstep` — binary (0/1), accent rise marker (Japanese); 0 for languages without lexical accent
  - dim 1: `accent_downstep` — binary (0/1), accent fall marker (Japanese); 0 for languages without lexical accent
  - dim 2: `phrase_break` — binary (0/1), phrase-boundary marker; applicable to all languages
  - dim 3: `lexical_tone_id` — normalized tone index (0.0–1.0); 0.0 for non-tonal languages, language-specific mapping for tonal languages (e.g., Mandarin 4-tone + neutral mapped to 0.2/0.4/0.6/0.8/0.0)
  - languages that do not use a given dimension must set it to 0 (not omit it)
  - future dimensions may be appended; existing dimensions must not be renumbered
- implementation rule:
  - `phoneme_ids` remain the canonical text-unit ids
  - `text_suprasegmentals` are companion features, not a replacement vocabulary
  - the same serializable contract must survive dataset export, training, serving, and ONNX/Rust consumption

RoPE integration must be specified together with pointer execution:

- freeze how decoder-side attention interprets stalled or advanced pointer positions against RoPE-encoded text memory
- **Prevention of attention collapse during stalls:** To avoid the model misinterpreting prolonged pointer holds as repetitive text, the implementation must decouple "temporal steps" from "text progression" in the positional encoding.
  - **Preferred Strategy:** Use **Relative Position Accumulation** or **Progress-aware RoPE (PM-RoPE)** where the positional index for text units remains stable during stalls, while a separate temporal offset tracks the frame count within that unit.
- define whether cross-attention uses:
  - pointer-relative positional bias, or
  - an equivalent pointer-conditioned masking / biasing scheme
- define how repeated stall frames avoid drifting the effective text-position semantics
- define how skip-protection and force-advance update the effective relative-position reference so cached text memory remains coherent

Progress-aware positional structure is a valid research direction for this section. Worker 01 may evaluate a progress-aware RoPE-style scheme only as an ablation against the frozen explicit pointer-head contract. It must not delay or redefine the mainline pointer schema consumed by Workers 02/04/06.

### Pointer State

Per batch item:

- `text_index`
- `progress_value`
- `advance_logit` or `advance_prob`
- optional `boundary_confidence`
- optional `stall_frames`

Pointer state must evolve at every 10 ms frame step, regardless of whether text index advances on that step.

Worker 01 must also freeze the invariants:

- `text_index` is monotonic non-decreasing
- one frame step may advance by at most one text unit in the initial mainline
- `progress_value` clamp/reset behavior is explicit
- EOS / stop behavior at the final text unit is explicit
- recovery behavior after pathological prolonged hold is explicit and serializable
- force-advance semantics are explicit:
  - how `progress_value` is reset or clipped
  - how `acoustic_history` continuity is preserved without hidden-state surgery
  - how the runtime marks `forced_advance` in telemetry and serialized state
  - force-advance may mutate only serialized pointer-state fields and documented threshold/bias terms; it must not perform ad hoc transformer-cache smoothing, cache rewriting, or decoder-state cross-fades that are absent from the shared contract
- the advance decision rule is numeric and deterministic:
  - thresholding / comparison method for `advance_logit` or `advance_prob`
  - tie-break behavior at equality or near-equality
  - parity-safe handling of denormals / rounding across Python and Rust

### Differentiable Pointer Training Contract

The runtime pointer is discrete and serializable. Training may not hand-wave this by relying on an undefined hidden alignment path.

- hard runtime/exported state remains:
  - `text_index`
  - `progress_value`
  - `advance_logit` / `advance_prob`
  - optional `boundary_confidence`
  - optional `stall_frames`
- teacher-forced `latent_only` training must use a differentiable local surrogate over at most two text units: the current unit `i_t` and the next unit `i_t + 1`
- canonical teacher-forced text-conditioning rule:
  - compute `advance_prob_t = sigmoid(advance_logit_t)`
  - build the conditioning vector as an expected local mixture of text memory at `i_t` and `i_t + 1`
  - no hidden argmax, offline duration expansion, or full-sequence resampling is allowed in the mainline `latent_only` recipe
- canonical update rule split:
  - training: differentiable local expected update, optionally followed by straight-through hardening during the scheduled transition window
  - runtime/export: hard thresholded discrete advance using the same pointer-head outputs and the same frozen numeric rule
- supervised pointer losses, when present, must attach to the same `advance_logit` / `progress_delta` outputs used by the `latent_only` path; a separate training-only pointer head is forbidden
- any training-only surrogate state (for example a 2-way local pointer mass) is internal-only and must not replace the exported discrete pointer contract
- Worker 02 must define the annealing and hardening schedule against this exact contract rather than inventing a second pointer semantics

### Pointer Outputs

Model forward output must support:

- `logits_a`
- `logits_b`
- `advance_logit` (canonical key name for pointer advance logits; `pointer_logits` must not be used as the primary key)
- `progress_delta`
- optional `boundary_confidence` (must be model-predicted, not a dummy zero tensor)
- optional `legacy_log_durations`
- `hidden_states` (for downstream diagnostics and diversity loss)

The streaming forward path (`forward_streaming`) must also produce `advance_logit` and `progress_delta` via the pointer head. Omitting the pointer head from streaming makes pointer-driven runtime progression impossible.

### Acting Control Inputs

The architecture must reserve explicit inputs for:

- `explicit_voice_state` (`[B, T, 8]` or `[B, 8]`)
- `delta_voice_state` (`[B, T, 8]` or `[B, 8]`)
- optional `ssl_voice_state`
- `cfg_scale` (scalar guidance weight applied at the transformer level; replaces a dedicated `style_guidance_embedding` input)
- dialogue context embedding
- utterance-level acting intent
- local prosody latent
- turn-taking pressure or interruption pressure
- phrase-boundary / breath tendency
- runtime pacing controls (`pace`, `hold_bias`, `boundary_bias`) — these are consumed by the pointer head or pointer-state update rule as bias terms on the advance/hold decision, not as model embedding inputs. Worker 01 must document how each parameter modulates the advance threshold or progress accumulation.

CFG-based acting control is achieved by scaling between conditional and unconditional forward passes using `cfg_scale`, not by injecting a separate style guidance embedding.

The unconditional pass contract must be frozen exactly here:

- drop or zero:
  - `explicit_voice_state`
  - `delta_voice_state`
  - `ssl_voice_state`
  - `speaker_embed`
  - `prompt_codec_tokens` or `prompt_kv_cache`
  - `dialogue_context`
  - `acting_intent`
  - `local_prosody_latent`
- preserve:
  - `phoneme_ids`
  - `language_ids`
  - causal `acoustic_history`
  - pointer state
- implementation rule:
  - the same conditioning mask schema must be used in training, PyTorch inference, ONNX export, and Rust runtime

This prevents CFG from leaking timbre or prosody through an undocumented side path.

The CFG contract must explicitly cover pointer-side outputs, not only acoustic logits:

- `guided_logits_a = uncond_logits_a + cfg_scale * (cond_logits_a - uncond_logits_a)`
- `guided_logits_b = uncond_logits_b + cfg_scale * (cond_logits_b - uncond_logits_b)`
- `guided_advance_logit = uncond_advance_logit + cfg_scale * (cond_advance_logit - uncond_advance_logit)`
- `guided_progress_delta = uncond_progress_delta + cfg_scale * (cond_progress_delta - uncond_progress_delta)`
- `guided_boundary_confidence = uncond_boundary_confidence + cfg_scale * (cond_boundary_confidence - uncond_boundary_confidence)` when boundary confidence is enabled
- the pointer update rule in `full` CFG mode must consume the guided pointer outputs above
- `off` mode consumes the conditional outputs directly
- if `lazy` or `distilled` CFG are later enabled, they must approximate the same pointer-side semantics and pass the same parity tests; they are not initial v3.0 requirements

`state_cond` may remain as an internal fused representation, but it must not be the only documented public conditioning contract.


## Tensor Contract That Must Be Fixed Here

### Physical-First Control

Initial v3 mainline must preserve the project rule that explicit 8-D physical control is first-class.

- canonical external control inputs:
  - `explicit_voice_state`
  - `delta_voice_state`
- canonical internal fused condition:
  - `state_cond`
- allowed abstraction:
  - style / acting labels may modulate or supervise these paths
- forbidden abstraction:
  - replacing physical control with label-only conditioning in the mainline contract

Canonical supervised artifact contract:

- `voice_state_targets`
  - shape: `[B, T, 8]`
- `voice_state_observed_mask`
  - shape: `[B, T, 8]`
  - indicates whether each physical dimension has usable evidence
- `voice_state_confidence`
  - shape: `[B, T, 8]` or `[B, T, 1]`
  - numeric confidence from the curation/export pipeline
- `voice_state_target_source`
  - serializable enum or provenance record identifying whether the target came from direct labels, pseudo-label estimation, or is absent

Worker 01 must document how missing or low-confidence dimensions are masked in loss computation rather than silently treated as zeros.

Before any downstream worker may bind controls, losses, or export fields to `voice_state`, Worker 01 must freeze the canonical dimension registry itself, not only the tensor rank.

Required per-dimension registry fields:

- **Extractability constraint:** Before fixing a dimension, Worker 01 must confirm that a stable OSS estimator exists (e.g., Pitch, Energy, CPP/H1-H2, HNR). Abstract dimensions (e.g., "Tension", "Husky") must NOT be included in the canonical 8-D state unless a reliable continuous-value extractor is identified.
- stable dimension id
- short public name
- physical interpretation
- unit or normalized scale
- valid numeric range
- directionality semantics
  - what positive deltas mean
  - what negative deltas mean
- neutral / default value
- whether the dimension is frame-local, slowly varying, or both
- expected primary observable proxies for validation
- estimator-to-dimension mapping contract used by pseudo-label providers

This registry must be owned in `tmrvc-core` and referenced by docs, training, serving, export, VST automation, and Gradio. Reusing index `k` with different semantics across workers is forbidden.

**Frozen initial 8-D `voice_state` dimension registry:**

| idx | id | public name | physical interpretation | unit (normalized) | range | positive delta means | negative delta means | neutral | temporal | primary OSS estimator |
|-----|-----|-------------|----------------------|-------------------|-------|---------------------|---------------------|---------|----------|-----------------------|
| 0 | `pitch_level` | Pitch Level | fundamental frequency (F0) | log-Hz, min-max normalized | [0.0, 1.0] | higher pitch | lower pitch | 0.5 | frame-local | FCPE / CREPE / parselmouth |
| 1 | `pitch_range` | Pitch Range | F0 standard deviation over local window | normalized std | [0.0, 1.0] | more melodic variation | flatter intonation | 0.3 | slowly varying | derived from F0 trajectory (window = 0.5s) |
| 2 | `energy_level` | Energy Level | RMS amplitude | dB, min-max normalized | [0.0, 1.0] | louder | quieter | 0.5 | frame-local | librosa / torchaudio RMS |
| 3 | `pressedness` | Pressedness | phonation compression / glottal adduction proxy | normalized CPP + inverse H1-H2 composite | [0.0, 1.0] | firmer / more pressed closure | looser / more relaxed closure | 0.35 | slowly varying | parselmouth CPP + H1-H2 |
| 4 | `spectral_tilt` | Spectral Tilt | spectral brightness (slope of log-spectrum) | normalized slope | [0.0, 1.0] | brighter / tenser | darker / softer | 0.5 | frame-local | linear regression on log-magnitude spectrum |
| 5 | `breathiness` | Breathiness | inverse harmonics-to-noise ratio | normalized 1/HNR | [0.0, 1.0] | more breathy | more clear/pressed | 0.2 | frame-local | parselmouth HNR (inverted and normalized) |
| 6 | `voice_irregularity` | Voice Irregularity | jitter + shimmer composite | normalized composite | [0.0, 1.0] | more irregular / tense | more regular / smooth | 0.15 | frame-local | parselmouth jitter (local) + shimmer (local) |
| 7 | `openness` | Openness | vocal-tract opening proxy | normalized F1 / vowel-openness proxy | [0.0, 1.0] | more open / wider articulation | more closed / muffled articulation | 0.5 | slowly varying | formant tracker / vowel-region F1 proxy |

Registry rules:
- all dimensions are normalized to [0.0, 1.0] so they can share a single UI slider range and loss scale
- pseudo-label providers (Worker 08) must map their raw estimator outputs to these normalized ranges using the calibration contract
- dimensions where the estimator is unavailable or unreliable for a given sample must be marked as unobserved in `voice_state_observed_mask`, not set to the neutral value
- future dimension additions are append-only (idx >= 8) and require a registry version bump in `configs/constants.yaml`
- timing authority is explicitly outside the canonical `voice_state` registry:
  - `pace`, `hold_bias`, and `boundary_bias` are the runtime-authoritative timing controls
  - `voice_state` may correlate with timing indirectly, but must not become a second primary timing-control path

### Dialogue Context

`dialogue_context` in v3 mainline is text-side context, not raw waveform.

- canonical multi-turn representation: encoded previous-turn text/context embeddings with turn-role markers
- canonical shape: `[B, C_ctx, d_model]`
- reduced form after pooling: `[B, d_model]`
- contents:
  - previous turn text tokens after text encoder
  - speaker/role embedding per turn
  - optional scene/context tag embedding

The multi-turn `[B, C_ctx, d_model]` form is the canonical API-boundary contract. Internal projectors (e.g. `DialogueContextProjector`) may pool or reduce to `[B, d_model]` before injection into frame features. Both forms must be supported: the 3D form for full context fidelity, and the 2D pooled form as a convenience shorthand.

This keeps the runtime causal and bounded in VRAM. Raw prior audio is not the default dialogue-context modality.

### Suprasegmental Text Features

Languages such as Japanese and tonal languages require structured suprasegmental cues that cannot be recovered from bare `phoneme_ids` alone.

- canonical public input:
  - `text_suprasegmentals`
- canonical storage policy:
  - exported as a text-unit-aligned artifact derived deterministically from the text frontend
- canonical semantics:
  - every row aligns to one canonical text unit
  - missing features must be represented explicitly with masks or sentinel values, not silently dropped
- canonical training/runtime rule:
  - if a language/backend declares suprasegmental support, the mainline batch/request contract must either provide `text_suprasegmentals` or mark the sample/request as downgraded fallback mode
- validation rule:
  - Worker 06 must include at least one parity test proving that suprasegmental features survive Python -> export -> Rust/ONNX roundtrips without index drift

VC semantic-context policy:

- VC does not adopt the TTS pointer loop by default
- VC must still expose a semantic-context path combining:
  - semantic features extracted from source audio, and
  - optional text / dialogue context from surrounding turns
- Worker 01 must document the fusion site, such as bounded cross-attention from VC semantic features into dialogue-context memory
- this path must remain causal and must not force offline look-ahead into the VC runtime

Canonical export policy:

- exported training assets preserve raw text/context graph as the canonical source
- pre-encoded dialogue-context embeddings may exist only as optional derived caches keyed by encoder/checkpoint hash
- model correctness or reproducibility must never depend on a stale pre-encoded context artifact

### Acoustic History

`acoustic_history` is mandatory for causal autoregressive TTS decoding.

- purpose: provide past emitted acoustic/control-token context for next-step prediction
- **frozen canonical representation: raw token ids before embedding**
  - canonical shape: `[B, T_hist, n_codebooks + n_slots]` (acoustic tokens from Stream A plus control tokens from Stream B)
  - rationale: token ids are serializable across Python / Rust / ONNX / VST without depending on a specific embedding checkpoint; the model's own embedding layer converts them to `[B, T_hist, d_model]` internally
  - the embedded form `[B, T_hist, d_model]` is an internal intermediate, not the public API-boundary contract
  - Worker 04 runtime caches may store embedded KV projections for efficiency, but the authoritative state-transition contract operates on token ids

Training and runtime must use the same conceptual state machine:

- teacher-forced training:
  - `acoustic_history` is built from gold prior codec/control tokens
  - `pointer_state_t` is consumed and `pointer_state_t+1` is supervised
- autoregressive runtime:
  - `acoustic_history` is built from emitted prior codec/control tokens
  - `pointer_state_t` is consumed and `pointer_state_t+1` is fed back into the next step

Worker 01 must freeze this state-transition contract so Worker 04 does not invent a different runtime loop.

### Codec / Token Hierarchy

Worker 01 must explicitly choose the initial mainline token schedule instead of leaving `n_codebooks` implicit.

**Frozen initial v3 policy: flattened per-frame multi-codebook prediction.**

Rationale:

- the existing codebase predicts `n_codebooks=8` acoustic tokens (stream A) and `n_slots=4` control tokens (stream B) per frame simultaneously
- a flattened schedule keeps the 10 ms frame contract simple and avoids sub-step latency
- hierarchical/delayed prediction may be revisited as a future quality improvement, but must not be the initial v3 mainline to avoid scope creep

**SOTA context:** Nearly all top-performing systems (CosyVoice 3, MiniMax-Speech, DiSTAR, Qwen3-TTS) use a 2-stage pipeline where AR generates coarse/semantic tokens and a non-AR module (flow matching, diffusion, or DiT) refines them into full acoustic detail. The flattened single-stage policy is a known quality ceiling trade-off accepted for initial v3 simplicity. The 2-stage upgrade path is defined in § Acoustic Refinement Roadmap below.

Flattened policy details:

- all codebook slots within one frame step are predicted in parallel by the transformer
- pointer advancement couples to frame progression only (one frame = one pointer step opportunity)
- CFG applies at the frame level across all codebook slots simultaneously
- waveform-decoder quality gates evaluate full-frame codec token tuples, not individual codebook slots

Interaction rules:

- pointer pacing: coupled to frame steps only
- CFG: applied to the full logits output per frame
- waveform-decoder quality: evaluated on complete frame-level token tuples

### Local Prosody Latent and Prosody Predictor

`local_prosody_latent` is separate from `dialogue_context`.

- purpose: local delivery shape within the current utterance
- shape:
  - utterance-global form: `[B, d_prosody]` (canonical for initial v3; `d_prosody` is frozen in `configs/constants.yaml`)
  - time-local planning form: `[B, T_plan, d_prosody]` (future extension)

**Frozen initial v3 policy: utterance-global `[B, d_prosody]`.**

The `ProsodyPredictor` outputs `[B, d_prosody]`. When injecting into frame features via `DialogueContextProjector`, the projector must handle the 2D `[B, d_prosody]` input by broadcasting over the time dimension (unsqueeze to `[B, 1, d_prosody]`). The projector must also accept the 3D `[B, T, d_prosody]` form for future time-local planning without code changes.

**Time-local prosody upgrade schedule:**
Utterance-global prosody is insufficient for fine-grained drama-grade acting (mid-sentence emotion shifts, selective emphasis, trailing-off). Recent surveyed work supports token-level or segment-level prosody granularity. The upgrade to time-local `[B, T_plan, d_prosody]` must be evaluated at Stage B completion and scheduled for Stage C if drama acting quality gates are not met with the utterance-global form.
- **Latency constraint mitigation:** To maintain the 10 ms causal streaming budget, time-local prosody must NOT be predicted frame-by-frame. It must be generated in a single batched pass over the text-window *before* audio decoding starts, cached, and then referenced causally by the 10 ms pointer loop.
- **Pointer-synchronous addressing requirement:** Any cached time-local prosody plan must be anchored to canonical text units / pointer position, not to absolute elapsed frame index. Real-time controls such as `pace`, `hold_bias`, and `boundary_bias` are allowed to stretch or compress wall-clock realization, but they must not detach a prosodic event from its intended phoneme/span.
- **Runtime consumption rule:** If the future `[B, T_plan, d_prosody]` path is activated, the runtime must address it by current `text_index` plus optional within-unit `progress_value` interpolation. Holding longer on one unit prolongs that unit's prosody region; early advance moves the active region forward with the pointer.
- **Forbidden design:** Precomputing an absolute-time prosody schedule and replaying it regardless of pointer drift is forbidden, because it would conflict with causal pacing control and produce positionally broken prosody.

**Prosody Predictor Requirement:**
The architecture must define a stable prosody-prediction interface and satisfy it with a flow-matching predictor.

Required interface:

- one stable `predict_prosody(...)` API
- deterministic seed handling
- serializable latent contract
- bounded runtime cost compatible with the 10 ms serving budget

Required implementation:

- a flow-matching based prosody predictor that predicts the `local_prosody_latent` from text tokens and dialogue context during inference

- **Why Flow-matching?** It provides a deterministic mapping with high-diversity potential, superior to simple VAEs and more efficient than Diffusion for 1-step inference.
- Training mode:
  - Latent is extracted from the target waveform via the `ReferenceEncoder` and used as a target for Flow-matching training.
- Inference mode:
  - Predictor generates the latent in a single ODE-step (or N-step for higher quality).
  - Optional: Manual override or sampling for diversity.

### Reference Encoder (Training-Time Prosody Target Extraction)

The `ReferenceEncoder` is a training-time utility module that extracts the ground-truth `local_prosody_latent` from target audio. It is not used at inference time (the `ProsodyPredictor` replaces it).

- **Architecture:** Lightweight convolutional or transformer encoder over mel-spectrogram or codec features of the target utterance, projecting to `[B, d_prosody]`.
- **Training:** The `ReferenceEncoder` is trained jointly with the main model. Its output serves as:
  1. The direct conditioning input for the decoder during teacher-forced training (so the decoder learns to use prosody latents).
  2. The regression target for the flow-matching `ProsodyPredictor` loss.
- **Frozen contract:**
  - output shape: `[B, d_prosody]` (must match `ProsodyPredictor` output shape exactly)
  - the `ReferenceEncoder` must NOT have access to `speaker_embed` or `prompt_codec_tokens` to prevent timbre information from leaking into the prosody latent
  - if information leakage is detected (prosody latent predicts speaker identity above chance), an information bottleneck (e.g., VQ or capacity-limited linear projection) must be added
- **Ownership:** Worker 01 defines the interface and architecture. Worker 02 integrates it into the training loop. Worker 06 validates that the extracted latent is prosody-informative but speaker-agnostic.
- **Primary file:** `tmrvc-train/src/tmrvc_train/models/reference_encoder.py`

### Speaker Module Architecture (Speaker Encoder + SpeakerPromptEncoder + Prompt Resampler)

The speaker conditioning path comprises three distinct modules with clear responsibilities:

1. **Speaker Encoder** (`SpeakerEncoder`)
   - **Purpose:** Extract a global, utterance-level timbre embedding from reference audio.
   - **Input:** Raw audio waveform or pre-extracted features from a 3–10s reference clip.
   - **Output:** `speaker_embed` — shape `[B, d_speaker]`.
   - **Architecture:** Pre-trained speaker verification backbone (WavLM / Wespeaker / ECAPA-TDNN), optionally with a trainable projection head.
   - **Role at inference:** Always available. Serves as the primary timbre anchor and as the `speaker_embed`-only fallback when prompt cache budget is exceeded.

2. **SpeakerPromptEncoder** (`SpeakerPromptEncoder`)
   - **Purpose:** Encode the reference audio's codec tokens into a dense prompt representation that captures fine-grained timbre detail beyond what `speaker_embed` alone provides.
   - **Input:** `prompt_codec_tokens` — shape `[B, T_prompt, n_codebooks]`.
   - **Output:** `prompt_features` — shape `[B, T_prompt, d_model]` (before resampling).
   - **Architecture:** Lightweight 2-layer transformer encoder. Explicitly excluded from the RoPE/GQA/SwiGLU modernization requirement.
   - **Disentanglement bottleneck:** An Information Bottleneck or VQ layer must be applied to the output, restricting information flow to low-frequency timbre components and preventing prosody leakage.

3. **Prompt Resampler** (`PromptResampler`)
   - **Purpose:** Compress the variable-length `prompt_features` into a fixed-size summary to maintain the 10ms causal budget.
   - **Input:** `prompt_features` from `SpeakerPromptEncoder` — shape `[B, T_prompt, d_model]`.
   - **Output:** `prompt_kv_cache` — shape `[B, N_summary, d_model]` where `N_summary = 32` (frozen initial value; recorded in `configs/constants.yaml` as `n_prompt_summary_tokens`).
   - **Architecture:** Q-Former or Perceiver Resampler with learned query tokens.

**Inference-time combination rule:**
- `speaker_embed` is always injected as a global conditioning bias (added to or concatenated with frame features).
- `prompt_kv_cache` is prepended to the transformer's key/value sequences for cross-attention enrichment.
- When both are available, both are used. `speaker_embed` provides a stable global timbre anchor; `prompt_kv_cache` provides fine-grained texture detail.
- When only `speaker_embed` is available (prompt budget exceeded, Rust/VST fallback), the model operates in `speaker_embed`-only mode with documented quality degradation.
- Conflicts are resolved by design: `speaker_embed` and `prompt_kv_cache` are complementary (global + local timbre), not competing.

### Speaker Prompting and Timbre-Prosody Disentanglement (Zero-Shot / Few-Shot)

To support SOTA zero-shot/few-shot voice cloning while maintaining drama-grade acting, the architecture must abandon static speaker IDs in favor of an **In-Context Prompting** paradigm, combined with explicit disentanglement.

- **Acoustic & Text Prompting:** The model must accept `prompt_codec_tokens` (and optionally `prompt_text_tokens`) representing a 3-10 second reference audio clip. This serves as the in-context acoustic conditioning for the Codec LM.
- **Prompt Resampler (Computation Bottleneck):** To maintain the 10ms causal budget on CPU/ONNX, the model MUST NOT attend directly to the full sequence of prompt tokens (e.g., 1000 frames for 10s). Instead, the architecture must include a **Prompt Resampler** (e.g., Q-Former or Perceiver Resampler) to condense the 3-10s prompt into a fixed, small number of summary tokens (e.g., 32 or 64 tokens) before injection into the main transformer.
- **Speaker Encoder (Global Timbre):** A dedicated `Speaker Encoder` (e.g., integrating pre-trained WavLM/Wespeaker or a custom neural feature extractor) to extract a highly robust continuous `speaker_embed` from the prompt audio.
- **Disentanglement Constraint:** The model must explicitly disentangle *Timbre* from *Prosody*. When cloning a voice from a neutral read speech prompt, the model must borrow only the timbre from the `speaker_embed` and `prompt_codec_tokens`, while allowing the `dialogue_context` and the `Prosody Predictor` to override the acting/prosody.
- **Disentanglement Bottleneck:** To enforce this separation, the `Speaker Prompt Encoder` must incorporate an Information Bottleneck or Vector Quantization (VQ) layer, ensuring that attention mechanisms are restricted to low-frequency timbre components and cannot silently copy the prompt's prosody.

The few-shot contract must also freeze strict runtime budgets so prompt conditioning cannot silently violate the 10 ms causal target:

- persisted `SpeakerProfile` evidence may retain up to the raw enrollment clip and derived prompt artifacts for reproducibility
- active real-time conditioning must obey constants owned in `configs/constants.yaml`:
  - `max_prompt_seconds_active`
  - `max_prompt_frames`
  - `max_prompt_kv_tokens`
  - `max_prompt_cache_bytes`
- when enrollment audio exceeds the active runtime budget, Worker 04 must apply a deterministic prompt-selection/compression policy and record that policy in `SpeakerProfile`
- Rust/VST hard-real-time paths must have a documented downgrade path:
  - preferred order:
    - bounded `prompt_kv_cache`
    - reduced prompt summary
    - `speaker_embed`-only fallback
- prompt-budget enforcement is part of the public contract, not an implementation detail

Worker 01 must define the injection points for prompt tokens and ensure the attention masks (and bottleneck layers) prevent the prompt's neutral prosody from flattening the target utterance's dramatic intent.

**Codec-level disentanglement (future investigation path):**
Recent surveyed work suggests that resolving timbre-prosody entanglement at the codec/tokenizer level can be more fundamental than model-level bottlenecks. If model-level bottlenecking proves insufficient for zero-shot disentanglement quality, codec-level factorization should be evaluated as a v3.1 or v4 path. This is not an initial v3 requirement.

**SpeakerPromptEncoder modernization scope:** The `SpeakerPromptEncoder` is a lightweight utility module (2-layer encoder) and is explicitly excluded from the RoPE/GQA/SwiGLU modernization requirement. The modernized transformer backbone applies to `CodecTransformer` (the main decoder). The prompt encoder may adopt modern components later if prompt encoding becomes a quality bottleneck, but this is not a v3 initial mainline requirement.

### Waveform Decoder / Vocoder

The architecture must plan for the waveform decoder as a first-class quality bottleneck.

- initial v3 policy:
  - the codec decoder (inverse RVQ + neural vocoder) is the primary waveform generation path
  - decoder quality is evaluated as part of the end-to-end TTS quality gate, not assumed from token-level metrics alone
- candidates for enhancement (future):
  - `Vocos` (fast neural vocoder)
  - `HiFi-GAN` variants
  - codec-native decoders from `Encodec` / `DAC`
- streaming budget:
  - waveform decoding must complete within the 10 ms frame budget for real-time use
  - if an enhanced decoder exceeds this budget, a fast fallback path must exist
- Worker 04 must define the runtime decoder selection policy and Worker 06 must include waveform artifact metrics in the quality gate

### Acoustic Refinement Roadmap (v3.1 Quality Upgrade)

The dominant SOTA pattern (CosyVoice 3, MiniMax-Speech, DiSTAR, Qwen3-TTS) uses a 2-stage pipeline: AR over coarse/semantic tokens → non-AR refinement (flow matching / diffusion / DiT) for fine-grained acoustic detail. This factorization lifts the quality ceiling above what single-stage flattened prediction can achieve.

**v3 initial policy:** flattened single-stage codec prediction (see § Codec / Token Hierarchy). This is accepted as a scope trade-off.

**v3.1 planned upgrade path:**

- introduce a lightweight flow-matching or DiT-based acoustic refinement module as Stage 2
- Stage 2 takes coarse codec tokens (e.g., first 1-2 RVQ layers) from the AR model and refines to full RVQ depth
- the AR model's responsibility shifts to semantic/coarse token prediction, improving both quality and inference efficiency
- design references:
  - see `plan/arxiv_survey_2026_03.md` for the dated evidence supporting flow-matching and low-latency refinement
- streaming compatibility:
  - block-wise refinement (à la Qwen3-TTS) preserves low-latency streaming
  - the 10 ms causal AR core remains unchanged; refinement operates on buffered blocks
- ownership: Worker 01 defines the interface contract between AR core and refinement module. Worker 04 defines the runtime integration. Worker 06 validates quality uplift.
- trigger: if v3 initial quality gates (Worker 06 § External Baseline Parity) show a quality gap attributable to fine-grained acoustic detail, the v3.1 path is activated.
  - operational definition of "attributable to fine-grained acoustic detail": the gap is considered acoustic-detail-attributable when (a) token-level pointer alignment and prosody metrics are within acceptable range, AND (b) waveform artifact rate or spectral detail metrics (e.g., high-frequency energy, formant clarity) are measurably worse than the baseline, OR (c) ablating the vocoder/codec-decoder with a higher-fidelity alternative closes a significant portion of the MOS gap. Worker 06 must freeze this diagnostic protocol before Stage B.

Worker 01 must ensure that the initial v3 flattened codec output can serve as input to a future refinement module without architectural rework.

## Concrete Tasks

1. Update the mainline design notes in `docs/design/architecture.md` and `docs/design/unified-codec-lm.md`.
2. Align `docs/design/onnx-contract.md` and `docs/design/rust-engine-design.md` to the same pointer / voice-state contract.
3. Evaluate the modernization bundle (`RoPE`, `GQA`, `SwiGLU`, related efficiency changes) and land it only behind a documented rollback path if parity and latency hold.
4. Specify a new `forward_tts_pointer(...)` path in `uclm_model.py`.
5. Define the prosody-prediction interface and, if validated, the flow-matching variant that plugs into the main model graph.
6. Define the `Speaker Prompt Encoder` and the cross-attention/prefix-attention mechanism for `prompt_codec_tokens`.
7. Keep `forward_tts(...)` only as legacy compatibility or wrapper.
8. Introduce a dedicated pointer head module inside `uclm_model.py` or adjacent model file if separation is cleaner.
9. Document exact tensor shapes for:
   - text features
   - pointer state
   - explicit voice state
   - delta voice state
   - optional ssl voice state
   - acoustic history (now handling prompt prefixes)
   - dialogue-context features
   - local prosody latent (and its predicted version)
   - speaker prompt embeddings (`speaker_embed` and `prompt_codec_tokens`)
   - frame-step outputs
10. Define how pointer state interacts with cached streaming inference.
11. Define the frame-step state transition explicitly:
   - inputs at step `t`
   - outputs at step `t`
   - how `text_index` advances or holds
   - how `progress_value` clamps or resets
   - how EOS and terminal hold are represented
   - how `stall_frames` or equivalent deadlock detection is handled
   - how teacher-forced and autoregressive `acoustic_history` stay contract-compatible
12. Explicitly document which fields remain loadable from old checkpoints.
13. Define where dramatic acting lives:
   - what is controlled by pointer pacing
   - what is controlled by explicit 8-D voice state
   - what is controlled by prosody latent (predicted by the Prosody Predictor)
   - what is controlled by dialogue context
   - what is controlled by the Speaker Encoder (strictly Timbre, not pacing).
14. Define VC interaction with pointer state:
   - baseline v3 policy:
     - TTS uses pointer as the primary progression mechanism
     - VC does not require pointer for frame-synchronous conversion
   - Zero-Shot Few-Shot constraint for VC:
     - Applying few-shot `SpeakerPromptEncoder` features directly to VC is excluded from the initial v3.0 required scope. VC remains focused on converting against a fixed model/speaker space initially to reduce complexity, while TTS carries the zero-shot proof obligations.
   - optional future path:
     - expressive VC may consume a pacing-control analogue or pointer-conditioned semantic plan
   - explicit non-goal for initial v3:
     - do not force VC onto the TTS pointer path if it adds latency or architectural bloat without a validated benefit
   - required semantic-context path:
     - define how VC semantic features cross-attend to optional dialogue/text context
15. Specify anti-collapse expectations:
   - same text under different context must not map to a single delivery due to dialogue-context and prosody-latent variation.
   - Zero-shot prompting must not flatten the acting; same text under different context must still vary despite using the exact same speaker prompt.
16. **Specify Quantization Policy:**
    - define quantization compatibility constraints, but treat FP8 / INT8 deployment optimization as post-v3.0 unless explicitly promoted into the release checklist.
17. Freeze the canonical public API boundary:
   - training/model API accepts raw state inputs (`explicit_voice_state`, `delta_voice_state`, optional `ssl_voice_state`)
   - export/runtime wrappers may consume or emit fused `state_cond`, but only if the mapping to raw state inputs is documented and tested
18. Define RoPE / pointer interaction rules for cross-attention and cached text memory.
    - evaluate VoiceStar PM-RoPE (arXiv:2505.19462) only as a post-freeze comparison point
    - do not allow PM-RoPE evaluation to replace, postpone, or weaken the explicit pointer-head mainline contract in v3.0
19. Freeze the unconditional-conditioning mask schema shared by CFG training and inference.
    - freeze the guided pointer-output formulas and post-guidance clamps used by `full` CFG mode
20. Freeze the `SpeakerProfile` serialization contract in `tmrvc-core/src/tmrvc_core/types.py` and ownership fields used by Worker 04 and Worker 12. The canonical field list (prompt evidence, speaker embedding metadata, ownership, reproducibility) must be frozen here and referenced by `docs/design/speaker-profile-spec.md`.
    - include prompt-budget metadata and deterministic prompt-selection/compression provenance
21. Document how `voice_state_targets`, masks, and confidences enter the model or loss path without conflating unknown with neutral.
22. Evaluate time-local prosody as a Stage C upgrade only if Worker 06 quality gates show the utterance-global latent is insufficient.
23. Define the v3.1 acoustic refinement interface contract so future modules can attach without breaking mainline contracts.
24. Document codec-level disentanglement as a future investigation path rather than an initial v3 requirement.
25. Define the `ReferenceEncoder` module:
    - architecture (convolutional or lightweight transformer over mel/codec features)
    - output shape `[B, d_prosody]` matching the `ProsodyPredictor` target
    - information isolation: no access to speaker identity inputs
    - leakage detection test: prosody latent must not predict speaker identity above chance
26. Define the `SpeakerEncoder` / `SpeakerPromptEncoder` / `PromptResampler` module boundary:
    - `SpeakerEncoder` outputs `speaker_embed` `[B, d_speaker]`
    - `SpeakerPromptEncoder` outputs `prompt_features` `[B, T_prompt, d_model]` with disentanglement bottleneck
    - `PromptResampler` compresses to `prompt_kv_cache` `[B, N_summary, d_model]` with `N_summary` frozen in `configs/constants.yaml`
    - document the inference-time combination rule and `speaker_embed`-only fallback behavior


## Proposed API Shape

```python
# Voice Cloning / Prompt Extraction Phase
speaker_embed, prompt_kv_cache = model.encode_speaker_prompt(
    prompt_audio=...,        # or prompt_codec_tokens
    prompt_text=...,         # optional text of the prompt
)

# Prosody Prediction Phase
prosody_latent = model.predict_prosody(
    phoneme_ids=...,
    dialogue_context=...,
    speaker_embed=speaker_embed,
    style_prompt=..., # optional
)
```

```python
# Decoding Phase
out = model.forward_tts_pointer(
    phoneme_ids=...,
    text_suprasegmentals=...,   # [B, L, d_supra], optional only for unsupported-language fallback
    language_ids=...,
    pointer_state=...,
    acoustic_history=...,        # past a_ctx (Stream A) AND b_ctx (Stream B)
    prompt_kv_cache=prompt_kv_cache, # cached in-context prompt to avoid recomputation
    dialogue_context=...,        # [B, C_ctx, d_model]
    speaker_embed=speaker_embed,
    explicit_voice_state=...,
    delta_voice_state=...,
    ssl_voice_state=...,
    local_prosody_latent=prosody_latent,
    frame_horizon=...,           # teacher-forced training only
)
```

Notes:

- `frame_horizon` is allowed only for bounded training windows or offline evaluation. In the implementation, this is passed as `target_length` which must be `Optional[int]` (default `None`). When `None`, the model derives the frame count from `target_b.shape[-1]` during training or runs open-ended during inference.
- mainline runtime must not require a precomputed target length.
- serving/runtime APIs must drive termination from pointer state and EOS conditions.
- `acoustic_history` is not optional in the causal decoding contract, even if a wrapper builds it from cache state. The implementation must actually consume this input, not accept and ignore it.
- `prompt_kv_cache` must be consumed by the model when provided (e.g., prepended to the transformer's key/value sequences). The implementation must not accept and silently ignore it.
- `dialogue_context` is text-side context embedding in the initial v3 contract; raw audio context must not be implied silently. The canonical shape is `[B, C_ctx, d_model]` but `[B, d_model]` (pooled) is also accepted.
- `text_suprasegmentals` is the canonical companion feature tensor for accent / tone / phrase-boundary cues. `phoneme_ids` without this tensor is not considered fully train-ready for languages/backends that declare suprasegmental support.
- `predict_prosody` and `encode_speaker_prompt` must be efficient. `prompt_kv_cache` enables retaining the zero-shot voice across long conversational turns without re-encoding the reference audio.
- `state_cond` is an internal/export-facing fused representation, not the canonical model-level public input contract.
- `n_codebooks` scheduling policy is frozen as flattened per-frame prediction (see Codec / Token Hierarchy section).

Expected output keys:

```python
{
    "logits_a": ...,
    "logits_b": ...,
    "advance_logit": ...,       # canonical key (not "pointer_logits")
    "progress_delta": ...,
    "boundary_confidence": ..., # must be model-predicted, not dummy zeros
    "hidden_states": ...,       # for diagnostics and diversity loss
    "next_pointer_state": ...,  # populated at inference time
}
```


## Guardrails

- do not remove legacy duration code in this worker
- do not change trainer semantics yet
- do not invent hidden implicit state; all runtime state must be serializable
- do not collapse 8-D physical control into undocumented style embeddings
- do not treat missing `voice_state` supervision as zero-valued physical neutrality
- do not claim drama-grade acting from pointer alone; prosody predictor and context paths must be explicit
- **do not let the speaker prompt dictate prosody; ensure architectural bottlenecking or disentanglement separates timbre from timing/pitch curves.**
- do not claim Japanese/tonal naturalness while discarding `text_suprasegmentals` after G2P
- do not reintroduce target-length estimation as a runtime dependency
- do not define `dialogue_context` as an ambiguous placeholder; modality and shape must be explicit
- do not leave teacher-forced and autoregressive pointer loops semantically mismatched
- do not force initial v3 VC onto pointer progression unless a concrete validated use case justifies it
- do not introduce backbone changes that break parity gates without a documented rollback path
- do not use force-advance to justify undocumented hidden-state edits that Python/Rust/ONNX/VST cannot serialize identically


## Handoff Contract

Before handing off:

- `docs/design/architecture.md` and `docs/design/unified-codec-lm.md` reflect the stable contract
- `docs/design/onnx-contract.md` and `docs/design/rust-engine-design.md` reflect the same contract
- `tmrvc-core` exposes the shared serializable schema used by all runtimes
- `uclm_model.py` has stable function names and includes the Prosody Predictor interface
- `reference_encoder.py` implements `ReferenceEncoder` with verified speaker-agnosticism and `[B, d_prosody]` output shape
- Speaker modules (`SpeakerEncoder`, `SpeakerPromptEncoder`, `PromptResampler`) have stable interfaces; `PromptResampler` output shape is `[B, 32, d_model]`
- `tmrvc-core` exposes `SpeakerProfile`, pointer state, and `voice_state` supervision schema without runtime-specific reinterpretation
- output dict keys are fixed
- worker 02 and worker 04 can code against the interfaces without guessing


## Required Tests

- shape tests for pointer outputs
- backward compatibility test for loading a v2 checkpoint with `strict=False`
- model smoke test for `forward_tts_pointer`
- test for `predict_prosody` output shape and determinism (given same inputs/seed)
- shape test for explicit / delta / ssl voice-state conditioning inputs
- shape test for dialogue-context and prosody-conditioning inputs
- conditional vs unconditional mask contract test covering all conditioning fields
- state-transition contract test for teacher-forced versus autoregressive loop compatibility
- contract test that exported `state_cond` wrapper matches the canonical raw-state API
- force-advance state-transition test covering `progress_value` reset and `acoustic_history` continuity semantics
- codec-hierarchy schedule contract test for the chosen `n_codebooks` policy
- shape/mask/confidence test for `voice_state_targets`
- `ReferenceEncoder` output shape test (`[B, d_prosody]` matches `ProsodyPredictor` target)
- `ReferenceEncoder` speaker-agnosticism test (prosody latent does not predict speaker identity above chance)
- `SpeakerEncoder` → `SpeakerPromptEncoder` → `PromptResampler` pipeline shape test
- `PromptResampler` output shape test (`[B, N_summary, d_model]` with frozen `N_summary`)
- `speaker_embed`-only fallback mode smoke test (model produces valid output without `prompt_kv_cache`)
