# Worker 01: Architecture and Model Contract

## Scope

Define the new v3 model contract so every other worker can build against stable interfaces.


## Primary Files

- `tmrvc-train/src/tmrvc_train/models/uclm_model.py`
- `tmrvc-train/src/tmrvc_train/models/uclm_transformer.py` (Modernized)
- `tmrvc-train/src/tmrvc_train/models/text_encoder.py`
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
- define release-critical versus research-extension boundaries inside the model stack
- evaluate but do not hard-block mainline on modernization features such as `RoPE`, `GQA`, `SwiGLU`
- evaluate but do not hard-block mainline on flow-matching prosody variants beyond the minimum stable prosody path
- define the single source of truth for serializable pointer / `voice_state` / cache-state schemas in `tmrvc-core`


## New Concepts To Introduce

### Shared Contract Ownership

Worker 01 must freeze the canonical serializable state schema in one place.

- `tmrvc-core` owns:
  - pointer-state field names and types
  - `voice_state` / `delta_state` external contract
  - dialogue-context record types
  - `SpeakerProfile` schema covering prompt evidence, speaker embedding metadata, ownership, and reproducibility fields
  - cache-state serialization contract needed by Python / Rust / ONNX / VST
- `configs/constants.yaml` remains the single source of truth for constants that affect tensor layout or stepping semantics
- model wrappers in train / serve / export may adapt this contract, but must not redefine it
- exported docs must point back to the same `tmrvc-core` schema rather than prose-only duplication

### Release-Critical vs Research Boundary

Worker 01 must explicitly separate:

- release-critical Tier 1:
  - pointer state machine
  - causal text/context conditioning
  - 8-D physical-control contract
  - few-shot prompt contract
  - stable streaming/cache semantics
- research-extension Tier 2:
  - backbone modernization
  - advanced prosody predictors
  - CFG acceleration variants
  - alternative acoustic refinement stages

Tier 2 features may be adopted only if they preserve Tier 1 interfaces and keep a documented rollback path.

### Modern Transformer Backbone (Tier 2)

The UCLM v3 core should evaluate and adopt the following SOTA LLM practices if parity and latency gates hold. These are Tier 2 research-extensions; the mainline must not hard-block on them (see § Release-Critical vs Research Boundary).

- **Positional Embedding:** **RoPE (Rotary Positional Embeddings)** for better extrapolation to long dialogues.
- **Attention Mechanism:** **GQA (Grouped Query Attention)** to reduce KV-cache memory footprint by 4x-8x during 10ms-step serving.
- **Activation Function:** **SwiGLU** for better representational capacity.
- **Normalization:** **RMSNorm** with pre-norm configuration for training stability.
- **Training Acceleration:** Native **Flash Attention 2** support.
- **Cross-Attention Layer:** To support global prosody planning and non-local text context, the decoder block must include a Cross-Attention layer against the full `phoneme_features` sequence, replacing or augmenting the naive uniform frame-level interpolation.
- **Acoustic Autoregression:** The model must explicitly consume past emitted acoustic tokens (`a_ctx` from Stream A) in addition to control tokens (`b_ctx`), ensuring phase continuity and preventing waveform glitches during autoregressive generation.

RoPE integration must be specified together with pointer execution:

- freeze how decoder-side attention interprets stalled or advanced pointer positions against RoPE-encoded text memory
- define whether cross-attention uses:
  - pointer-relative positional bias, or
  - an equivalent pointer-conditioned masking / biasing scheme
- define how repeated stall frames avoid drifting the effective text-position semantics
- define how skip-protection and force-advance update the effective relative-position reference so cached text memory remains coherent

Progress-aware positional structure is a valid research direction for this section. Worker 01 may evaluate a progress-aware RoPE-style scheme if it simplifies the pointer loop, but it must be treated as a research-track alternative until it proves parity with the explicit pointer contract.

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
  - how `acoustic_history` continuity is preserved to avoid audible discontinuity
  - how the runtime marks `forced_advance` in telemetry and serialized state
- the advance decision rule is numeric and deterministic:
  - thresholding / comparison method for `advance_logit` or `advance_prob`
  - tie-break behavior at equality or near-equality
  - parity-safe handling of denormals / rounding across Python and Rust

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
- representation:
  - either raw past codec/control token ids before embedding, or
  - already embedded autoregressive history accepted by the decoder wrapper
- minimum conceptual shape:
  - token ids: `[B, T_hist, n_codebooks]` for acoustic tokens plus control history
  - embedded history: `[B, T_hist, d_model]`

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
  - utterance-global form: `[B, d_prosody]` (canonical for initial v3)
  - time-local planning form: `[B, T_plan, d_prosody]` (future extension)

**Frozen initial v3 policy: utterance-global `[B, d_prosody]`.**

The `ProsodyPredictor` outputs `[B, d_prosody]`. When injecting into frame features via `DialogueContextProjector`, the projector must handle the 2D `[B, d_prosody]` input by broadcasting over the time dimension (unsqueeze to `[B, 1, d_prosody]`). The projector must also accept the 3D `[B, T, d_prosody]` form for future time-local planning without code changes.

**Time-local prosody upgrade schedule:**
Utterance-global prosody is insufficient for fine-grained drama-grade acting (mid-sentence emotion shifts, selective emphasis, trailing-off). Recent surveyed work supports token-level or segment-level prosody granularity. The upgrade to time-local `[B, T_plan, d_prosody]` must be evaluated at Stage B completion and scheduled for Stage C if drama acting quality gates are not met with the utterance-global form.

**Prosody Predictor Requirement:**
The architecture must define a stable prosody-prediction interface in Tier 1 and may satisfy it with a flow-matching predictor if the latency/parity gates hold.

Tier 1 minimum:

- one stable `predict_prosody(...)` API
- deterministic seed handling
- serializable latent contract
- bounded runtime cost compatible with the 10 ms serving budget

Tier 2 preferred path:

- a flow-matching based prosody predictor that predicts the `local_prosody_latent` from text tokens and dialogue context during inference

- **Why Flow-matching?** It provides a deterministic mapping with high-diversity potential, superior to simple VAEs and more efficient than Diffusion for 1-step inference.
- Training mode:
  - Latent is extracted from the target waveform (e.g., via a Reference Encoder) and used as a target for Flow-matching training.
- Inference mode:
  - Predictor generates the latent in a single ODE-step (or N-step for higher quality).
  - Optional: Manual override or sampling for diversity.

### Speaker Prompting and Timbre-Prosody Disentanglement (Zero-Shot / Few-Shot)

To support SOTA zero-shot/few-shot voice cloning while maintaining drama-grade acting, the architecture must abandon static speaker IDs in favor of an **In-Context Prompting** paradigm, combined with explicit disentanglement.

- **Acoustic & Text Prompting:** The model must accept `prompt_codec_tokens` (and optionally `prompt_text_tokens`) representing a 3-10 second reference audio clip. This serves as the in-context acoustic conditioning for the Codec LM.
- **Speaker Encoder (Global Timbre):** A dedicated `Speaker Encoder` (e.g., integrating pre-trained WavLM/Wespeaker or a custom neural feature extractor) to extract a highly robust continuous `speaker_embed` from the prompt audio.
- **Disentanglement Constraint:** The model must explicitly disentangle *Timbre* from *Prosody*. When cloning a voice from a neutral read speech prompt, the model must borrow only the timbre from the `speaker_embed` and `prompt_codec_tokens`, while allowing the `dialogue_context` and the `Prosody Predictor` to override the acting/prosody.
- **Disentanglement Bottleneck:** To enforce this separation, the `Speaker Prompt Encoder` must incorporate an Information Bottleneck or Vector Quantization (VQ) layer, ensuring that attention mechanisms are restricted to low-frequency timbre components and cannot silently copy the prompt's prosody.

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
  - see `plan/arxiv_survey_2026_03.md` for the dated evidence supporting flow-matching and low-latency refinement as a research-track direction
- streaming compatibility:
  - block-wise refinement (à la Qwen3-TTS) preserves low-latency streaming
  - the 10 ms causal AR core remains unchanged; refinement operates on buffered blocks
- ownership: Worker 01 defines the interface contract between AR core and refinement module. Worker 04 defines the runtime integration. Worker 06 validates quality uplift.
- trigger: if v3 initial quality gates (Worker 06 § External Baseline Parity) show a quality gap attributable to fine-grained acoustic detail, the v3.1 path is activated.

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
    - ensure the model graph is compatible with **FP8** or **INT8 (SmoothQuant/Weight-only)** for the Rust engine.
17. Freeze the canonical public API boundary:
   - training/model API accepts raw state inputs (`explicit_voice_state`, `delta_voice_state`, optional `ssl_voice_state`)
   - export/runtime wrappers may consume or emit fused `state_cond`, but only if the mapping to raw state inputs is documented and tested
18. Define RoPE / pointer interaction rules for cross-attention and cached text memory.
    - evaluate VoiceStar PM-RoPE (arXiv:2505.19462) as a candidate or comparison point
    - document whether PM-RoPE can replace or simplify the explicit pointer head
19. Freeze the unconditional-conditioning mask schema shared by CFG training and inference.
20. Freeze the `SpeakerProfile` serialization contract in `tmrvc-core/src/tmrvc_core/types.py` and ownership fields used by Worker 04 and Worker 12. The canonical field list (prompt evidence, speaker embedding metadata, ownership, reproducibility) must be frozen here and referenced by `docs/design/speaker-profile-spec.md`.
21. Document how `voice_state_targets`, masks, and confidences enter the model or loss path without conflating unknown with neutral.
22. Evaluate time-local prosody as a Stage C upgrade only if Worker 06 quality gates show the utterance-global latent is insufficient.
23. Define the v3.1 acoustic refinement interface contract so future research-track modules can attach without breaking Tier 1.
24. Document codec-level disentanglement as a future investigation path rather than an initial v3 requirement.


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
- do not reintroduce target-length estimation as a runtime dependency
- do not define `dialogue_context` as an ambiguous placeholder; modality and shape must be explicit
- do not leave teacher-forced and autoregressive pointer loops semantically mismatched
- do not force initial v3 VC onto pointer progression unless a concrete validated use case justifies it
- do not make Tier 1 release contingent on a Tier 2 backbone swap unless a rollback path is documented


## Handoff Contract

Before handing off:

- `docs/design/architecture.md` and `docs/design/unified-codec-lm.md` reflect the stable contract
- `docs/design/onnx-contract.md` and `docs/design/rust-engine-design.md` reflect the same contract
- `tmrvc-core` exposes the shared serializable schema used by all runtimes
- `uclm_model.py` has stable function names and includes the Prosody Predictor interface
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
