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
- **Adopt Modern LLM Architecture (RoPE, GQA, SwiGLU) for SOTA performance and efficiency**
- **Define Flow-Matching based Prosody Predictor for high-diversity acting**
- define the single source of truth for serializable pointer / `voice_state` / cache-state schemas in `tmrvc-core`


## New Concepts To Introduce

### Shared Contract Ownership

Worker 01 must freeze the canonical serializable state schema in one place.

- `tmrvc-core` owns:
  - pointer-state field names and types
  - `voice_state` / `delta_state` external contract
  - dialogue-context record types
  - cache-state serialization contract needed by Python / Rust / ONNX / VST
- `configs/constants.yaml` remains the single source of truth for constants that affect tensor layout or stepping semantics
- model wrappers in train / serve / export may adapt this contract, but must not redefine it
- exported docs must point back to the same `tmrvc-core` schema rather than prose-only duplication

### Modern Transformer Backbone

The UCLM v3 core must adopt SOTA LLM practices:

- **Positional Embedding:** **RoPE (Rotary Positional Embeddings)** for better extrapolation to long dialogues.
- **Attention Mechanism:** **GQA (Grouped Query Attention)** to reduce KV-cache memory footprint by 4x-8x during 10ms-step serving.
- **Activation Function:** **SwiGLU** for better representational capacity.
- **Normalization:** **RMSNorm** with pre-norm configuration for training stability.
- **Training Acceleration:** Native **Flash Attention 2** support.

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

### Pointer Outputs

Model forward output must support:

- `logits_a`
- `logits_b`
- `advance_logit`
- `progress_delta`
- optional `boundary_confidence`
- optional `legacy_log_durations`

### Acting Control Inputs

The architecture must reserve explicit inputs for:

- `explicit_voice_state` (`[B, T, 8]` or `[B, 8]`)
- `delta_voice_state` (`[B, T, 8]` or `[B, 8]`)
- optional `ssl_voice_state`
- **`style_guidance_embedding`** (for CFG-based acting control)
- dialogue context embedding
- utterance-level acting intent
- local prosody latent
- turn-taking pressure or interruption pressure
- phrase-boundary / breath tendency

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

### Dialogue Context

`dialogue_context` in v3 mainline is text-side context, not raw waveform.

- representation: encoded previous-turn text/context embeddings with turn-role markers
- shape: `[B, C_ctx, d_model]`
- contents:
  - previous turn text tokens after text encoder
  - speaker/role embedding per turn
  - optional scene/context tag embedding

This keeps the runtime causal and bounded in VRAM. Raw prior audio is not the default dialogue-context modality.

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

### Local Prosody Latent and Prosody Predictor

`local_prosody_latent` is separate from `dialogue_context`.

- purpose: local delivery shape within the current utterance
- shape:
  - utterance-global form: `[B, d_prosody]`, or
  - time-local planning form: `[B, T_plan, d_prosody]`

**Prosody Predictor Requirement:**
The architecture must include a **Flow-matching based Prosody Predictor** that predicts the `local_prosody_latent` from text tokens and dialogue context during inference.

- **Why Flow-matching?** It provides a deterministic mapping with high-diversity potential, superior to simple VAEs and more efficient than Diffusion for 1-step inference.
- Training mode:
  - Latent is extracted from the target waveform (e.g., via a Reference Encoder) and used as a target for Flow-matching training.
- Inference mode:
  - Predictor generates the latent in a single ODE-step (or N-step for higher quality).
  - Optional: Manual override or sampling for diversity.

Worker 01 must document which form is canonical for initial v3 implementation and define the interface for the Prosody Predictor.

### Speaker Prompting and Timbre-Prosody Disentanglement (Zero-Shot / Few-Shot)

To support SOTA zero-shot/few-shot voice cloning while maintaining drama-grade acting, the architecture must abandon static speaker IDs in favor of an **In-Context Prompting** paradigm, combined with explicit disentanglement.

- **Acoustic & Text Prompting:** The model must accept `prompt_codec_tokens` (and optionally `prompt_text_tokens`) representing a 3-10 second reference audio clip. This serves as the in-context acoustic conditioning for the Codec LM.
- **Speaker Encoder (Global Timbre):** A dedicated `Speaker Encoder` (e.g., integrating pre-trained WavLM/Wespeaker or a custom neural feature extractor) to extract a highly robust continuous `speaker_embed` from the prompt audio.
- **Disentanglement Constraint:** The model must explicitly disentangle *Timbre* from *Prosody*. When cloning a voice from a neutral read speech prompt, the model must borrow only the timbre from the `speaker_embed` and `prompt_codec_tokens`, while allowing the `dialogue_context` and the `Prosody Predictor` to override the acting/prosody.

Worker 01 must define the injection points for prompt tokens and ensure the attention masks prevent the prompt's neutral prosody from flattening the target utterance's dramatic intent.

## Concrete Tasks

1. Update the mainline design notes in `docs/design/architecture.md` and `docs/design/unified-codec-lm.md`.
2. Align `docs/design/onnx-contract.md` and `docs/design/rust-engine-design.md` to the same pointer / voice-state contract.
3. **Implement the Modernized Transformer Backbone (RoPE, GQA, SwiGLU) in `uclm_transformer.py`.**
4. Specify a new `forward_tts_pointer(...)` path in `uclm_model.py`.
5. Define the **Flow-matching Prosody Predictor** architecture and its integration into the main model graph.
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
15. Specify anti-collapse expectations:
   - same text under different context must not map to a single delivery due to dialogue-context and prosody-latent variation.
   - Zero-shot prompting must not flatten the acting; same text under different context must still vary despite using the exact same speaker prompt.
16. **Specify Quantization Policy:**
    - ensure the model graph is compatible with **FP8** or **INT8 (SmoothQuant/Weight-only)** for the Rust engine.
17. Freeze the canonical public API boundary:
   - training/model API accepts raw state inputs (`explicit_voice_state`, `delta_voice_state`, optional `ssl_voice_state`)
   - export/runtime wrappers may consume or emit fused `state_cond`, but only if the mapping to raw state inputs is documented and tested


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

# Decoding Phase
out = model.forward_tts_pointer(
    phoneme_ids=...,
    language_ids=...,
    pointer_state=...,
    acoustic_history=...,        # required autoregressive audio/control context (includes generated tokens)
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

- `frame_horizon` is allowed only for bounded training windows or offline evaluation.
- mainline runtime must not require a precomputed target length.
- serving/runtime APIs must drive termination from pointer state and EOS conditions.
- `acoustic_history` is not optional in the causal decoding contract, even if a wrapper builds it from cache state.
- `dialogue_context` is text-side context embedding in the initial v3 contract; raw audio context must not be implied silently.
- `predict_prosody` and `encode_speaker_prompt` must be efficient. `prompt_kv_cache` enables retaining the zero-shot voice across long conversational turns without re-encoding the reference audio.
- `state_cond` is an internal/export-facing fused representation, not the canonical model-level public input contract.

Expected output keys:

```python
{
    "logits_a": ...,
    "logits_b": ...,
    "advance_logit": ...,
    "progress_delta": ...,
    "boundary_confidence": ...,
    "next_pointer_state": ...,
}
```


## Guardrails

- do not remove legacy duration code in this worker
- do not change trainer semantics yet
- do not invent hidden implicit state; all runtime state must be serializable
- do not collapse 8-D physical control into undocumented style embeddings
- do not claim drama-grade acting from pointer alone; prosody predictor and context paths must be explicit
- **do not let the speaker prompt dictate prosody; ensure architectural bottlenecking or disentanglement separates timbre from timing/pitch curves.**
- do not reintroduce target-length estimation as a runtime dependency
- do not define `dialogue_context` as an ambiguous placeholder; modality and shape must be explicit
- do not leave teacher-forced and autoregressive pointer loops semantically mismatched
- do not force initial v3 VC onto pointer progression unless a concrete validated use case justifies it


## Handoff Contract

Before handing off:

- `docs/design/architecture.md` and `docs/design/unified-codec-lm.md` reflect the stable contract
- `docs/design/onnx-contract.md` and `docs/design/rust-engine-design.md` reflect the same contract
- `tmrvc-core` exposes the shared serializable schema used by all runtimes
- `uclm_model.py` has stable function names and includes the Prosody Predictor interface
- output dict keys are fixed
- worker 02 and worker 04 can code against the interfaces without guessing


## Required Tests

- shape tests for pointer outputs
- backward compatibility test for loading a v2 checkpoint with `strict=False`
- model smoke test for `forward_tts_pointer`
- test for `predict_prosody` output shape and determinism (given same inputs/seed)
- shape test for explicit / delta / ssl voice-state conditioning inputs
- shape test for dialogue-context and prosody-conditioning inputs
- state-transition contract test for teacher-forced versus autoregressive loop compatibility
- contract test that exported `state_cond` wrapper matches the canonical raw-state API
