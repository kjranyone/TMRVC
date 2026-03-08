# Migration Guide: UCLM v2 (Duration-Based) to v3 (Pointer-Based)

## Overview

UCLM v3 replaces the duration-based alignment mechanism with a pointer-based one.
In v2, a `DurationPredictor` module produced per-phoneme durations that were
expanded into mel frames. In v3, a `PointerHead` directly attends to phoneme
positions at each decoder step, deciding *which* phoneme to read and *when* to
advance. This removes the need for externally supplied duration labels and
enables more natural, expressive synthesis.

This guide is an overview only. Canonical v3 contracts live in:

- `docs/design/architecture.md`
- `docs/design/unified-codec-lm.md`
- `docs/design/onnx-contract.md`

Key wins:

- No dependency on forced-alignment durations at training time.
- Runtime control over pacing, pauses, and breathing.
- Optional expressive conditioning via dialogue context, acting intent,
  `local_prosody_latent`, and explicit physical control.

---

## Model Changes

### Removed / Legacy

- `forward_tts()` is removed. All TTS inference now goes through
  `forward_tts_pointer()`.
- `DurationPredictor` remains in the codebase for checkpoint loading but is
  never called during v3 training or inference.

### New Modules

| Module | Purpose |
|---|---|
| `PointerHead` | Soft monotonic attention that produces a pointer distribution over encoder positions at each decoder step. |
| `PointerState` | Tracks cumulative attention, progress, and boundary decisions across steps. |
| `DialogueContextProjector` | Projects external dialogue/scene embeddings into the decoder's conditioning space for expressive synthesis. |

### New Optional Inputs to `forward_tts_pointer()`

```python
forward_tts_pointer(
    ...,
    dialogue_context=None,   # (B, C_ctx, d_model) or (B, d_model)
    acting_intent=None,      # (B, num_intents) soft intent vector
    local_prosody_latent=None,  # (B, d_prosody) or (B, T, D_prosody)
    explicit_voice_state=None,  # (B, T, 8) or (B, 8)
    delta_voice_state=None,     # (B, T, 8) or (B, 8)
    ssl_voice_state=None,       # optional SSL-derived state evidence
)
```

All three are optional; the model falls back to neutral conditioning when they
are absent.

---

## Training Changes

### Duration Labels No Longer Required

`durations.npy` files are not needed for v3 training. If present, they are
ignored unless you explicitly select legacy mode.

### Default TTS Mode

`--tts-mode pointer` is now the default. To revert to duration-based training,
pass `--tts-mode legacy_duration`.

### New CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--pointer-loss-weight` | `1.0` | Weight of the pointer attention loss. |
| `--progress-loss-weight` | `0.1` | Weight of the monotonic progress loss. |
| `--alignment-loss-type` | `"forward"` | Alignment loss variant (`forward`, `ctc`, `diagonal`). |

### New Batch Fields

The dataloader may now yield additional fields:

- `dialogue_context` -- bounded multi-turn text-context tensor.
- `acting_intent` -- intent label or soft vector.
- `local_prosody_latent` -- optional prosody control latent.
- `voice_state_targets` / `voice_state_observed_mask` /
  `voice_state_confidence` -- optional physical-control supervision bundle.

These fields are optional; batches without them simply skip the corresponding
loss terms.

### Anti-Collapse Diversity Loss

v3 adds a diversity regulariser that penalises the pointer distribution when it
collapses to a single phoneme for too many consecutive frames. This is enabled
by default and controlled by the existing `--pointer-loss-weight` flag (it
scales proportionally).

---

## Dataset Changes

### `tts_mode` Parameter

When constructing a dataset, the `tts_mode` parameter selects the alignment
strategy:

| Value | Behaviour |
|---|---|
| `auto` (default) | Uses pointer mode; falls back to durations if pointer head is absent in checkpoint. |
| `pointer` | Pointer-only. Errors if the model lacks a pointer head. |
| `legacy_duration` | Duration-based. Requires `durations.npy`. |

### Optional Expressive Data Files

Place these alongside your existing data to enable expressive conditioning:

- `dialogue_context.npy` -- per-utterance dialogue embedding.
- `acting_intent.json` -- per-utterance intent labels.
- `prosody_targets.npy` -- per-frame prosody features.

None of these are required; the pipeline works without them.

### Diagnostic Methods

- `supervision_report()` -- summarises which supervision signals (durations,
  pointers, prosody) are available for the dataset.
- `expressive_readiness_report()` -- checks whether the optional expressive
  files are present and correctly shaped.

---

## Serving Changes

### Unified TTS Endpoint

`tts_pointer()` has been merged into the existing `tts()` method. There is no
separate `tts_mode` parameter on the engine; the engine inspects the loaded
checkpoint and uses pointer inference automatically when the pointer head is
present.

### New TTS API Parameters

```python
engine.tts(
    text="...",
    speaker_id=0,
    pace=1.0,            # global speed multiplier (0.5 = half speed)
    hold_bias=0.0,       # bias toward holding the current phoneme longer
    boundary_bias=0.0,   # bias toward advancing at word/phrase boundaries
    phrase_pressure=0.0,  # compresses or expands phrase-level timing
    breath_tendency=0.0,  # likelihood of inserting breath pauses
)
```

All new parameters default to neutral values and are ignored for legacy
duration checkpoints.

### Scene State

The engine exposes a `scene_state_available` property that returns `True` when
the loaded model includes a `DialogueContextProjector` and the engine has been
supplied with a dialogue context embedding. Use this to decide whether to pass
expressive conditioning at call time.

---

## Checkpoint Compatibility

v2 checkpoints are loadable into a v3 model:

```python
model.load_state_dict(torch.load("v2_checkpoint.pt"), strict=False)
```

- `strict=False` is required because the v3 model contains new keys that are
  absent in the v2 checkpoint.
- `PointerHead` weights will be randomly initialised. Fine-tuning on a small
  amount of data is recommended before serving.
- `DialogueContextProjector` weights will also be randomly initialised. If you
  do not plan to use expressive conditioning, these weights remain inert and
  do not affect output quality.
- The legacy `DurationPredictor` weights in the v2 checkpoint are loaded
  successfully but are unused during v3 inference.
