# Track: v4 Control Plane Cutover

## Scope

This track owns the GUI/control-plane cutover for `v4`.
This is not a request to keep polishing the current `8-D` workshop.

The critical-path UI slice is:

- canonical backend enrollment
- basic physical controls
- advanced physical controls
- acting prompt and acting macro controls
- trajectory inspection, patch, replay, and transfer

## Primary Files

- `tmrvc-gui/src/tmrvc_gui/gradio_app.py`
- `tmrvc-gui/src/tmrvc_gui/gradio_state.py`
- `tmrvc-gui/tests/test_gradio_app.py`
- `docs/design/gui-design.md`
- `docs/design/v4-master-plan.md`

## Open Tasks

### 1. Replace the old `8-D` workshop surface

The new `v4` Workshop must expose at least:

- basic physical panel
- advanced physical panel
- acting prompt panel
- acting macro panel
- reference-driven panel
- trajectory panel

Rules:

- raw latent axes must not be shown as a default control surface
- the UI must not present the `v3` `8-D` vector as the active mainline model

### 2. Keep canonical backend enrollment only

Required behavior:

- upload reference audio
- call backend encode/persist route
- save/load canonical speaker profiles
- expose backend errors and progress explicitly

No dummy embedding path is claim-valid.

### 3. Add compile and trajectory panels

Required UI outputs:

- compile warnings
- compile summary
- generated `trajectory_id`
- schema version
- provenance label

Required trajectory actions:

- inspect compiled controls
- inspect realized physical/acting traces
- patch a local region
- replay the edited artifact

### 4. Add transfer as a first-class UI flow

Required actions:

- replay the same trajectory on the same speaker
- replay it on another speaker profile
- explicitly label transfer results

The UI must label each result as:

- fresh compile
- deterministic replay
- cross-speaker transfer
- patched replay

### 5. Add structured evaluation review modes

When the validation track enables the protocols, the UI must support review flows for:

- bootstrap QC review
- replay fidelity review
- edit locality review
- transfer quality review

## Required Tests

- backend enrollment flow test
- compile -> edit -> replay flow test
- cross-speaker transfer flow test
- UI labeling test for compile vs replay vs transfer vs patched replay
- basic/advanced panel serialization test

## Out Of Scope

Do not reopen:

- preserving the current `v3` slider layout as the mainline UX
- generic Gradio shell creation
- old local dummy enrollment shortcuts

## Exit Criteria

- no dummy enrollment path remains in the claim-valid flow (verified by code search for dummy/mock embedding paths)
- the Workshop presents the `v4` physical-plus-latent control model with basic (6 controls) and advanced (12 controls) panels
- trajectory patch/replay/transfer are accessible UI actions with distinct result labels (fresh compile / deterministic replay / cross-speaker transfer / patched replay)
- result provenance is visible to the user as a label on every generated output
- acting macro panel exposes at least 4 macro controls (intensity, instability, tenderness, tension)
