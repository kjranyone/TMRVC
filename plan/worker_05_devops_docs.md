# Worker 05: Tooling, Menus, Config Flow, and Documentation

## Scope

Make v3 pointer mode the visible default in developer tooling, document v2/MFA as legacy, and ensure operator-facing documentation covers the WebUI-first workflow defined by Worker 12.


## Primary Files

- `dev.py`
- `README.md`
- `TRAIN_GUIDE.md`
- `docs/README.md`
- `docs/training/README.md`
- `docs/design/speaker-profile-spec.md`
- optional new docs under `docs/design/`


## Required Outcomes

- `dev.py` clearly separates `v2 legacy` and `v3 pointer`
- docs stop presenting MFA as the default path
- config/setup docs reflect the new training contracts
- operator guides document the WebUI-first workflow for ingest, curation, export, audition, and evaluation
- `SpeakerProfile` / Casting Gallery usage is documented for operators
- external baseline evaluation setup procedure is documented
- `voice_state` supervision flow from curation through training is documented for operators


## Concrete Tasks

1. Rework `dev.py` menu labels:
   - v3 pointer full training
   - v3 pointer train from cache
   - v2 legacy MFA flow
2. Add dependency map entries for v3 mode.
3. Ensure preflight checks in `dev.py` align with v3:
   - no hard requirement for MFA
   - no hard requirement for `durations.npy`
4. Rewrite README training section:
   - v3 default flow first
   - v2 legacy flow later
5. Update `TRAIN_GUIDE.md`:
   - explain pointer mode
   - explain that MFA is optional legacy tooling
6. Add migration notes for existing users and existing experiments.
7. Document WebUI-first operator workflow:
   - dataset upload and legality assignment via browser
   - curation launch, monitoring, and resume via browser
   - export and artifact download via browser
   - drama workshop audition and take management via browser
   - blind evaluation session setup via browser
   - clarify that `dev.py` and CLI remain available for developers but are not the mainline human operator path
8. Document `SpeakerProfile` / Casting Gallery operator guide:
   - how to create, save, and load speaker profiles from the WebUI
   - how `speaker_profile_id` is used in TTS API requests
   - cache invalidation behavior when the prompt encoder changes
9. Document external baseline evaluation setup:
   - how to freeze a baseline entry in `docs/design/external-baseline-registry.md`
   - how to configure the evaluation arena for blind A/B comparison
   - how to reproduce baseline inference with pinned settings
10. Document `voice_state` supervision operator flow:
    - how pseudo-labels are generated during curation (Stage 6)
    - how masks and confidences are exported (Worker 10)
    - how the trainer consumes partial supervision (Worker 02)
    - how controllability uplift is validated (Worker 06 / Worker 11)
11. Add `dev.py` curation entrypoints:
    - run curation
    - resume curation
    - export promoted subset
    - show curation summary
    - these must call through the authoritative backend API, not bypass it


## Guardrails

- do not remove legacy commands abruptly if they are still useful for comparison
- do not leave ambiguous wording like “recommended” on MFA flow
- do not claim v3 is complete until worker 06 signs off
- do not document CLI-only workflows for operations that the WebUI is required to support
- do not omit WebUI setup instructions from the operator guide


## Handoff Contract

- a new contributor can choose the correct v3 flow without reading code
- a non-technical operator can follow the WebUI guide to complete ingest, curation, export, and evaluation without CLI
- menus and docs agree on naming
- all mentions of MFA default flow are removed or marked legacy


## Required Tests

- `dev.py` menu tests for new labels and dependency hints
- `dev.py` curation entrypoint smoke test
- doc sanity review for internal consistency
- doc coverage check: every Worker 12 human workflow step has a corresponding operator guide section
