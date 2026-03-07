# Worker 05: Tooling, Menus, Config Flow, and Documentation

## Scope

Make v3 pointer mode the visible default in developer tooling and document v2/MFA as legacy.


## Primary Files

- `dev.py`
- `README.md`
- `TRAIN_GUIDE.md`
- `docs/README.md`
- `docs/training/README.md`
- optional new docs under `docs/design/`


## Required Outcomes

- `dev.py` clearly separates `v2 legacy` and `v3 pointer`
- docs stop presenting MFA as the default path
- config/setup docs reflect the new training contracts


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


## Guardrails

- do not remove legacy commands abruptly if they are still useful for comparison
- do not leave ambiguous wording like “recommended” on MFA flow
- do not claim v3 is complete until worker 06 signs off


## Handoff Contract

- a new contributor can choose the correct v3 flow without reading code
- menus and docs agree on naming
- all mentions of MFA default flow are removed or marked legacy


## Required Tests

- `dev.py` menu tests for new labels and dependency hints
- doc sanity review for internal consistency
