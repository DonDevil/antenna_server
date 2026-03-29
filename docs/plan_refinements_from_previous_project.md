# Plan Refinements From Previous Project

This document updates the server plan using the earlier monolithic CST optimization project as evidence.

## What Should Be Carried Forward

### 1. Keep the inverse-model boundary explicit

The previous project had a useful split between model loading, bounded parameter prediction, CST adaptation, and simulation feedback.

Keep these boundaries in the new server:

- `ann_adapter.py`: loads the inverse ANN and exposes prediction only
- `validators.py`: enforces training-domain and physical bounds before any client command package is produced
- `command planner`: converts validated dimensions into a deterministic CST command package

This is stronger than the old design because the server never directly mixes ANN output with CST execution code.

### 2. Preserve bounded prediction before execution

The old `ParameterEngine` correctly clamps predicted dimensions before CST use. That should stay, but the clamp should become a validated contract rather than silent correction.

Update for the new server:

- store training-domain ranges in config and context files
- reject or downgrade confidence when the request falls outside ANN training support
- record all clamp or correction actions in the session manifest

### 3. Preserve feedback logging, but make it auditable

The feedback CSV from the old project was useful, but too weak for reproducibility.

Replace it with session artifacts containing:

- normalized user intent
- LLM parse result
- ANN input and output
- command package checksum
- client feedback payload
- iteration evaluation decision
- model versions and prompt bundle version

## What Must Change

### 4. Move all CST-specific execution out of the server

The previous project mixed CST SDK calls, macro generation, and orchestration in the same code path. That is the main architectural problem to avoid.

New rule:

- the server never imports CST SDK modules
- the server never emits raw VBA text from the LLM
- the server emits only validated high-level commands
- the client is the only place that translates commands into CST VBA or CST API calls

This change is the core reason the new project will be easier to test and safer against hallucination.

### 5. Add a forward-validation stage after inverse ANN prediction

The old project had forward and inverse models, but the inverse prediction was not sanity-checked by a forward surrogate before simulation.

Improve the plan:

- after inverse ANN predicts dimensions, optionally run a forward surrogate check
- compare expected metrics against the requested target
- derive a confidence score from residual error before generating commands
- route low-confidence cases to clarification or best-effort mode depending on policy

This should sit inside `central_brain.py` or a dedicated `surrogate_validator.py` later.

### 6. Replace silent heuristics with explicit state transitions

The old iterative loop used reasonable heuristics for adjusting patch length, feed width, and substrate height, but the logic was buried inside a procedural loop.

Improve the plan:

- define explicit refinement strategies in the optimization state machine
- log why each refinement was chosen
- record whether the adjustment came from ANN, heuristic correction, or rule-based fallback
- add rollback when a refinement worsens the objective

### 7. Version everything that affects a result

The previous project had model files and feedback logs, but weak provenance.

The new server should version:

- ANN checkpoint and metadata
- prompt/context bundle
- command catalog
- acceptance policy
- client capability profile

Each session manifest should be sufficient to replay the exact decision path.

### 8. Replace hard-coded paths with environment-backed config

The old project used machine-specific absolute paths for CST files. That should not exist in the new repository.

Improve the plan:

- all paths resolved from `config.py`
- environment overrides for models, logs, sessions, and exports
- no platform-specific path literals inside domain code

### 9. Separate online learning from serving

The old project retrained correction logic in the same workflow as optimization. For the new server, serving and retraining should be different concerns.

For v1:

- the server may log feedback suitable for retraining
- the server should not retrain models during request handling
- retraining becomes an offline experiment pipeline later

This improves determinism and keeps request latency stable.

## Concrete Plan Changes

The server plan should now include these additions:

1. Add `surrogate_validator.py` or equivalent forward-check logic after ANN inference.
2. Add `family_registry.py` later so antenna family support is configuration-driven instead of hard-coded.
3. Add `artifact_manifest.json` per session as the canonical reproducibility record.
4. Add explicit `stop_reason` and `decision_reason` fields to each iteration result.
5. Treat online retraining as out of scope for the server runtime and keep only feedback capture in v1.

## Git and Artifact Policy

Based on the previous project, the new repository should not commit:

- runtime logs
- session outputs
- local CST exports
- local environment files
- temporary files
- large model checkpoints unless explicitly managed with Git LFS

This policy is enforced by the repository `.gitignore`.