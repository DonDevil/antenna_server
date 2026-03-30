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

## Next Implementation Plan: Dynamic Action Planner With Rule Book

This section defines the next execution plan for adding a robust CST automation strategy that combines:

- user intent parsing via Ollama `deepseek-r1:8b`
- ANN-based inverse prediction as baseline geometry
- deterministic command/action execution from a strict catalog
- rule-book-guided refinement after simulation feedback

### Architecture Goal

Do not generate raw VBA dynamically from LLM output.

Instead:

1. LLM selects from whitelisted actions only.
2. Deterministic server code validates action parameters and ordering.
3. Client-side CST adapter converts validated actions into stable template-based VBA.

This preserves flexibility while minimizing CST fragility.

## Proposed File Structure Changes

Add the following modules while keeping current behavior as fallback:

```text
app/
  llm/
	 ollama_client.py                # low-level Ollama calls, retries, timeouts
	 intent_parser.py                # prompt -> structured intent extraction
	 action_ranker.py                # feedback signature -> ranked candidate actions
	 session_context_builder.py      # compact context for LLM calls
  planning/
	 action_catalog.py               # typed action registry + validation helpers
	 action_rules.py                 # deterministic rule-book loader and scorer
	 dynamic_planner.py              # build/validate ordered action plan
	 command_compiler.py             # action plan -> command_package entries
  core/
	 feedback_features.py            # derive error signature from CST feedback
	 policy_runtime.py               # call-budget and policy decisions for LLM usage
context_files/
  rule_book/
	 s11_refinement_rules.v1.yaml    # detailed engineering rules by failure pattern
	 action_effect_priors.v1.yaml    # expected directionality and confidence priors
schemas/
  planning/
	 action_plan.v1.json             # strict schema for planner output
	 action_catalog.v1.json          # schema for action catalog definition
```

Keep existing `app/commands/planner.py` as `fixed` planner mode until dynamic mode is proven stable.

## Phase Plan

### Phase 1: Contracts and Catalog Foundations

1. Add `action_catalog.v1` with action definitions, parameter types, bounds, units, prerequisites, and incompatibilities.
2. Add `action_plan.v1` schema with ordered actions, rationale tags, and expected effect metadata.
3. Add command compiler that maps actions to existing high-level command package structure.
4. Introduce planner mode flag in runtime config:
	- `fixed` (default)
	- `dynamic` (feature-gated)

Exit criteria:

- Dynamic plan can be validated and compiled without any LLM call.
- Existing integration tests remain green in `fixed` mode.

### Phase 2: Rule Book and Deterministic Scoring

1. Implement rule book file format (`s11_refinement_rules.v1.yaml`) with:
	- trigger conditions over feedback features
	- ranked candidate actions
	- parameter update formulas
	- guardrails and hard-block constraints
2. Add feature extraction from feedback (`feedback_features.py`):
	- center-frequency error
	- bandwidth shortfall
	- S11 dip depth/shape proxies
	- VSWR and gain failures
3. Add deterministic scorer that produces top-k candidate actions with confidence.

Exit criteria:

- Given feedback, server produces reproducible ranked actions with no LLM.
- Rule evaluation is logged with `decision_reason` and per-rule provenance.

### Phase 3: DeepSeek Intent and Action Selection (Constrained)

1. Add `intent_parser.py` for user request normalization and clarification prompts.
2. Add `action_ranker.py` that consumes compact context + rule candidates and returns ranked choice from catalog only.
3. Enforce strict output schema validation and fallback:
	- if invalid LLM output, fallback to deterministic top-ranked rule action
4. Add policy switches:
	- `llm_enabled_for_intent`
	- `llm_enabled_for_refinement`

Exit criteria:

- LLM never emits VBA.
- LLM cannot introduce unknown actions.
- Invalid LLM output cannot break planning pipeline.

### Phase 4: Session Context and Resource Optimization

1. Add compact session memory snapshots for LLM use:
	- original intent summary
	- latest target and constraints
	- last N iteration outcomes (small N, e.g., 2 or 3)
	- current rule-book candidate shortlist
2. Add context builder that creates short prompt context from session state instead of full history.
3. Add call budget controls per session:
	- max LLM calls per optimize request
	- max LLM calls per refinement loop
	- timeout and token/output limits
4. Add invocation policy to run DeepSeek only at key points:
	- initial intent parse
	- refinement decision only when deterministic confidence is below threshold

Exit criteria:

- Average LLM calls per session is bounded and observable.
- ANN and planner continue to run when LLM is unavailable.

### Phase 5: Observability, Replay, and Safety Hardening

1. Persist planner provenance in session manifest:
	- planner mode (`fixed` or `dynamic`)
	- rule ids considered
	- LLM used or bypassed
	- action selection confidence and fallback reason
2. Extend replay script to reconstruct action selection timeline.
3. Add safety checks for every action plan:
	- precondition failures
	- geometry sanity bounds
	- prohibited sequence patterns
4. Add integration tests for failure paths:
	- malformed LLM output
	- out-of-budget call behavior
	- deterministic fallback correctness

Exit criteria:

- Replay can explain exactly why each refinement action was selected.
- Safety gate blocks invalid action plans before CST client execution.

### Phase 6: Data Collection and Optional Custom Policy Model

1. Log per-iteration tuples for learning:
	- feedback signature
	- candidates proposed
	- action chosen
	- post-action improvement metrics
2. Train an offline lightweight action policy model (for example, gradient boosting).
3. Introduce optional `policy_model` ranker in front of LLM for cheaper inference.
4. Use DeepSeek as fallback/tie-breaker, not primary path, once model quality is sufficient.

Exit criteria:

- Measured reduction in LLM usage and latency without quality regression.

## Runtime Resource Policy (DeepSeek-R1:8b + ANN)

Use the following serving policy to minimize resource waste:

1. Keep ANN loaded in-process (small, frequent calls).
2. Use DeepSeek only for:
	- initial intent parsing
	- low-confidence refinement decisions
3. Cache session-level LLM outputs:
	- parsed intent
	- clarified constraints
	- last accepted action rationale
4. Build short prompts from structured context only; do not pass full transcript/history.
5. Apply strict latency guardrails:
	- per-call timeout
	- bounded output tokens
	- deterministic fallback on timeout
6. Do not call LLM if deterministic rule confidence exceeds threshold.
7. Track metrics:
	- LLM calls/session
	- timeout rate
	- fallback rate
	- median optimize latency

This keeps DeepSeek available where it adds value while ensuring the pipeline remains deterministic and operational under load or model unavailability.

## Implementation Status Update

The following phases are now implemented in code with deterministic fallback enabled by default:

- Phase 2: Rule-book scoring and feedback feature extraction.
- Phase 3: Constrained LLM action ranking modules (action selection only, no VBA generation).
- Phase 4: Session context compaction and runtime call-budget policy for LLM usage.
- Phase 5: Planning provenance persisted into session manifest and replay output.

Current runtime mode remains conservative:

- Fixed command planner remains active by default.
- Rule-based refinement runs in the feedback loop.
- LLM refinement calls are policy-gated and disabled by default unless explicitly enabled in planner settings.