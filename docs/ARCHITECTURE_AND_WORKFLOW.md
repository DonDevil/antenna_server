# Architecture and Workflow (Current Repository State)

This document is based on the repository as it exists now. It is meant to be a code-backed architecture reference, not an aspirational summary.

## 1) Project Purpose

`antenna_server` is a FastAPI backend that orchestrates antenna design sessions. It does not run CST itself. Instead it:

1. accepts structured antenna requirements,
2. normalizes them against antenna-family rules,
3. generates a deterministic starting recipe,
4. blends that recipe with ANN predictions when trained artifacts exist,
5. safety-checks the result with a lightweight surrogate,
6. returns a validated CST command package for the client to execute locally,
7. ingests measured CST feedback,
8. decides whether the session is accepted, refined, or stopped,
9. stores feedback for future retraining.

### Server/client boundary

- The server owns validation, orchestration, ANN inference, rule-based and LLM-assisted planning, session state, and dataset/retraining workflows.
- The client owns CST execution, export of S-parameter and far-field artifacts, UI, and submission of feedback.

## 2) Current-Code Facts That Matter

- All public endpoints are under `/api/v1/*`.
- The optimize, feedback, and session APIs are versioned `v1`, but the emitted CST command package is `cst_command_package.v2`.
- The initial command package is always built by the fixed planner in `app/commands/planner.py`.
- The dynamic planning logic is used after feedback, when the system selects the next refinement action.
- `amc_patch` is not backed by a server-side ANN model right now. The server returns baseline patch geometry and tells the client to create AMC geometry locally with `implement_amc`.
- AMC feedback is stored and validated, but automatic AMC live retraining is intentionally not triggered.
- Chat remembers captured requirements by `session_id` and can rehydrate them from saved sessions.
- The WebSocket schema supports more event types than the server currently emits. The current implementation sends `iteration.completed`, `session.completed`, and `session.failed`.

## 3) Repository Scope

Architecturally relevant files are listed below. The workspace also contains local environment and cache directories such as `.git/`, `.venv/`, `env/`, `__pycache__/`, and `.pytest_cache/`; those are real directories in the workspace but are not part of the application architecture.

Current artifact counts in the workspace:

- `sessions/`: 389 UUID-named JSON files
- `data/raw/`: 8 files
- `data/validated/`: 6 files
- `data/rejected/`: 4 files
- `tests/integration/`: 7 test files
- `tests/unit/`: 15 test files

## 4) Exact Repository Structure

```text
antenna_server/
├── .gitignore
├── central_brain.py
├── config.py
├── hanoff.md
├── main.py
├── requirements.txt
├── server.py
├── app/
│   ├── __init__.py
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── baseline.py
│   │   ├── family_ann_trainer.py
│   │   ├── features.py
│   │   ├── live_retraining.py
│   │   ├── model.py
│   │   ├── predictor.py
│   │   ├── rect_patch_evaluator.py
│   │   ├── rect_patch_trainer.py
│   │   └── trainer.py
│   ├── antenna/
│   │   ├── __init__.py
│   │   ├── materials.py
│   │   └── recipes.py
│   ├── api/
│   │   └── (currently empty; reserved for future route modularization)
│   ├── commands/
│   │   ├── __init__.py
│   │   └── planner.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── capabilities_catalog.py
│   │   ├── exceptions.py
│   │   ├── family_registry.py
│   │   ├── feedback_features.py
│   │   ├── json_contracts.py
│   │   ├── objectives.py
│   │   ├── policy_runtime.py
│   │   ├── refinement.py
│   │   ├── schemas.py
│   │   ├── session_store.py
│   │   ├── surrogate_validator.py
│   │   └── test_command.txt
│   ├── data/
│   │   ├── __init__.py
│   │   ├── family_dataset_generators.py
│   │   ├── family_feedback.py
│   │   ├── rect_patch_feedback.py
│   │   ├── rect_patch_feedback_logger.py
│   │   ├── schema.py
│   │   ├── store.py
│   │   └── validator.py
│   ├── llm/
│   │   ├── action_ranker.py
│   │   ├── intent_parser.py
│   │   ├── ollama_client.py
│   │   └── session_context_builder.py
│   └── planning/
│       ├── action_catalog.py
│       ├── action_rules.py
│       ├── command_compiler.py
│       ├── dynamic_planner.py
│       ├── geometry_guardrails.py
│       └── v2_command_contract.py
├── context_files/
│   ├── antenna_design_bounds.md
│   ├── command_catalog.md
│   ├── examples_e2e.md
│   ├── feedback_examples.md
│   ├── optimization_guide.md
│   ├── safety_guardrails.md
│   ├── system_prompt.md
│   ├── capabilities/
│   │   └── antenna_capabilities.v1.json
│   └── rule_book/
│       ├── action_effect_priors.v1.json
│       └── s11_refinement_rules.v1.json
├── data/
│   ├── raw/
│   │   ├── amc_patch_feedback_v1.csv
│   │   ├── dataset.csv
│   │   ├── live_results_v1.jsonl
│   │   ├── rect_patch_feedback_v1.csv
│   │   ├── rect_patch_formula_plus_client_v1.csv
│   │   ├── rect_patch_formula_synth_v1.csv
│   │   ├── wban_patch_feedback_v1.csv
│   │   └── wban_patch_formula_synth_v1.csv
│   ├── rejected/
│   │   ├── amc_patch_feedback_rejected_v1.csv
│   │   ├── dataset_rejected.csv
│   │   ├── rect_patch_feedback_rejected_v1.csv
│   │   └── wban_patch_feedback_rejected_v1.csv
│   └── validated/
│       ├── amc_patch_feedback_validated_v1.csv
│       ├── dataset_validated.csv
│       ├── rect_patch_feedback_validated_v1.csv
│       ├── rect_patch_forward_train_v1.csv
│       ├── rect_patch_inverse_train_v1.csv
│       └── wban_patch_feedback_validated_v1.csv
├── docs/
│   ├── ARCHITECTURE_AND_WORKFLOW.md
│   ├── CLIENT_SIDE_ANN_AND_LIVE_RETRAIN_HANDOFF.md
│   ├── COMPLETE_PROJECT_WORKING_AND_FILE_STRUCTURE.md
│   ├── INSTALLATION_AND_USAGE.md
│   ├── PROJECT_REFERENCE_FOR_COPILOT.md
│   ├── amc_rules_server.md
│   ├── rect_patch_rules_server.md
│   └── wban_rules_server.md
├── logs/
│   └── antenna_client.log
├── models/
│   └── ann/
│       ├── rect_patch_v1/
│       │   ├── inverse_ann.pt
│       │   ├── live_retrain_state.json
│       │   └── metadata.json
│       ├── v1/
│       │   ├── inverse_ann.pt
│       │   └── metadata.json
│       └── wban_patch_v1/
│           ├── inverse_ann.pt
│           └── metadata.json
├── schemas/
│   ├── commands/
│   │   ├── cst_command_package.v1.json
│   │   ├── cst_command_package.v2.command_contract.json
│   │   └── cst_command_package.v2.json
│   ├── data/
│   │   ├── amc_patch_feedback.v1.json
│   │   ├── rect_patch_feedback.v1.json
│   │   └── wban_patch_feedback.v1.json
│   ├── http/
│   │   ├── client_feedback.v1.json
│   │   ├── optimize_request.v1.json
│   │   └── optimize_response.v1.json
│   ├── planning/
│   │   ├── action_catalog.v1.json
│   │   └── action_plan.v1.json
│   └── ws/
│       └── session_event.v1.json
├── scripts/
│   ├── derive_rect_patch_datasets.py
│   ├── evaluate_rect_patch_ann.py
│   ├── generate_family_ann_datasets.py
│   ├── generate_synthetic_dataset.py
│   ├── generate_sysnthetic_dataset.py
│   ├── replay_session.py
│   ├── run_rect_patch_pipeline.py
│   ├── train_ann.py
│   ├── train_family_anns.py
│   ├── train_rect_patch_ann.py
│   ├── validate_dataset.py
│   └── validate_rect_patch_feedback.py
├── sessions/
│   └── 389 persisted session snapshots named <uuid>.json
├── tests/
│   ├── conftest.py
│   ├── integration/
│   │   ├── test_capabilities_chat.py
│   │   ├── test_family_registry.py
│   │   ├── test_iteration_flow.py
│   │   ├── test_llm_live.py
│   │   ├── test_result_ingest.py
│   │   ├── test_surrogate_policy.py
│   │   └── test_ws_stream.py
│   └── unit/
│       ├── test_ann_recipe_predictor.py
│       ├── test_dataset_validator.py
│       ├── test_family_ann_trainer.py
│       ├── test_family_dataset_generators.py
│       ├── test_family_feedback_workflow.py
│       ├── test_family_geometry_commands.py
│       ├── test_feedback_objectives.py
│       ├── test_geometry_guardrails.py
│       ├── test_intent_parser_llm.py
│       ├── test_live_retraining.py
│       ├── test_planning_phase1.py
│       ├── test_recipe_pipeline.py
│       ├── test_rect_patch_feedback_workflow.py
│       ├── test_refinement_geometry_corrections.py
│       └── test_rule_based_refinement.py
└── web/
        └── index.html
```

## 5) Purpose of Each File

### Top-level files

| File | Purpose |
| --- | --- |
| `.gitignore` | Ignores local environments, caches, and generated artifacts from version control. |
| `main.py` | Minimal entrypoint that starts the FastAPI app with Uvicorn. |
| `server.py` | Main API layer: app creation, CORS, warmup, health/capabilities, chat, optimize, feedback, session lookup, and WebSocket streaming. |
| `central_brain.py` | Orchestration core for optimize and feedback flows. |
| `config.py` | Central settings for paths, model locations, planner behavior, and numeric bounds. |
| `requirements.txt` | Direct Python dependencies used by the repository. |
| `hanoff.md` | Detailed handoff notes for the rectangular-patch ANN data/model workflow. |

### `app/ann/`

| File | Purpose |
| --- | --- |
| `app/ann/__init__.py` | Marks `app.ann` as a package. |
| `app/ann/baseline.py` | Deterministic microstrip-style fallback dimensions for cold starts. |
| `app/ann/family_ann_trainer.py` | Generic family ANN training and evaluation framework for microstrip and WBAN family models. |
| `app/ann/features.py` | Builds model input features from request targets, priorities, materials, and recipe-derived family parameters. |
| `app/ann/live_retraining.py` | Ingests feedback, updates family datasets, writes the results ledger, and triggers asynchronous retraining when thresholds are met. |
| `app/ann/model.py` | Defines the feed-forward PyTorch regressor used by the inverse ANN models. |
| `app/ann/predictor.py` | Selects artifacts, loads checkpoints and metadata, generates recipes, runs inference, blends outputs, and handles AMC as a client-side special case. |
| `app/ann/rect_patch_evaluator.py` | Compares the rectangular-patch ANN against deterministic recipe outputs on validated feedback rows. |
| `app/ann/rect_patch_trainer.py` | Trains the rectangular microstrip inverse ANN and writes its checkpoint and metadata. |
| `app/ann/trainer.py` | Trains the legacy generic ANN against the older mixed-family validated dataset. |

### `app/antenna/`

| File | Purpose |
| --- | --- |
| `app/antenna/__init__.py` | Marks `app.antenna` as a package. |
| `app/antenna/materials.py` | Built-in conductor and substrate property library used by recipes and command planning. |
| `app/antenna/recipes.py` | Deterministic geometry generators for rectangular, circular, AMC-backed, and WBAN patch starting points. |

### `app/api/`

| Path | Purpose |
| --- | --- |
| `app/api/` | Empty placeholder directory reserved for future API refactoring or route modularization. |

### `app/commands/`

| File | Purpose |
| --- | --- |
| `app/commands/__init__.py` | Marks `app.commands` as a package. |
| `app/commands/planner.py` | Builds the fixed action plan and initial or follow-up CST command package from a request and ANN prediction. |

### `app/core/`

| File | Purpose |
| --- | --- |
| `app/core/__init__.py` | Marks `app.core` as a package. |
| `app/core/capabilities_catalog.py` | Loads the public capabilities catalog, with built-in fallbacks if the JSON file is missing or invalid. |
| `app/core/exceptions.py` | Custom exceptions for invalid families, constraint violations, missing models, and similar domain errors. |
| `app/core/family_registry.py` | Declares supported families and validates that selected materials and substrates are allowed for each family. |
| `app/core/feedback_features.py` | Converts measured CST feedback into normalized refinement features such as frequency error, bandwidth shortfall, and severity. |
| `app/core/json_contracts.py` | Central JSON Schema validator wrapper used at API and planning boundaries. |
| `app/core/objectives.py` | Builds the initial objective state and re-evaluates primary and secondary objectives after feedback. |
| `app/core/policy_runtime.py` | Tracks whether LLM refinement is allowed and enforces per-session and per-iteration LLM budgets. |
| `app/core/refinement.py` | Applies acceptance criteria and refines geometry with heuristics or rule and LLM-selected strategies. |
| `app/core/schemas.py` | Pydantic models for optimize requests and responses, ANN predictions, request constraints, objectives, and client capabilities. |
| `app/core/session_store.py` | Persists session JSON files, computes payload checksums, and initializes the artifact manifest and history. |
| `app/core/surrogate_validator.py` | Lightweight heuristic forward surrogate that gates unsafe ANN outputs before command generation. |
| `app/core/test_command.txt` | Small checked-in test artifact used during command-related experiments. |

### `app/data/`

| File | Purpose |
| --- | --- |
| `app/data/__init__.py` | Marks `app.data` as a package. |
| `app/data/family_dataset_generators.py` | Generates synthetic formula-based datasets for microstrip, AMC, and WBAN family training. |
| `app/data/family_feedback.py` | Validates, appends, and materializes AMC and WBAN feedback datasets into raw, validated, and rejected CSVs. |
| `app/data/rect_patch_feedback.py` | Validates rectangular-patch feedback and derives inverse and forward training CSVs. |
| `app/data/rect_patch_feedback_logger.py` | Appends one schema-compliant rectangular-patch feedback row to the raw feedback CSV. |
| `app/data/schema.py` | Declares the required columns for the legacy mixed-family training dataset. |
| `app/data/store.py` | Reads raw CSV data and writes validated and rejected outputs after validation. |
| `app/data/validator.py` | Range-checks numeric dataset columns against configured bounds. |

### `app/llm/`

| File | Purpose |
| --- | --- |
| `app/llm/action_ranker.py` | Calls the LLM to choose among top refinement candidates when the deterministic rule confidence is not strong enough. |
| `app/llm/intent_parser.py` | Extracts frequency, bandwidth, family, shape, and materials from free-form user text using regex plus optional LLM JSON extraction. |
| `app/llm/ollama_client.py` | Thin HTTP client for Ollama health checks, JSON generation, text generation, and model warmup. |
| `app/llm/session_context_builder.py` | Creates a compact session and history summary for refinement prompts. |

### `app/planning/`

| File | Purpose |
| --- | --- |
| `app/planning/action_catalog.py` | Human-readable and machine-readable catalog of allowed high-level CST actions, parameters, prerequisites, and phases. |
| `app/planning/action_rules.py` | Loads the rule book and action priors, scores matching rule candidates, and supplies deterministic fallback refinement actions. |
| `app/planning/command_compiler.py` | Converts an action plan into a `cst_command_package.v2` payload and validates it. |
| `app/planning/dynamic_planner.py` | Picks the next refinement strategy using rule scores first and LLM tie-breaking or fallback only when allowed. |
| `app/planning/geometry_guardrails.py` | Limits per-iteration geometry deltas so refinement actions cannot move dimensions too aggressively. |
| `app/planning/v2_command_contract.py` | Strict per-command preflight validator for v2 CST command packages. |

### `context_files/`

| File | Purpose |
| --- | --- |
| `context_files/antenna_design_bounds.md` | Human-readable engineering bounds and design ranges. |
| `context_files/command_catalog.md` | Human-readable description of the supported CST command vocabulary. |
| `context_files/examples_e2e.md` | Example end-to-end request, response, and workflow references. |
| `context_files/feedback_examples.md` | Example client-feedback payloads and metric shapes. |
| `context_files/optimization_guide.md` | Optimization guidance injected into the refinement-action ranking prompt. |
| `context_files/safety_guardrails.md` | Narrative safety rules behind geometry and planning limits. |
| `context_files/system_prompt.md` | System-prompt context used for conversational and LLM-assisted flows. |
| `context_files/capabilities/antenna_capabilities.v1.json` | The catalog returned by `/api/v1/capabilities`. |
| `context_files/rule_book/action_effect_priors.v1.json` | Prior score adjustments for refinement actions. |
| `context_files/rule_book/s11_refinement_rules.v1.json` | Condition-to-action rule book used by the deterministic refinement planner. |

### `data/`

| File | Purpose |
| --- | --- |
| `data/raw/amc_patch_feedback_v1.csv` | Raw AMC feedback rows ingested from client results. |
| `data/raw/dataset.csv` | Legacy mixed-family synthetic dataset used by the generic ANN training path. |
| `data/raw/live_results_v1.jsonl` | Append-only results ledger used to deduplicate ingested feedback. |
| `data/raw/rect_patch_feedback_v1.csv` | Raw client feedback rows for rectangular microstrip designs. |
| `data/raw/rect_patch_formula_plus_client_v1.csv` | Augmented rectangular-patch training dataset that merges formula-generated rows with client reference data. |
| `data/raw/rect_patch_formula_synth_v1.csv` | Synthetic rectangular-patch formula dataset used to bootstrap the family model. |
| `data/raw/wban_patch_feedback_v1.csv` | Raw WBAN feedback rows ingested from client results. |
| `data/raw/wban_patch_formula_synth_v1.csv` | Synthetic WBAN formula dataset used to bootstrap the WBAN family model. |
| `data/rejected/amc_patch_feedback_rejected_v1.csv` | AMC feedback rows rejected by validation. |
| `data/rejected/dataset_rejected.csv` | Rejected rows from the legacy mixed-family dataset validation pass. |
| `data/rejected/rect_patch_feedback_rejected_v1.csv` | Rectangular-patch feedback rows rejected by validation. |
| `data/rejected/wban_patch_feedback_rejected_v1.csv` | WBAN feedback rows rejected by validation. |
| `data/validated/amc_patch_feedback_validated_v1.csv` | AMC feedback rows accepted by validation. |
| `data/validated/dataset_validated.csv` | Validated rows used by the legacy generic ANN trainer. |
| `data/validated/rect_patch_feedback_validated_v1.csv` | Validated rectangular-patch feedback rows. |
| `data/validated/rect_patch_forward_train_v1.csv` | Forward-model-style rectangular-patch dataset derived from validated feedback. |
| `data/validated/rect_patch_inverse_train_v1.csv` | Inverse-model rectangular-patch dataset derived from validated feedback. |
| `data/validated/wban_patch_feedback_validated_v1.csv` | Validated WBAN feedback rows. |

### `docs/`

| File | Purpose |
| --- | --- |
| `docs/ARCHITECTURE_AND_WORKFLOW.md` | Current code-backed architecture and workflow document. |
| `docs/CLIENT_SIDE_ANN_AND_LIVE_RETRAIN_HANDOFF.md` | Handoff guidance for client-side ANN and live-retraining integration. |
| `docs/COMPLETE_PROJECT_WORKING_AND_FILE_STRUCTURE.md` | Older long-form repository walkthrough; useful, but not the authoritative current-code summary. |
| `docs/INSTALLATION_AND_USAGE.md` | Setup, run, and API usage instructions. |
| `docs/PROJECT_REFERENCE_FOR_COPILOT.md` | Compact reference guide for future Copilot sessions. |
| `docs/amc_rules_server.md` | AMC-specific backend planning and rule notes. |
| `docs/rect_patch_rules_server.md` | Rectangular-patch rule notes. |
| `docs/wban_rules_server.md` | WBAN-specific backend planning and rule notes. |

### `logs/`

| File | Purpose |
| --- | --- |
| `logs/antenna_client.log` | Current client-side runtime log captured in the repository workspace. |

### `models/ann/`

| File | Purpose |
| --- | --- |
| `models/ann/rect_patch_v1/inverse_ann.pt` | Trained rectangular microstrip inverse ANN checkpoint. |
| `models/ann/rect_patch_v1/live_retrain_state.json` | Persisted retraining state and row thresholds for live retraining. |
| `models/ann/rect_patch_v1/metadata.json` | Feature order, scaling, metrics, safe bounds, and version metadata for the rectangular model. |
| `models/ann/v1/inverse_ann.pt` | Legacy generic inverse ANN checkpoint used as a fallback artifact. |
| `models/ann/v1/metadata.json` | Metadata for the legacy generic ANN. |
| `models/ann/wban_patch_v1/inverse_ann.pt` | Trained WBAN family inverse ANN checkpoint. |
| `models/ann/wban_patch_v1/metadata.json` | Feature order, scaling, metrics, and version metadata for the WBAN model. |

### `schemas/`

| File | Purpose |
| --- | --- |
| `schemas/commands/cst_command_package.v1.json` | Legacy v1 command-package schema kept for reference and backward compatibility. |
| `schemas/commands/cst_command_package.v2.command_contract.json` | Command-level contract used by the strict v2 preflight validator. |
| `schemas/commands/cst_command_package.v2.json` | Top-level JSON Schema for the runtime v2 CST command package. |
| `schemas/data/amc_patch_feedback.v1.json` | JSON Schema for AMC feedback row structure. |
| `schemas/data/rect_patch_feedback.v1.json` | JSON Schema for rectangular-patch feedback row structure. |
| `schemas/data/wban_patch_feedback.v1.json` | JSON Schema for WBAN feedback row structure. |
| `schemas/http/client_feedback.v1.json` | HTTP feedback contract used by `/api/v1/client-feedback` and `/api/v1/result`. |
| `schemas/http/optimize_request.v1.json` | HTTP optimize-request contract used by `/api/v1/optimize`. |
| `schemas/http/optimize_response.v1.json` | HTTP optimize-response contract returned by the optimize path. |
| `schemas/planning/action_catalog.v1.json` | Schema for the planning action catalog. |
| `schemas/planning/action_plan.v1.json` | Schema for the fixed action plan before compilation to v2 commands. |
| `schemas/ws/session_event.v1.json` | WebSocket event schema used by session streaming. |

### `scripts/`

| File | Purpose |
| --- | --- |
| `scripts/derive_rect_patch_datasets.py` | Rebuilds validated and rejected rectangular feedback outputs plus inverse and forward training CSVs. |
| `scripts/evaluate_rect_patch_ann.py` | Evaluates the rectangular-patch ANN against validated feedback data. |
| `scripts/generate_family_ann_datasets.py` | CLI for generating formula-based family datasets for microstrip and WBAN training. |
| `scripts/generate_synthetic_dataset.py` | Generates the legacy mixed-family dataset used by the generic ANN path. |
| `scripts/generate_sysnthetic_dataset.py` | Typo-compatible wrapper that simply runs `generate_synthetic_dataset.py`. |
| `scripts/replay_session.py` | Replays a saved session JSON and prints its decisions, iterations, metrics, and history. |
| `scripts/run_rect_patch_pipeline.py` | One-shot rectangular-patch workflow: build datasets, then train the rectangular model. |
| `scripts/train_ann.py` | Trains the legacy generic ANN from `data/validated/dataset_validated.csv`. |
| `scripts/train_family_anns.py` | Generates datasets, augments rectangular data with client rows, and trains family-specific ANNs. |
| `scripts/train_rect_patch_ann.py` | Trains only the rectangular-patch inverse ANN. |
| `scripts/validate_dataset.py` | Validates the legacy mixed-family dataset against the configured bounds. |
| `scripts/validate_rect_patch_feedback.py` | Validates raw rectangular-patch feedback rows before dataset derivation. |

### `sessions/`

| Path | Purpose |
| --- | --- |
| `sessions/<uuid>.json` | Persisted session state, history, current ANN prediction, command package, policy runtime, and artifact manifest for each optimization run. |

### `tests/`

#### `tests/conftest.py`

- Shared test fixtures and test configuration.

#### `tests/integration/`

| File | Purpose |
| --- | --- |
| `tests/integration/test_capabilities_chat.py` | End-to-end checks for `/capabilities`, `/intent/parse`, and `/chat`. |
| `tests/integration/test_family_registry.py` | Integration checks for family normalization and constraint validation. |
| `tests/integration/test_iteration_flow.py` | End-to-end optimize, feedback, refine, and complete session flow tests. |
| `tests/integration/test_llm_live.py` | Integration coverage for live LLM and Ollama-backed behavior. |
| `tests/integration/test_result_ingest.py` | Feedback and result-ingestion endpoint coverage. |
| `tests/integration/test_surrogate_policy.py` | Surrogate low-confidence policy behavior for clarification and error branches. |
| `tests/integration/test_ws_stream.py` | Session WebSocket streaming behavior and terminal events. |

#### `tests/unit/`

| File | Purpose |
| --- | --- |
| `tests/unit/test_ann_recipe_predictor.py` | Unit checks for recipe generation plus ANN prediction behavior. |
| `tests/unit/test_dataset_validator.py` | Unit checks for legacy dataset validation rules. |
| `tests/unit/test_family_ann_trainer.py` | Unit checks for family ANN training and evaluation helpers. |
| `tests/unit/test_family_dataset_generators.py` | Unit checks for synthetic family dataset generation. |
| `tests/unit/test_family_feedback_workflow.py` | Unit checks for AMC and WBAN feedback validation and dataset generation. |
| `tests/unit/test_family_geometry_commands.py` | Unit checks that family-specific command packages emit the right geometry commands. |
| `tests/unit/test_feedback_objectives.py` | Unit checks for objective-state and acceptance calculations. |
| `tests/unit/test_geometry_guardrails.py` | Unit checks for per-action geometry guardrails. |
| `tests/unit/test_intent_parser_llm.py` | Unit checks for regex and LLM intent parsing behavior. |
| `tests/unit/test_live_retraining.py` | Unit checks for feedback ingestion and retraining trigger logic. |
| `tests/unit/test_planning_phase1.py` | Unit checks for early planning and command compilation behavior. |
| `tests/unit/test_recipe_pipeline.py` | Unit checks for the deterministic antenna recipe pipeline. |
| `tests/unit/test_rect_patch_feedback_workflow.py` | Unit checks for rectangular-patch feedback logging and dataset derivation. |
| `tests/unit/test_refinement_geometry_corrections.py` | Unit checks for geometry refinement corrections after feedback. |
| `tests/unit/test_rule_based_refinement.py` | Unit checks for rule-book-based refinement candidate selection. |

### `web/`

| File | Purpose |
| --- | --- |
| `web/index.html` | Single-file static demo client for health checks, chat, pipeline start, and response inspection. |

## 6) Libraries Used and Why

### Direct third-party libraries imported by the repository

| Library | Where it is used | Purpose in this project |
| --- | --- | --- |
| `fastapi` | `server.py` | Defines HTTP routes, request handling, exceptions, and the WebSocket endpoint. |
| `uvicorn` | `main.py` | Runs the ASGI application. |
| `pydantic` | `app/core/schemas.py` | Declares strongly typed request and response models and runtime validation rules. |
| `jsonschema` | `app/core/json_contracts.py`, `app/data/rect_patch_feedback_logger.py` | Validates HTTP payloads, planning payloads, and feedback rows against JSON Schema contracts. |
| `numpy` | `app/ann/*`, `app/data/family_dataset_generators.py` | Handles numeric arrays, scaling, synthetic data generation, and metric calculations. |
| `pandas` | `app/data/*`, `app/ann/*trainer.py` | Reads and writes CSV datasets and performs tabular validation and cleanup. |
| `torch` | `app/ann/model.py`, `app/ann/predictor.py`, trainer modules | Defines, trains, saves, loads, and runs the ANN models. |
| `httpx` | `app/llm/ollama_client.py` | Makes HTTP calls to the local Ollama server for health checks and model generation. |
| `pytest` | `tests/` | Executes unit and integration tests. |

### Runtime dependency and service that is not a Python import in the repo

| Dependency | Purpose |
| --- | --- |
| `Ollama` | External local LLM service used for intent parsing, natural chat replies, and refinement candidate ranking. |

### Standard-library modules used heavily

| Module | Purpose in this project |
| --- | --- |
| `asyncio` | Polling and pacing the WebSocket session stream. |
| `threading` | Background ANN and LLM warmup and asynchronous live retraining workers. |
| `pathlib` | Central path construction for models, datasets, schemas, and sessions. |
| `dataclasses` | Immutable settings objects and training artifact containers. |
| `json` | Session persistence, schema loading, metadata loading, and results ledger serialization. |
| `uuid` | Session IDs and trace IDs. |
| `datetime` | Timestamping history entries, manifests, and feedback rows. |
| `csv` | Appending feedback rows to raw dataset files. |
| `math` | Patch-geometry formulas and analytic recipe generation. |
| `re` | Regex-based extraction of frequencies, bandwidths, families, and materials from user text. |
| `hashlib` | Stable SHA-256 checksums for command packages in session manifests. |

### Installed but not directly imported by repository code

The active environment also contains transitive dependencies such as `starlette`, `anyio`, `h11`, and `websockets`. They matter at runtime because FastAPI and Uvicorn depend on them, but the repository code does not import them directly.

## 7) Exact Runtime Workflow

### 7.1 Startup and warmup

1. `main.py` calls `uvicorn.run("server:app", ...)`.
2. `server.py` creates global singletons:
     - `brain = CentralBrain()`
     - `session_store = SessionStore()`
3. `FastAPI` is initialized and CORS is opened broadly.
4. On startup, `start_background_warmup()` launches a thread that:
     - warms ANN artifacts via `brain.ann_predictor.warm_up()`,
     - checks Ollama connectivity,
     - warms the fast model and the chat model,
     - updates `_runtime_health`.

### 7.2 Health and capability discovery

- `GET /api/v1/health`
    - ensures warmup is in progress or restarted if needed,
    - reports ANN artifact readiness, model-load state, Ollama reachability, configured model names, and live retraining status.
- `GET /api/v1/capabilities`
    - returns the catalog from `context_files/capabilities/antenna_capabilities.v1.json`,
    - falls back to built-in bounds and supported families if the catalog file is missing or invalid.

### 7.3 Optional intent parsing and chat

- `POST /api/v1/intent/parse`
    - validates that `user_request` exists,
    - runs `summarize_user_intent()`,
    - returns a structured intent summary only.

- `POST /api/v1/chat`
    - accepts `message`, optional `requirements`, and optional `session_id`,
    - rehydrates remembered requirements from in-memory chat state and from a saved optimization session if one exists,
    - parses the current message with `summarize_user_intent()`,
    - merges parsed values into the remembered requirements,
    - optionally calls the larger Ollama model for a natural-language reply,
    - always returns structured state:
        - `session_id`
        - `assistant_message`
        - `intent_summary`
        - `requirements`
        - `missing_requirements`
        - `ready_to_start_pipeline`
        - supported families and capability ranges.

The chat path is not the real optimizer. It is a requirements-capture and explanation layer.

### 7.4 Optimize request path

`POST /api/v1/optimize` is the real session-creation entrypoint.

#### Request validation

1. `server.optimize()` validates the raw payload against `schemas/http/optimize_request.v1.json`.
2. The payload is then converted into `OptimizeRequest` via Pydantic.
3. If validation fails, the server returns `422`.

#### `CentralBrain.optimize()` execution

1. A `session_id` and `trace_id` are generated unless the caller supplied a `session_id`.
2. `apply_family_profile()` validates the selected family and the requested materials and substrates.
3. `summarize_user_intent()` records an intent summary for the session.
4. `build_initial_objective_state()` initializes the objective tracker.

#### ANN prediction path

1. `AnnPredictor.predict()` first builds a deterministic recipe with `generate_recipe()`.
2. Special case: if the family is `amc_patch`, the predictor does not load or run a server ANN. It returns:
     - `ann_model_version = "amc_client_local_implementation"`
     - deterministic patch, substrate, and feed dimensions
     - `family_parameters = {}`
     - `optimizer_hint = "client_implement_amc"`
3. For other families, the predictor:
     - resolves the request family and patch shape,
     - tries a family-specific checkpoint first (`rect_patch_v1` or `wban_patch_v1`),
     - falls back to the legacy generic model `models/ann/v1` if needed,
     - builds a feature vector with `build_ann_feature_map()`,
     - loads checkpoint and metadata,
     - standardizes inputs,
     - runs the PyTorch regressor,
     - unscales outputs,
     - blends model outputs with deterministic recipe dimensions,
     - clamps outputs to safe bounds from metadata or global bounds,
     - returns an `AnnPrediction`.

#### Surrogate safety gate

1. `validate_with_surrogate()` estimates center frequency and bandwidth from geometry heuristics.
2. It combines:
     - ANN confidence,
     - frequency score,
     - bandwidth score,
     - domain-support score.
3. It returns `accepted`, `confidence`, `threshold`, residuals, and a decision reason.

#### Policy branch if surrogate confidence is low

The behavior depends on `optimization_policy.fallback_behavior`:

- `best_effort`
    - continue and still build a command package.
- `require_user_confirmation`
    - store a session with `status = clarification_required`,
    - return `clarification_required`,
    - do not return a command package.
- `return_error`
    - store a session with `status = error`,
    - return `error`,
    - do not return a command package.

#### Command package generation

If execution continues:

1. `build_command_package()` calls `build_fixed_action_plan()`.
2. The fixed planner:
     - chooses conductor and substrate properties,
     - maps predicted dimensions to CST parameter names,
     - emits ordered setup, material, geometry, simulation, and export actions,
     - emits `implement_amc` when the family is `amc_patch`,
     - emits `add_farfield_monitor` when `supports_farfield_export` is true,
     - uses parameter updates instead of full geometry recreation on later iterations.
3. `compile_action_plan()` turns the action plan into `cst_command_package.v2`.
4. `validate_command_package_v2()` enforces:
     - top-level required fields,
     - per-command parameter validity,
     - strictly increasing sequence numbers,
     - a `rebuild_model` after parameter changes and before simulation or export.
5. JSON Schema validation is run again on the final response and on the command package.

#### Session creation

1. `SessionStore.create()` writes `sessions/<session_id>.json`.
2. The session stores:
     - request payload,
     - current ANN prediction,
     - surrogate validation,
     - current command package,
     - max iterations,
     - policy runtime state,
     - objective targets and state,
     - history,
     - artifact manifest.
3. `CentralBrain.optimize()` then writes `intent_summary` into the session and returns the optimize response.

### 7.5 What the client receives and must do

The response contains:

- `session_id`
- `trace_id`
- `ann_prediction`
- `objective_state`
- `command_package`
- `warnings` from the surrogate gate

The client must:

1. persist `session_id`, `trace_id`, and `design_id`,
2. execute `command_package.commands` strictly in order,
3. export the requested artifacts,
4. send measured results back via `/api/v1/client-feedback` or `/api/v1/result`.

The server never launches CST directly.

### 7.6 Feedback and refinement loop

`POST /api/v1/client-feedback` and `POST /api/v1/result` share the same handler.

#### Feedback validation and session loading

1. `server._process_client_result()` validates the payload against `schemas/http/client_feedback.v1.json`.
2. `CentralBrain.process_feedback()` loads the session JSON.
3. The reported `iteration_index` must match the current session iteration.
4. One special-case tolerance exists: when `completion_requested = true`, the server accepts a one-step-ahead iteration index to accommodate a known QML restore flow.

#### Acceptance evaluation

1. `evaluate_acceptance()` compares actual metrics against:
     - center-frequency tolerance,
     - minimum bandwidth,
     - minimum return loss,
     - maximum VSWR,
     - minimum gain.
2. `evaluate_objective_state()` updates primary and secondary objective tracking.
3. A `feedback_evaluation` history entry is appended to the session.

#### Dataset ingestion and live retraining side effects

`OnlineRetrainingManager.ingest_result()` always appends a JSONL ledger entry if it is not already recorded, then branches by family:

- `amc_patch`
    - builds an AMC feedback row,
    - appends it to `data/raw/amc_patch_feedback_v1.csv`,
    - materializes validated and rejected AMC CSVs,
    - does not trigger automatic retraining because `_trigger_retraining_if_needed()` returns `False` for AMC.
- `wban_patch`
    - builds a WBAN feedback row,
    - appends it to `data/raw/wban_patch_feedback_v1.csv`,
    - materializes validated and rejected WBAN CSVs,
    - may trigger asynchronous retraining if enough new valid rows exist.
- rectangular `microstrip_patch`
    - builds a rectangular feedback row,
    - appends it to `data/raw/rect_patch_feedback_v1.csv`,
    - materializes validated and rejected rectangular CSVs,
    - rebuilds `rect_patch_inverse_train_v1.csv` and `rect_patch_forward_train_v1.csv`,
    - may trigger asynchronous retraining if enough new valid rows exist.
- unsupported combinations
    - store only the results ledger entry and skip family retraining.

#### Completion and stop branches

After evaluation and dataset ingestion:

- If acceptance criteria are met:
    - session becomes `completed`,
    - no new command package is generated.
- If `completion_requested` is true:
    - session becomes `completed` with `stop_reason = user_marked_done`,
    - no new command package is generated.
- If max iterations are reached:
    - session becomes `max_iterations_reached`,
    - no new command package is generated.

#### Refinement branch

If the run is not accepted and can continue:

1. `derive_feedback_features()` computes normalized refinement features and severity.
2. `plan_refinement_strategy()` ranks candidates from `s11_refinement_rules.v1.json` plus action priors.
3. If deterministic confidence is below the LLM threshold and the session budget allows it:
     - the system builds a compact context,
     - calls Ollama via `choose_action_with_llm()`,
     - uses the returned action if it matches a candidate,
     - otherwise falls back to the top-scored rule candidate.
4. `refine_prediction_with_strategy()` updates geometry:
     - uses explicit scale and offset strategies when present,
     - otherwise applies heuristic corrections for frequency, bandwidth, matching, and gain,
     - applies per-action geometry guardrails,
     - clamps to request constraints and global bounds.
5. A new command package is built for the next iteration.
6. The session is updated with:
     - incremented iteration index,
     - refined ANN prediction,
     - new command package,
     - `refinement_plan` history entry,
     - updated artifact manifest history.

### 7.7 Session lookup and WebSocket streaming

- `GET /api/v1/sessions/{session_id}`
    - returns the current session snapshot, including:
        - `trace_id`
        - `design_id`
        - `status`
        - `stop_reason`
        - current and max iteration
        - intent summary
        - full surrogate validation and a compact surrogate summary
        - policy runtime counters
        - latest planning decision
        - latest history entry
        - current command package

- `WS /api/v1/sessions/{session_id}/stream`
    - accepts the socket,
    - polls the session JSON once per second,
    - whenever history length increases, emits `iteration.completed` events for new entries,
    - when the session reaches `completed`, `max_iterations_reached`, or `stopped`, emits a terminal event and closes,
    - emits `session.failed` if the session is missing or an exception occurs.

Important implementation detail:

- The schema allows many event types, but the current implementation only emits `iteration.completed`, `session.completed`, and `session.failed`.

## 8) Training and Data-Maintenance Workflow

The repository has two parallel model and data tracks.

### 8.1 Legacy generic ANN track

This older path is centered on:

- `data/raw/dataset.csv`
- `data/validated/dataset_validated.csv`
- `models/ann/v1/`

Typical flow:

1. `scripts/generate_synthetic_dataset.py`
2. `scripts/validate_dataset.py`
3. `scripts/train_ann.py`

This path supports the legacy fallback model.

### 8.2 Family-specific ANN track

This is the actively used path for microstrip rectangular and WBAN families.

#### Synthetic bootstrap data

- `scripts/generate_family_ann_datasets.py` generates formula-based family datasets.
- Those datasets feed `train_family_anns.py`.

#### Rectangular-patch feedback loop

1. Client or server writes raw rows into `data/raw/rect_patch_feedback_v1.csv`.
2. `scripts/validate_rect_patch_feedback.py` checks raw rows.
3. `scripts/derive_rect_patch_datasets.py` materializes:
     - validated feedback,
     - rejected feedback,
     - inverse-training CSV,
     - forward-training CSV.
4. `scripts/train_rect_patch_ann.py` trains the rectangular model.
5. `scripts/evaluate_rect_patch_ann.py` compares recipe vs ANN on validated data.
6. `scripts/run_rect_patch_pipeline.py` runs the derive-and-train pipeline end to end.

#### Family ANN training

- `scripts/train_family_anns.py`
    - can regenerate family datasets,
    - augments rectangular training data with client reference rows when available,
    - trains the family-specific ANNs,
    - prints evaluation summaries.

#### Session audit

- `scripts/replay_session.py <session_id>` pretty-prints the full stored decision path for a saved session.

## 9) API Surface Summary

| Method | Endpoint | What it does |
| --- | --- | --- |
| `GET` | `/api/v1/health` | Reports service, ANN, LLM, and live-retraining state. |
| `GET` | `/api/v1/capabilities` | Returns supported families, material options, and range limits. |
| `POST` | `/api/v1/intent/parse` | Parses a natural-language request into a structured intent summary. |
| `POST` | `/api/v1/chat` | Captures requirements conversationally and returns merged structured state. |
| `POST` | `/api/v1/optimize` | Validates a design request, creates or resumes a session, and emits the CST command package. |
| `POST` | `/api/v1/client-feedback` | Accepts CST feedback and moves the session to accept, refine, or stop. |
| `POST` | `/api/v1/result` | Alias of `/api/v1/client-feedback`. |
| `GET` | `/api/v1/sessions/{session_id}` | Returns the current saved session snapshot. |
| `WS` | `/api/v1/sessions/{session_id}/stream` | Streams session history updates and terminal events. |

## 10) Practical Mental Model

If you need the shortest accurate picture of the codebase, it is this:

1. `server.py` is the API shell.
2. `central_brain.py` is the orchestration brain.
3. `app/antenna/recipes.py` provides deterministic starting geometry.
4. `app/ann/predictor.py` adds family-aware ANN inference on top of the recipe when model artifacts exist.
5. `app/core/surrogate_validator.py` safety-checks the result.
6. `app/commands/planner.py` and `app/planning/command_compiler.py` emit a strict `cst_command_package.v2`.
7. The client executes CST locally and posts measured feedback back to the server.
8. `app/core/refinement.py`, `app/planning/action_rules.py`, and `app/planning/dynamic_planner.py` decide the next refinement step.
9. `app/core/session_store.py` and `sessions/<uuid>.json` are the persistent memory of every run.
10. `app/ann/live_retraining.py` turns accepted and unaccepted feedback into future training data.
