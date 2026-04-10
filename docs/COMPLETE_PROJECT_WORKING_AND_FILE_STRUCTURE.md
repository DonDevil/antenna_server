# Complete Project Working, Architecture, and File Structure

This document gives a single, practical explanation of how the `antenna_server` project works, how data flows through it, and what each major file/folder is used for.

---

## 1) Project Purpose

`antenna_server` is a **backend antenna-design orchestration service**.
It accepts antenna requirements from a client, predicts an initial geometry using an **ANN**, validates the prediction with safety checks, returns a **safe CST command package**, and then improves the design iteratively based on **client feedback** from simulation results.

> **Important boundary:** the server plans and validates the optimization flow, but it does **not** run CST itself. The **client side** executes CST locally and sends the measured results back.

---

## 2) High-Level Architecture

| Layer | Main Files | Responsibility |
|---|---|---|
| API / Entry | `main.py`, `server.py` | Starts FastAPI server, exposes HTTP and WebSocket endpoints |
| Orchestration | `central_brain.py` | Coordinates optimize + feedback workflows |
| ANN Inference | `app/ann/` | Loads trained models, prepares features, predicts antenna dimensions |
| LLM Support | `app/llm/` | Parses intent and helps rank refinement actions via Ollama |
| Planning | `app/commands/`, `app/planning/` | Builds safe CST command packages and selects next actions |
| Core Rules / State | `app/core/` | Schemas, validation, sessions, objectives, safety, exceptions |
| Antenna Knowledge | `app/antenna/` | Materials database and deterministic recipe generation |
| Contracts | `schemas/` | JSON schemas for HTTP, commands, planning, data, and WS payloads |
| Data / Models | `data/`, `models/` | Training data, validated feedback, trained ANN checkpoints |
| QA / Verification | `tests/` | Unit and integration tests for workflow correctness |

---

## 3) End-to-End Working of the Project

### Step 1: Server startup
- `main.py` launches the FastAPI app with Uvicorn.
- `server.py` creates the app, configures CORS, initializes `CentralBrain`, and starts background warmup.
- During warmup:
  - ANN models are loaded through `app/ann/predictor.py`
  - Ollama models are warmed through `app/llm/ollama_client.py`

### Step 2: Health and capability discovery
The client can call:
- `GET /api/v1/health` — checks ANN + LLM readiness
- `GET /api/v1/capabilities` — returns supported families, bounds, materials, and runtime capabilities

### Step 3: Optional chat / intent parsing
The client can first collect or refine requirements through:
- `POST /api/v1/intent/parse`
- `POST /api/v1/chat`

These endpoints help convert natural language like:
> “Design a 2.45 GHz microstrip patch antenna with 100 MHz bandwidth”

into structured fields such as frequency, bandwidth, family, shape, materials, and constraints.

### Step 4: Optimization request begins
The main pipeline starts at:
- `POST /api/v1/optimize`

The request is validated against the JSON contract in:
- `schemas/http/optimize_request.v1.json`

The corresponding Pydantic model lives in:
- `app/core/schemas.py`

### Step 5: Central brain orchestration
`central_brain.py` is the heart of the system. Its `CentralBrain.optimize()` method:
1. normalizes the request
2. applies family-specific defaults (`app/core/family_registry.py`)
3. summarizes the intent (`app/llm/intent_parser.py`)
4. runs ANN prediction (`app/ann/predictor.py`)
5. validates with the surrogate safety gate (`app/core/surrogate_validator.py`)
6. builds a command package (`app/commands/planner.py`)
7. creates and saves a session (`app/core/session_store.py`)

### Step 6: Command package generation
The planner returns a safe, ordered set of high-level CST actions such as:
- `create_project`
- `set_units`
- `define_material`
- `create_substrate`
- `create_patch`
- `create_feedline`
- `run_simulation`
- `export_s_parameters`

These commands are **not executed by the server**. They are sent back to the client for local CST automation.

### Step 7: Session persistence
Every optimization run creates a session JSON under `sessions/`.
A session stores:
- the original request
- current ANN prediction
- current command package
- objective state
- decision history
- iteration count
- stop reason / acceptance status

### Step 8: Feedback loop
After the client runs CST locally, it sends results to:
- `POST /api/v1/client-feedback`
- `POST /api/v1/result` (alternate path in the server)

`CentralBrain.process_feedback()` then:
1. loads the saved session
2. evaluates acceptance criteria
3. updates objective state
4. derives feedback features
5. selects a refinement strategy
6. emits a revised command package if another iteration is needed

### Step 9: Streaming session status
The client can watch progress through:
- `WS /api/v1/sessions/{session_id}/stream`

This is used for live updates such as iteration completion, final success, or failure.

---

## 4) Practical Flow Summary

```text
Client Request
   -> `server.py`
   -> request/schema validation
   -> `central_brain.py`
   -> ANN prediction (`app/ann/predictor.py`)
   -> surrogate safety validation (`app/core/surrogate_validator.py`)
   -> command planning (`app/commands/planner.py`)
   -> session save (`app/core/session_store.py`)
   -> command package returned to client
   -> client executes CST locally
   -> client sends measured feedback
   -> refinement strategy chosen (`app/planning/dynamic_planner.py`)
   -> accept / iterate / stop
```

---

## 5) File Structure Overview

```text
antenna_server/
├── main.py
├── server.py
├── central_brain.py
├── config.py
├── requirements.txt
├── hanoff.md
├── app/
│   ├── ann/
│   ├── antenna/
│   ├── api/
│   ├── commands/
│   ├── core/
│   ├── data/
│   ├── llm/
│   └── planning/
├── context_files/
├── data/
│   ├── raw/
│   ├── rejected/
│   └── validated/
├── docs/
├── env/
├── logs/
├── models/
│   └── ann/
├── schemas/
│   ├── commands/
│   ├── data/
│   ├── http/
│   ├── planning/
│   └── ws/
├── scripts/
├── sessions/
├── tests/
└── web/
```

---

## 6) Top-Level Files and Their Use

| File | Use |
|---|---|
| `main.py` | Simple startup entry point; runs the FastAPI app with Uvicorn. |
| `server.py` | Main API layer; defines routes, startup warmup, health checks, chat, optimization, feedback, session lookup, and WebSocket streaming. |
| `central_brain.py` | Core orchestration engine; coordinates ANN prediction, validation, planning, refinement, and session state updates. |
| `config.py` | Central configuration for API settings, model paths, planner behavior, bounds, datasets, and runtime constants. |
| `requirements.txt` | Python package dependencies such as `fastapi`, `uvicorn`, `pydantic`, `torch`, `numpy`, `pandas`, and `pytest`. |
| `hanoff.md` | Handoff/reference notes for project continuity or implementation context. |

---

## 7) `app/` Package Breakdown

### `app/__init__.py`
- Marks `app/` as a Python package.

### `app/ann/` — ANN model loading, prediction, training, retraining

| File | Use |
|---|---|
| `app/ann/baseline.py` | Baseline or fallback geometry logic when learned prediction is unavailable or needs comparison. |
| `app/ann/family_ann_trainer.py` | Training helpers for antenna-family-specific ANN models. |
| `app/ann/features.py` | Converts an optimization request into ANN-ready feature vectors. |
| `app/ann/live_retraining.py` | Online retraining manager; ingests real feedback and supports model improvement over time. |
| `app/ann/model.py` | PyTorch model definition for inverse ANN regression. |
| `app/ann/predictor.py` | Loads ANN artifacts, warms models, selects family-specific checkpoints, and performs dimension prediction. |
| `app/ann/rect_patch_evaluator.py` | Evaluation logic for rectangular patch ANN behavior and quality checks. |
| `app/ann/rect_patch_trainer.py` | Training pipeline focused on the rectangular patch model. |
| `app/ann/trainer.py` | Generic ANN training utilities and training entry support. |

### `app/antenna/` — antenna domain knowledge

| File | Use |
|---|---|
| `app/antenna/materials.py` | Material lookup helpers for conductors and substrates. |
| `app/antenna/recipes.py` | Deterministic antenna geometry recipes for rectangular, circular, AMC-backed, and family-specific designs. |

### `app/api/`
- Currently empty/reserved for future API modularization.

### `app/commands/` — command package generation

| File | Use |
|---|---|
| `app/commands/planner.py` | Converts request + ANN prediction into a validated CST command package. |

### `app/core/` — contracts, rules, sessions, and safety logic

| File | Use |
|---|---|
| `app/core/capabilities_catalog.py` | Loads/returns the supported design capabilities catalog exposed by the API. |
| `app/core/exceptions.py` | Custom domain exceptions such as unsupported family or invalid constraints. |
| `app/core/family_registry.py` | Applies family-specific profiles/default constraints to normalized requests. |
| `app/core/feedback_features.py` | Derives structured features from client simulation feedback for refinement logic. |
| `app/core/json_contracts.py` | JSON-schema validation helpers for API and command contracts. |
| `app/core/objectives.py` | Builds and evaluates optimization objective state (S11, bandwidth, gain, etc.). |
| `app/core/policy_runtime.py` | Runtime policy counters and LLM-gating state during planning/refinement. |
| `app/core/refinement.py` | Acceptance evaluation and parameter-refinement helpers after each feedback cycle. |
| `app/core/schemas.py` | Pydantic request/response/data models such as `OptimizeRequest`, `OptimizeResponse`, and `AnnPrediction`. |
| `app/core/session_store.py` | Saves, loads, and updates JSON-backed session state under `sessions/`. |
| `app/core/surrogate_validator.py` | Safety confidence checks to decide whether ANN output is acceptable for automatic execution. |
| `app/core/test_command.txt` | Small internal/test asset related to command handling. |

### `app/data/` — training/feedback ingestion utilities

| File | Use |
|---|---|
| `app/data/family_dataset_generators.py` | Generates family-specific datasets for model training. |
| `app/data/family_feedback.py` | Structures and processes feedback records across antenna families. |
| `app/data/rect_patch_feedback.py` | Rectangular-patch-specific feedback helpers/schema mapping. |
| `app/data/rect_patch_feedback_logger.py` | Appends or logs rectangular patch feedback for later retraining. |
| `app/data/schema.py` | Defines required dataset columns and data contract expectations. |
| `app/data/store.py` | Reads raw CSV data and writes validated/rejected outputs. |
| `app/data/validator.py` | Rejects invalid or out-of-range dataset rows using configured bounds. |

### `app/llm/` — intent parsing and LLM-assisted planning

| File | Use |
|---|---|
| `app/llm/action_ranker.py` | Uses the LLM to rank/select refinement actions when rule confidence is low. |
| `app/llm/intent_parser.py` | Parses or summarizes user intent from natural language requests. |
| `app/llm/ollama_client.py` | Low-level Ollama connectivity, health, generation, and warmup utilities. |
| `app/llm/session_context_builder.py` | Builds compact session context/history for LLM prompting. |

### `app/planning/` — refinement planning and command compilation

| File | Use |
|---|---|
| `app/planning/action_catalog.py` | Catalog of supported planning/refinement actions. |
| `app/planning/action_rules.py` | Deterministic rule ranking for choosing the next refinement move. |
| `app/planning/command_compiler.py` | Compiles high-level planning actions into executable command steps. |
| `app/planning/dynamic_planner.py` | Chooses refinement strategy using rules first, and optionally LLM assistance. |
| `app/planning/geometry_guardrails.py` | Protects geometry changes with safe bounds and guardrails. |
| `app/planning/v2_command_contract.py` | Validation rules for the versioned command-package contract. |

---

## 8) `schemas/` Directory

| Path | Use |
|---|---|
| `schemas/http/optimize_request.v1.json` | JSON contract for the optimize request payload. |
| `schemas/http/optimize_response.v1.json` | JSON contract for optimize responses. |
| `schemas/http/client_feedback.v1.json` | JSON contract for feedback/result payloads sent after CST execution. |
| `schemas/commands/` | Command package structure and command-level schema definitions. |
| `schemas/data/` | Dataset and data-interchange schema files. |
| `schemas/planning/` | Planning-decision and refinement related contracts. |
| `schemas/ws/` | WebSocket event schemas for session streaming. |

---

## 9) `context_files/` Directory

These files provide static context, rules, and prompts used by the system and by developers working on it.

| File / Folder | Use |
|---|---|
| `context_files/antenna_design_bounds.md` | Supported design ranges and engineering bounds. |
| `context_files/command_catalog.md` | Reference for allowed high-level CST commands. |
| `context_files/examples_e2e.md` | End-to-end example workflows and payloads. |
| `context_files/feedback_examples.md` | Sample feedback/result payload examples. |
| `context_files/optimization_guide.md` | Notes on optimization strategy and expected refinement behavior. |
| `context_files/safety_guardrails.md` | Safety constraints and restrictions for planning. |
| `context_files/system_prompt.md` | System prompt or model prompt context used by the LLM side. |
| `context_files/capabilities/` | Supporting capability definitions or catalog data. |
| `context_files/rule_book/` | Rulebook assets for the planner and domain logic. |

---

## 10) `scripts/` Directory

These are helper scripts for data generation, training, evaluation, and replay.

| Script | Use |
|---|---|
| `scripts/derive_rect_patch_datasets.py` | Derives model-ready datasets for rectangular patch training/evaluation. |
| `scripts/evaluate_rect_patch_ann.py` | Evaluates the quality of the rectangular patch ANN. |
| `scripts/generate_family_ann_datasets.py` | Generates synthetic or family-specific training datasets. |
| `scripts/generate_synthetic_dataset.py` | Creates a synthetic dataset for general training. |
| `scripts/generate_sysnthetic_dataset.py` | Duplicate/alternate synthetic-data generation script (note the filename typo). |
| `scripts/replay_session.py` | Replays a saved optimization session from `sessions/`. |
| `scripts/run_rect_patch_pipeline.py` | Convenience pipeline for rectangular patch data/train/eval flow. |
| `scripts/train_ann.py` | Trains the general ANN model. |
| `scripts/train_family_anns.py` | Trains family-specific ANN models. |
| `scripts/train_rect_patch_ann.py` | Trains the rectangular patch ANN variant. |
| `scripts/validate_dataset.py` | Validates generated dataset rows against bounds/schema. |
| `scripts/validate_rect_patch_feedback.py` | Validates feedback data for rectangular patch runs. |

---

## 11) Runtime / Asset Directories

| Path | Use |
|---|---|
| `data/raw/` | Raw datasets or raw feedback before validation. |
| `data/validated/` | Cleaned/accepted datasets used for training or retraining. |
| `data/rejected/` | Invalid or out-of-range records rejected during validation. |
| `models/ann/` | Stored ANN checkpoints and metadata (for example `inverse_ann.pt` and `metadata.json`). |
| `sessions/` | One JSON file per optimization session; acts as persistent server memory for runs. |
| `logs/` | Runtime logs or diagnostics. |
| `tests/` | Unit/integration tests that verify API flow, planning, and session lifecycle. |
| `web/` | Web-side assets or frontend/client-related resources. |
| `env/` | Local Python virtual environment; not part of the application logic itself. |

---

## 12) `docs/` Directory

| File | Use |
|---|---|
| `docs/amc_rules_server.md` | AMC-specific backend design/rule notes. |
| `docs/ARCHITECTURE_AND_WORKFLOW.md` | Existing architecture summary and runtime workflow notes. |
| `docs/CLIENT_SIDE_ANN_AND_LIVE_RETRAIN_HANDOFF.md` | Handoff notes for client-side ANN/live retraining integration. |
| `docs/INSTALLATION_AND_USAGE.md` | Setup, run, health-check, and API usage examples. |
| `docs/PROJECT_REFERENCE_FOR_COPILOT.md` | Quick reference for future development and Copilot context. |
| `docs/rect_patch_rules_server.md` | Rectangular patch rule notes for the backend. |
| `docs/wban_rules_server.md` | WBAN-specific backend rule notes. |
| `docs/COMPLETE_PROJECT_WORKING_AND_FILE_STRUCTURE.md` | This consolidated guide. |

---

## 13) Key Design Principles Used by the Project

1. **Contract-first API design**  
   HTTP payloads are validated with both Pydantic models and JSON schema contracts.

2. **Session-based optimization**  
   Every run is persisted so the system can resume, inspect, and refine over multiple iterations.

3. **Safety before execution**  
   ANN output is checked by surrogate validation and geometry guardrails before the client receives commands.

4. **Hybrid intelligence**  
   The system combines deterministic antenna formulas, ANN regression, rules, and optional LLM assistance.

5. **Client/server separation**  
   The server decides **what should be done**; the client performs **actual CST execution**.

---

## 14) Short “How It All Fits Together” Summary

If you want the shortest mental model of the codebase, it is this:

- `server.py` exposes the API.
- `central_brain.py` runs the optimization brain.
- `app/ann/` predicts starting geometry.
- `app/core/` validates, tracks objectives, and persists sessions.
- `app/commands/` and `app/planning/` generate safe next-step CST commands.
- `app/llm/` helps with intent understanding and harder refinement choices.
- `schemas/` defines what valid requests/responses must look like.
- `sessions/` keeps the history of each run.
- `scripts/` supports training, evaluation, and maintenance.

---

## 15) Recommended Reading Order for New Developers

1. `main.py`
2. `server.py`
3. `central_brain.py`
4. `app/core/schemas.py`
5. `app/ann/predictor.py`
6. `app/commands/planner.py`
7. `app/planning/dynamic_planner.py`
8. `app/core/session_store.py`
9. `schemas/http/optimize_request.v1.json`
10. `docs/INSTALLATION_AND_USAGE.md`

---

## 16) Bottom Line

This repository is a **server-side optimization brain for antenna design**.
It receives design goals, predicts a safe starting point, returns structured CST commands, learns from simulation feedback, and iterates until the design is accepted or stopped by policy.
