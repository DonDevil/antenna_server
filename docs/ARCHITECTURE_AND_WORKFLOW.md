# Architecture and Workflow

## System Overview

This project is a **server-side antenna optimization orchestrator**.
It accepts antenna requirements, predicts a starting geometry with an ANN, returns a safe CST command package, and processes CST simulation feedback for iterative refinement.

---

## Main Components

### Entry Layer
- `main.py` — starts the API server
- `server.py` — FastAPI app, endpoints, health, chat, WebSocket streaming

### Orchestration Layer
- `central_brain.py` — main workflow coordinator
- handles:
  - request normalization
  - intent summarization
  - ANN prediction
  - surrogate validation
  - command package generation
  - feedback refinement loop

### Model / Intelligence Layer
- `app/ann/` — ANN predictor and training logic
- `app/llm/` — LLM-based intent parsing and refinement ranking through Ollama
- `app/planning/` — rule-based and LLM-assisted planning / refinement decisions

### Core Contracts / State
- `app/core/schemas.py` — Pydantic models
- `schemas/` — JSON schema contracts used for validation
- `app/core/session_store.py` — persists optimization sessions in `sessions/`

### Data / Training Assets
- `data/` — raw, validated, rejected datasets
- `models/ann/` — ANN checkpoint and metadata
- `scripts/` — dataset generation, validation, training, session replay

---

## Runtime Workflow

### 1. Server Startup
On startup the server:
- initializes the FastAPI app
- warms the **ANN**
- warms the **LLM** via Ollama
- exposes readiness through `GET /api/v1/health`

Health states:
- `available`
- `loading`
- `none`

---

### 2. Optional Chat / Intent Parsing
The client may first call:
- `POST /api/v1/chat`
- `POST /api/v1/intent/parse`

Purpose:
- extract `frequency_ghz`
- extract `bandwidth_mhz`
- extract `antenna_family`
- guide the user before real optimization starts

---

### 3. Optimization Request
The main pipeline starts with:
- `POST /api/v1/optimize`

The request is validated against:
- `schemas/http/optimize_request.v1.json`

Important required sections:
- `target_spec`
- `design_constraints`
- `optimization_policy`
- `runtime_preferences`
- `client_capabilities`

---

### 4. ANN + Safety Gate
Inside `central_brain.py` the flow is:

1. normalize request and family profile
2. summarize intent
3. run ANN prediction
4. run surrogate validation
5. decide whether to:
   - proceed,
   - return clarification,
   - or return an error

---

### 5. Command Package Creation
If accepted, the server returns a `command_package` containing safe, high-level CST actions such as:
- `create_project`
- `set_units`
- `define_material`
- `create_substrate`
- `create_patch`
- `create_feedline`
- `run_simulation`
- `export_s_parameters`
- `extract_summary_metrics`

The server does **not** execute CST itself.
The **client side** must translate these commands into local CST automation.

---

### 6. Session Lifecycle
A session is created when `/api/v1/optimize` succeeds.
Each session stores:
- `session_id`
- `trace_id`
- request payload
- current ANN prediction
- current command package
- iteration history
- planning decisions

Sessions are persisted as JSON files under `sessions/`.

---

### 7. Feedback / Refinement Loop
After the client executes CST locally, it sends:
- `POST /api/v1/client-feedback`

The server then:
1. evaluates the simulation metrics
2. checks acceptance criteria
3. if not accepted, chooses a refinement strategy
4. emits a new command package for the next iteration

This continues until:
- the design is accepted, or
- max iterations are reached, or
- policy stops the run

---

### 8. Live Session Monitoring
The client can subscribe to:
- `WS /api/v1/sessions/{session_id}/stream`

Typical emitted events include:
- `iteration.completed`
- `session.completed`
- `session.failed`

---

## High-Level Flow Diagram

```text
Client UI / CST Client
    |
    |  chat / intent parse (optional)
    v
FastAPI Server (`server.py`)
    |
    v
Central Brain (`central_brain.py`)
    |
    +--> ANN prediction (`app/ann/`)
    +--> LLM intent/refinement (`app/llm/`)
    +--> surrogate validation (`app/core/`)
    +--> command planning (`app/commands/`, `app/planning/`)
    |
    v
Command Package returned to client
    |
    v
Client executes CST locally
    |
    v
Client feedback posted back to server
    |
    v
Accept / Refine / Stop
```

---

## Key Rule for Client Integration

> The backend owns the **optimization logic and safe command planning**.
> The client owns the **actual CST execution and artifact export**.
