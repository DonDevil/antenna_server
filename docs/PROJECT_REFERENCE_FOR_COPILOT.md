# Project Reference for Copilot

This file is the **single project reference** for any other Copilot instance working on the client side or on future server-side changes.

---

## Project Purpose

`antenna_server` is a backend service for **antenna design orchestration**.
It does not run CST directly. It:

- parses requirements
- predicts initial antenna geometry with an ANN
- returns a high-level CST command package
- receives simulation feedback
- plans the next refinement iteration if needed

---

## Repository Map

### Top-level files
- `main.py` — launch entry
- `server.py` — FastAPI routes
- `central_brain.py` — main orchestration logic
- `config.py` — settings and paths
- `requirements.txt` — Python dependencies

### Main packages
- `app/ann/` — ANN loading, inference, training support
- `app/llm/` — LLM / Ollama integrations
- `app/commands/` — command package planning
- `app/planning/` — rule-based and LLM-assisted refinement
- `app/core/` — schemas, validation, sessions, errors, policy helpers
- `schemas/` — JSON contracts for HTTP, WS, command packages
- `tests/` — integration and unit coverage

### Data folders
- `data/` — datasets
- `models/` — trained ANN checkpoint and metadata
- `sessions/` — persisted server sessions
- `context_files/` — prompt/context assets used by the project

---

## API Endpoints

| Method | Endpoint | Use |
|---|---|---|
| `GET` | `/api/v1/health` | check server, ANN, and LLM readiness |
| `GET` | `/api/v1/capabilities` | inspect supported capabilities and ranges |
| `POST` | `/api/v1/intent/parse` | parse natural language to structured intent |
| `POST` | `/api/v1/chat` | conversational requirements capture |
| `POST` | `/api/v1/optimize` | start or resume optimization |
| `POST` | `/api/v1/client-feedback` | send CST simulation results back |
| `GET` | `/api/v1/sessions/{session_id}` | inspect session state |
| `WS` | `/api/v1/sessions/{session_id}/stream` | stream session events |

---

## Contract Notes That Must Be Respected

### Optimize request
The request must match:
- `schemas/http/optimize_request.v1.json`

Important required nested fields:

```json
{
  "optimization_policy": {
    "fallback_behavior": "best_effort"
  },
  "runtime_preferences": {
    "priority": "normal"
  }
}
```

These are easy to miss and will cause **422 validation errors**.

### Feedback request
The feedback must match:
- `schemas/http/client_feedback.v1.json`

Important required artifact references:
- `artifacts.s11_trace_ref`
- `artifacts.summary_metrics_ref`

---

## Server Workflow Summary

```text
health -> optional chat/intent parse -> optimize -> session created
-> ANN prediction -> surrogate validation -> command package returned
-> client executes CST -> client-feedback -> accept/refine/stop
```

---

## Client-Side Integration Rules

Any Copilot building the client side should follow these rules:

1. Treat `/optimize` as the **real session creation step**.
2. Store and reuse:
   - `session_id`
   - `trace_id`
   - `design_id`
3. Execute `command_package.commands` strictly in order.
4. Translate server commands into **local CST automation**.
5. Always send feedback after each CST run.
6. Support model health states:
   - `available`
   - `loading`
   - `none`
7. Handle `clarification_required` and `error` responses gracefully.

---

## Important Design Boundary

### Backend owns
- optimization logic
- ANN inference
- refinement strategy selection
- safe CST command planning
- session persistence

### Client owns
- GUI / chat UX
- CST execution
- artifact export
- local progress display
- sending feedback back to the server

---

## Key Files to Read First

If another Copilot needs to understand this codebase quickly, start with:

1. `server.py`
2. `central_brain.py`
3. `app/core/schemas.py`
4. `schemas/http/optimize_request.v1.json`
5. `app/commands/planner.py`
6. `tests/integration/test_iteration_flow.py`
7. `tests/integration/test_surrogate_policy.py`

---

## Testing Reference

The verified workflow tests are:

```bash
python -m pytest tests/integration/test_iteration_flow.py tests/integration/test_surrogate_policy.py -q
```

These confirm that:
- optimize requests can succeed
- feedback refinement works
- sessions are persisted correctly

---

## Final Guidance for Other Copilot Instances

> If you are building the client side, use the server contracts exactly as written in the JSON schemas and the integration tests. Do not infer or simplify missing fields on your own.
