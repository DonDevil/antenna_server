# Installation and Usage

## Prerequisites

Make sure you have:
- Python `3.12+`
- the project virtual environment (`env/`)
- Ollama running locally
- the configured LLM model available: `deepseek-r1:8b`

---

## 1. Activate Environment

```bash
cd /home/darkdevil/Desktop/antenna_server
source env/bin/activate
```

Install dependencies if needed:

```bash
pip install -r requirements.txt
```

---

## 2. Optional Dataset / ANN Preparation

Generate synthetic data:

```bash
python scripts/generate_synthetic_dataset.py
```

Validate dataset:

```bash
python scripts/validate_dataset.py
```

Train the ANN:

```bash
python scripts/train_ann.py
```

This should populate the ANN model files under:
- `models/ann/v1/inverse_ann.pt`
- `models/ann/v1/metadata.json`

---

## 3. Start the Server

```bash
python main.py
```

On startup the server will warm:
- the ANN
- the LLM

The console should print readiness messages.

---

## 4. Check Server Health

```bash
curl http://localhost:8000/api/v1/health
```

Example response:

```json
{
  "status": "ok",
  "ann_status": "available",
  "llm_status": "available"
}
```

Status meanings:
- `available` = ready
- `loading` = warming up
- `none` = unavailable / not reachable

---

## 5. Check Capabilities

```bash
curl http://localhost:8000/api/v1/capabilities
```

Use this to inspect supported:
- frequency ranges
- bandwidth ranges
- materials
- substrates
- antenna family options

---

## 6. Optional Chat / Intent Parsing

### Parse natural language intent

```bash
curl -X POST http://localhost:8000/api/v1/intent/parse \
  -H "Content-Type: application/json" \
  -d '{
    "user_request": "Design a microstrip patch antenna at 2.45 GHz with 100 MHz bandwidth"
  }'
```

### Chat endpoint

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want a patch antenna around 2.45 GHz.",
    "requirements": {}
  }'
```

---

## 7. Start an Optimization Session

Use `POST /api/v1/optimize`.

### Canonical working request

```json
{
  "schema_version": "optimize_request.v1",
  "user_request": "Design an AMC patch antenna with 2.45 GHz center frequency and 80 MHz bandwidth.",
  "target_spec": {
    "frequency_ghz": 2.45,
    "bandwidth_mhz": 80.0,
    "antenna_family": "amc_patch"
  },
  "design_constraints": {
    "allowed_materials": ["Copper (annealed)"],
    "allowed_substrates": ["FR-4 (lossy)"]
  },
  "optimization_policy": {
    "mode": "auto_iterate",
    "max_iterations": 3,
    "stop_on_first_valid": true,
    "acceptance": {
      "center_tolerance_mhz": 20.0,
      "minimum_bandwidth_mhz": 80.0,
      "maximum_vswr": 2.0,
      "minimum_gain_dbi": 5.0
    },
    "fallback_behavior": "best_effort"
  },
  "runtime_preferences": {
    "require_explanations": false,
    "persist_artifacts": true,
    "llm_temperature": 0.0,
    "timeout_budget_sec": 300,
    "priority": "normal"
  },
  "client_capabilities": {
    "supports_farfield_export": true,
    "supports_current_distribution_export": false,
    "supports_parameter_sweep": false,
    "max_simulation_timeout_sec": 600,
    "export_formats": ["json"]
  }
}
```

> Important: `fallback_behavior` and `runtime_preferences.priority` must be present for the schema to pass.

---

## 8. Send Client Feedback After CST Execution

After the client executes the returned `command_package`, send back simulation results:

```json
{
  "schema_version": "client_feedback.v1",
  "session_id": "<session-id>",
  "trace_id": "<trace-id>",
  "design_id": "design_<session-id>",
  "iteration_index": 0,
  "simulation_status": "completed",
  "actual_center_frequency_ghz": 2.44,
  "actual_bandwidth_mhz": 92.0,
  "actual_return_loss_db": -18.5,
  "actual_vswr": 1.45,
  "actual_gain_dbi": 4.8,
  "notes": "Initial CST run completed.",
  "artifacts": {
    "s11_trace_ref": "artifacts/s11_iter0.json",
    "summary_metrics_ref": "artifacts/summary_iter0.json",
    "farfield_ref": null,
    "current_distribution_ref": null
  }
}
```

Endpoint:

```bash
curl -X POST http://localhost:8000/api/v1/client-feedback \
  -H "Content-Type: application/json" \
  -d @feedback.json
```

---

## 9. Query an Existing Session

```bash
curl http://localhost:8000/api/v1/sessions/<session_id>
```

Use this to inspect:
- current iteration
- status
- stop reason
- last planning decision
- latest history entry

---

## 10. WebSocket Session Streaming

Connect to:

```text
ws://localhost:8000/api/v1/sessions/<session_id>/stream
```

Use this for live updates while the session is running.

---

## 11. Run Tests

Run the main integration tests:

```bash
python -m pytest tests/integration/test_iteration_flow.py tests/integration/test_surrogate_policy.py -q
```

Run health / chat tests:

```bash
python -m pytest tests/integration/test_capabilities_chat.py tests/integration/test_llm_live.py -q
```

---

## 12. Typical Usage Pattern

1. Start the server
2. Wait until `GET /health` shows models as available
3. Optionally use chat/intent parsing
4. Send `POST /optimize`
5. Execute returned CST command package on the client side
6. Send `POST /client-feedback`
7. Repeat until accepted or stopped

---

## 13. Troubleshooting

### 422 validation error on `/optimize`
Check for missing:
- `user_request`
- `design_constraints.allowed_materials`
- `design_constraints.allowed_substrates`
- `optimization_policy.fallback_behavior`
- `runtime_preferences.priority`

### LLM stuck in `loading` or `none`
Check:
- Ollama is running
- the configured model exists
- local network access to `http://localhost:11434`

### ANN unavailable
Check that these files exist:
- `models/ann/v1/inverse_ann.pt`
- `models/ann/v1/metadata.json`
