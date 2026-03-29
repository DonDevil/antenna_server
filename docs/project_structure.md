# Project Structure (Current)

```text
antenna_server/
  app/
    ann/
      baseline.py
      model.py
      predictor.py
      trainer.py
    commands/
      planner.py
    core/
      exceptions.py
      family_registry.py
      json_contracts.py
      refinement.py
      schemas.py
      session_store.py
    data/
      schema.py
      store.py
      validator.py
  context_files/
    antenna_design_bounds.md
    command_catalog.md
    examples_e2e.md
    feedback_examples.md
    safety_guardrails.md
    system_prompt.md
  data/
    raw/
      dataset.csv
    validated/
      dataset_validated.csv
    rejected/
      dataset_rejected.csv
  docs/
    api_and_command_contracts.md
    plan_refinements_from_previous_project.md
    project_structure.md
  models/
    ann/
      v1/
        inverse_ann.pt
        metadata.json
  schemas/
    commands/
      cst_command_package.v1.json
    http/
      client_feedback.v1.json
      optimize_request.v1.json
      optimize_response.v1.json
    ws/
      session_event.v1.json
  scripts/
    generate_synthetic_dataset.py
    generate_sysnthetic_dataset.py  # typo-compat wrapper
    replay_session.py
    train_ann.py
    validate_dataset.py
  sessions/
    <session_id>.json
  tests/
    conftest.py
    integration/
      test_iteration_flow.py
      test_ws_stream.py
    unit/
      test_dataset_validator.py
  central_brain.py
  config.py
  main.py
  server.py
  requirements.txt
  README.md
```

## Runtime Flow

1. `scripts/generate_synthetic_dataset.py` creates initial training samples.
2. `scripts/validate_dataset.py` enforces schema and bounds, writing valid/rejected splits.
3. `scripts/train_ann.py` trains inverse ANN and writes checkpoint + metadata.
4. `server.py` validates HTTP/WS contracts and routes requests.
5. `central_brain.py` performs ANN prediction, command planning, iterative refinement, and session updates.
6. `app/core/session_store.py` persists per-session history to `sessions/`.
7. `scripts/replay_session.py` reconstructs an auditable iteration timeline from stored session JSON.
8. `app/core/family_registry.py` applies antenna-family profile defaults and enforces family-specific material/substrate constraints before planning.

## Notes

- `sessions/`, `data/`, and model artifacts are runtime/generated outputs and should not be treated as source code.
- WebSocket events are validated against `schemas/ws/session_event.v1.json` before emission.
- Family profile enforcement is centralized in `app/core/family_registry.py` and surfaced to API clients as schema-safe 422 errors.
