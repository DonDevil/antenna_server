# Project Structure (From Scratch)

```text
antenna_server/
  app/
    ann/
      model.py
      trainer.py
      predictor.py
      baseline.py
    commands/
      planner.py
    core/
      schemas.py
      exceptions.py
    data/
      schema.py
      validator.py
      store.py
  scripts/
    generate_synthetic_dataset.py
    validate_dataset.py
    train_ann.py
  context_files/
    system_prompt.md
    safety_guardrails.md
    antenna_design_bounds.md
    command_catalog.md
    examples_e2e.md
    feedback_examples.md
  models/
    ann/
      v1/
        inverse_ann.pt
        metadata.json
  data/
    raw/
    validated/
    rejected/
  schemas/
    http/
    ws/
    commands/
  server.py
  central_brain.py
  config.py
  main.py
```

## Data Flow

1. `scripts/generate_synthetic_dataset.py` builds a seed dataset.
2. `scripts/validate_dataset.py` filters invalid rows and stores rejected rows with reason.
3. `scripts/train_ann.py` trains the inverse ANN and writes checkpoint plus metadata.
4. `server.py` receives optimize requests and delegates to `central_brain.py`.
5. `central_brain.py` predicts dimensions and emits a whitelisted command package.
