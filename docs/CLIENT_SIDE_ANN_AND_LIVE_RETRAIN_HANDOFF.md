# Client-Side Copilot Handoff: Server ANN Status + Live CST Retrain

**Date:** 2026-04-09

## Short answers

- Keep using `POST /api/v1/optimize`.
- CST results can be posted to either `POST /api/v1/client-feedback` or `POST /api/v1/result`.
- There is still **no** `/api/v2/...` HTTP endpoint.
- The CST command package remains `cst_command_package.v2`.
- Rectangular microstrip and WBAN family ANNs are still routed in the optimize path.
- AMC now uses a **client-local** `implement_amc` command instead of server-side AMC ANN prediction.
- Automatic live retraining remains active for **rectangular microstrip and WBAN**; AMC feedback is stored but not retrained.

---

## Current server behavior

The server flow is still:

1. Client sends `POST /api/v1/optimize`.
2. `CentralBrain.optimize(...)` runs the recipe + ANN flow.
3. `AnnPredictor.predict(...)` selects the best family model.
4. The server returns `optimize_response.v1` with `ann_prediction`, `command_package`, `session_id`, and `trace_id`.

### Runtime models now wired

#### Rectangular microstrip

Used for:

- `antenna_family = microstrip_patch`
- `patch_shape = rectangular`
- `feed_type = edge` or `auto`

Artifacts:

- `models/ann/rect_patch_v1/inverse_ann.pt`
- `models/ann/rect_patch_v1/metadata.json`

#### AMC

Used for:

- `antenna_family = amc_patch`

Behavior:

- standard patch geometry still comes back in `ann_prediction.dimensions`
- `ann_prediction.family_parameters` is intentionally empty for AMC
- the command package includes `implement_amc`, which the updated client expands locally using its own AMC heuristic

#### WBAN

Used for:

- `antenna_family = wban_patch`

Artifacts:

- `models/ann/wban_patch_v1/inverse_ann.pt`
- `models/ann/wban_patch_v1/metadata.json`

Returned values:

- standard patch/substrate/feed values in `ann_prediction.dimensions`
- body-aware slot/notch values in `ann_prediction.family_parameters`
- WBAN guide parameters also mirrored into CST parameter definitions inside `command_package.commands`

---

## Endpoints to use

- `POST /api/v1/optimize` — start or resume a design session
- `POST /api/v1/client-feedback` — send CST feedback through the standard session flow
- `POST /api/v1/result` — product-facing alias for the same result-ingest logic
- `GET /api/v1/sessions/{session_id}` — inspect current session state
- `WS /api/v1/sessions/{session_id}/stream` — receive live updates

> The API stays under `/api/v1/...`; only the command package is already V2.

---

## What the client should post

### 1) Optimization request

Use `optimize_request.v1` as before.

Set `target_spec.antenna_family` to one of:

- `microstrip_patch`
- `amc_patch` (uses `implement_amc` in the returned command package)
- `wban_patch`

Optional family-aware hints can now be supplied through `design_constraints`:

```json
{
  "amc_air_gap_mm": { "min": 2.0, "max": 4.0 },
  "body_distance_mm": { "min": 4.0, "max": 6.0 },
  "bending_radius_mm": { "min": 50.0, "max": 70.0 }
}
```

### 2) CST feedback

After the client executes the returned command package, post `client_feedback.v1` with the usual fields:

- `session_id`
- `trace_id`
- `design_id`
- `iteration_index`
- `simulation_status`
- `actual_center_frequency_ghz`
- `actual_bandwidth_mhz`
- `actual_return_loss_db`
- `actual_vswr`
- `actual_gain_dbi`
- `artifacts.s11_trace_ref`
- `artifacts.summary_metrics_ref`

If the user explicitly finishes the run, the client can also send:

```json
"completion_requested": true
```

---

## What parameters the server passes back now

### Common fields for all families

Returned in `ann_prediction.dimensions`:

- `patch_length_mm`
- `patch_width_mm`
- `patch_height_mm`
- `patch_radius_mm`
- `substrate_length_mm`
- `substrate_width_mm`
- `substrate_height_mm`
- `feed_length_mm`
- `feed_width_mm`
- `feed_offset_x_mm`
- `feed_offset_y_mm`

### AMC-only fields

Returned in `ann_prediction.family_parameters`:

- `amc_unit_cell_period_mm`
- `amc_patch_size_mm`
- `amc_gap_mm`
- `amc_via_radius_mm`
- `amc_via_height_mm`
- `amc_ground_size_mm`
- `amc_array_rows`
- `amc_array_cols`
- `amc_air_gap_mm`
- `reflection_phase_target_deg`

### WBAN-only fields

Returned in `ann_prediction.family_parameters`:

- `body_distance_mm`
- `bending_radius_mm`
- `design_frequency_ghz`
- `ground_slot_length_mm`
- `ground_slot_width_mm`
- `notch_length_mm`
- `notch_width_mm`
- `detuning_compensation_ratio`

These values are the server-to-client contract for family-specific CST generation on the client side. For AMC trial runs, the server now also emits the reflector cell array directly in the command package, which matches the client’s `run_amc_pipeline_once` workflow more closely.

---

## Live retraining status

### What is automatic today

For **rectangular microstrip, AMC, and WBAN** feedback:

- results are logged automatically from `POST /api/v1/result` and `POST /api/v1/client-feedback`
- validated rows are written into the family-specific feedback pipeline
- a background retrain is triggered after **50 new valid rows** for that family
- the server reloads the ANN artifacts live without an API restart

### Manual retraining commands

Generate and train all three families:

```bash
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family all --rows 15000 --seed 42
```

Train from existing CSVs only:

```bash
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family all --skip-generate
```

### Family-specific feedback files

- `data/raw/rect_patch_feedback_v1.csv`
- `data/raw/amc_patch_feedback_v1.csv`
- `data/raw/wban_patch_feedback_v1.csv`

Matching schema/docs now exist for AMC and WBAN as well.

### Ledger deduplication

The shared ledger at `data/raw/live_results_v1.jsonl` now deduplicates repeated ingests for the same CST result key.

---

## What is still not fully automatic

The remaining gap is no longer the retrain trigger itself; the current improvement target is **collecting enough real CST rows to keep refining AMC/WBAN quality over time**.

The current status is:

- rect runtime prediction = active
- AMC runtime prediction = active
- WBAN runtime prediction = active
- rect live retrain = active
- AMC live retrain = active
- WBAN live retrain = active

---

## Recommended client action

1. Keep the same endpoints.
2. Keep posting `optimize_request.v1` and `client_feedback.v1`.
3. Use `target_spec.antenna_family` to select `microstrip_patch`, `amc_patch`, or `wban_patch`.
4. Use the returned `ann_prediction.dimensions` plus `ann_prediction.family_parameters` to build client-side CST commands.
5. Keep sending result artifacts (`s11_trace_ref`, `summary_metrics_ref`) so the server can store and deduplicate feedback correctly.

---

## One-line summary

**Use `/api/v1/optimize` for design generation and `/api/v1/result` (or `/api/v1/client-feedback`) for CST results; rect, AMC, and WBAN family ANNs are routed in the optimize path, and all three families now support automatic live retraining after enough validated feedback rows are collected.**
