# Client-Side Copilot Handoff: Server ANN Status + Live CST Retrain

_Date: 2026-04-09_

## Short answers

- **Yes, keep using the same server API endpoints**:
  - `POST /api/v1/optimize`
  - `POST /api/v1/client-feedback`
- **No new `/api/v2/...` endpoint was added.**
- The **returned CST command package is already V2** (`cst_command_package.v2`), but the HTTP API stays under `/api/v1/...`.
- **Rectangular microstrip ANN is wired into the server optimize path.**
- **AMC and WBAN family-specific ANNs are trained on disk, but they are not yet fully routed as production family-specific predictors in the server response path.**
- **Live CST retraining is not automatic yet**; right now it is **manual / batch retraining** using the same endpoint data and training scripts.

---

## Current server behavior

The server still uses the same flow:

1. Client sends `POST /api/v1/optimize`
2. Server runs `CentralBrain.optimize(...)`
3. `CentralBrain` calls `AnnPredictor.predict(...)`
4. Server returns the same `optimize_response.v1` contract with:
   - `ann_prediction`
   - `command_package`
   - `session_id`
   - `trace_id`

### What is already wired

#### 1) Rectangular microstrip patch ANN
This one **is wired**.

When the request is:
- `antenna_family = microstrip_patch`
- `patch_shape = rectangular`
- `feed_type = edge` (or `auto`)

the server prefers the family-specific model stored at:

- `models/ann/rect_patch_v1/inverse_ann.pt`
- `models/ann/rect_patch_v1/metadata.json`

So for the standard rectangular patch flow, the server is already using the new ANN inside the **same `/api/v1/optimize` endpoint**.

#### 2) AMC + WBAN family ANNs
These models **have been trained and saved**:

- `models/ann/amc_patch_v1/`
- `models/ann/wban_patch_v1/`

But they are **not yet fully exposed as family-specific production predictors** in the optimize-response path.

Why not fully wired yet?
- The current HTTP response schema (`AnnPrediction -> DimensionPrediction`) is still mostly **patch-geometry oriented**.
- AMC needs family-specific fields like:
  - `amc_unit_cell_period_mm`
  - `amc_patch_size_mm`
  - `amc_gap_mm`
  - `amc_via_radius_mm`
  - `amc_air_gap_mm`
- WBAN needs extra body-aware geometry / labels like:
  - `body_distance_mm`
  - `bending_radius_mm`
  - `ground_slot_length_mm`
  - `ground_slot_width_mm`

So today:
- **rect ANN = actively wired**
- **AMC/WBAN ANNs = trained and ready, but not yet fully promoted into the production server output contract**

---

## Do we need a new endpoint?

**No.**

The client should continue to use:

- `POST /api/v1/optimize` to start or resume a design session
- `POST /api/v1/client-feedback` to send CST results back
- `GET /api/v1/sessions/{session_id}` to inspect current session state
- `WS /api/v1/sessions/{session_id}/stream` for live updates

There is **no `/api/v2/optimize` or `/api/v2/client-feedback`**.

> Important distinction: the **API stays V1**, but the **returned CST command package already uses the V2 command contract**.

---

## What the client should post now

## 1) Start optimization

Use the existing optimize contract:

```json
{
  "schema_version": "optimize_request.v1",
  "user_request": "Design a rectangular microstrip patch antenna at 2.45 GHz with 100 MHz bandwidth.",
  "target_spec": {
    "frequency_ghz": 2.45,
    "bandwidth_mhz": 100.0,
    "antenna_family": "microstrip_patch",
    "patch_shape": "rectangular",
    "feed_type": "edge",
    "polarization": "linear"
  },
  "design_constraints": {
    "allowed_materials": ["Copper (annealed)"],
    "allowed_substrates": ["Rogers RT/duroid 5880"]
  },
  "optimization_policy": {
    "mode": "auto_iterate",
    "max_iterations": 5,
    "stop_on_first_valid": true,
    "acceptance": {
      "center_tolerance_mhz": 20.0,
      "minimum_bandwidth_mhz": 100.0,
      "maximum_vswr": 2.0,
      "minimum_gain_dbi": 4.0,
      "minimum_return_loss_db": -10.0
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
    "export_formats": ["json", "csv", "txt"]
  }
}
```

### For AMC requests
Only change the family and request text, for example:

```json
"antenna_family": "amc_patch"
```

### For WBAN requests
Use:

```json
"antenna_family": "wban_patch"
```

---

## 2) Send CST feedback back to the server

After the client executes the returned CST `command_package`, it should post the same feedback contract:

```json
{
  "schema_version": "client_feedback.v1",
  "session_id": "<session_id>",
  "trace_id": "<trace_id>",
  "design_id": "design_<session_id>",
  "iteration_index": 0,
  "simulation_status": "completed",
  "actual_center_frequency_ghz": 2.44,
  "actual_bandwidth_mhz": 92.0,
  "actual_return_loss_db": -18.5,
  "actual_vswr": 1.45,
  "actual_gain_dbi": 4.8,
  "actual_efficiency": 0.78,
  "actual_axial_ratio_db": 28.0,
  "actual_front_to_back_db": 14.0,
  "notes": "Initial CST run completed.",
  "artifacts": {
    "s11_trace_ref": "artifacts/s11_iter0.json",
    "summary_metrics_ref": "artifacts/summary_iter0.json",
    "farfield_ref": "artifacts/farfield_iter0.json",
    "current_distribution_ref": null
  }
}
```

### Optional: mark session complete explicitly
If the user clicks **Done** / **Finish**, the client can add:

```json
"completion_requested": true
```

That is already supported by the existing `POST /api/v1/client-feedback` route.

---

## How to do live CST retrain

## Current state: manual / batch retrain

Right now, live retraining is **not automatic on every feedback call**.

What exists today:
- training scripts
- 15k bootstrap datasets
- saved family ANN checkpoints
- rect-patch client calibration merge

### Manual retrain commands

Generate / refresh datasets and train all three:

```bash
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family all --rows 15000 --seed 42
```

Train from the **current CSVs only** without regenerating synthetic data:

```bash
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family all --skip-generate
```

Family-by-family:

```bash
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family microstrip_patch --skip-generate
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family amc_patch --skip-generate
/home/darkdevil/Desktop/antenna_server/env/bin/python scripts/train_family_anns.py --family wban_patch --skip-generate
```

---

## Recommended “live retrain” workflow

If you want practical live retraining with CST runs, do this:

### A. Keep the same client API calls
- Client **does not need a new endpoint**
- Keep posting to `/api/v1/client-feedback`

### B. After each successful CST run, log a training row
For rectangular patch, a logging helper already exists:

- `app/data/rect_patch_feedback_logger.py`

That can write rows into:

- `data/raw/rect_patch_feedback_v1.csv`

### C. For AMC and WBAN, add the same kind of family logger
Recommended files to add later:

- `data/raw/amc_patch_feedback_v1.csv`
- `data/raw/wban_patch_feedback_v1.csv`

### D. Retrain after every small batch of new accepted rows
Recommended thresholds:
- **Rect patch:** retrain after every `25-50` new accepted runs
- **AMC / WBAN:** retrain after every `20-40` new accepted runs

### E. Swap the model in place
The server already loads from model files on disk, so retraining can overwrite the existing family checkpoint/metadata paths.

---

## What is still missing for fully automatic live retrain

To make it truly automatic, the server still needs one more wiring step:

1. Inside `brain.process_feedback(...)`, after a successful CST result:
   - derive a family-specific training row from:
     - original request
     - current ANN / geometry
     - feedback metrics
     - artifact refs
2. append that row to the family CSV
3. if `new_rows_since_last_train >= threshold`, start a background retrain job
4. atomically replace the model files on success
5. keep the same `/api/v1/optimize` endpoint

### In other words
- **same API**
- **same client contract**
- **background retrain job on the server**
- **no new endpoint needed**

---

## Final status for the client-side Copilot

### Use now
- `POST /api/v1/optimize`
- `POST /api/v1/client-feedback`
- same schema versions
- same session flow

### Already working now
- rect-patch ANN in the optimize path
- command package V2
- family-aware request field: `antenna_family`
- batch retraining scripts for all 3 families

### Not yet fully automatic
- AMC and WBAN family-specific ANN routing into the production response path
- automatic live retrain trigger on every new CST feedback batch

---

## Recommended client-side action right now

Client-side Copilot should:

1. **Keep the same endpoints**
2. **Keep posting `optimize_request.v1` and `client_feedback.v1`**
3. Always include:
   - `session_id`
   - `trace_id`
   - `design_id`
   - `iteration_index`
   - `artifacts.s11_trace_ref`
   - `artifacts.summary_metrics_ref`
4. Set:
   - `target_spec.antenna_family` to one of:
     - `microstrip_patch`
     - `amc_patch`
     - `wban_patch`
5. Use `completion_requested: true` when the user explicitly finishes the session

---

## One-line summary

**No new endpoint is needed: the client should keep using `/api/v1/optimize` and `/api/v1/client-feedback`; the rectangular-patch ANN is already wired, AMC/WBAN models are trained but not yet fully promoted in the live server path, and live CST retraining is currently batch/manual rather than fully automatic.**
