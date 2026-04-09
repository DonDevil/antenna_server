# AMC Rules — Server Reference

## Scope

This file adapts the client AMC rules for the server-side flow. The server does **not** build CST commands anymore; it now focuses on:

- generating AMC-aware geometry priors
- returning AMC lattice parameters to the client
- storing AMC feedback for future retraining

---

## Server-side design rules

For `amc_patch`, the server keeps the radiating patch rectangular and computes AMC guidance from wavelength-scaled rules:

- unit-cell period `p ≈ 0.15–0.30 λ0`
- AMC patch size `a ≈ 0.60–0.90 p`
- gap `g = p - a`
- via radius ≈ `0.02–0.05 p`
- air gap ≈ `0.02–0.08 λ0`
- reflection phase target ≈ `0°` near the requested frequency

The server also expands the ground size to fit the recommended AMC array footprint.

---

## Parameters returned to the client

Standard patch geometry is still returned in `ann_prediction.dimensions`, and AMC-specific values are returned in `ann_prediction.family_parameters`:

| Field | Meaning |
| --- | --- |
| `amc_unit_cell_period_mm` | AMC lattice period |
| `amc_patch_size_mm` | conductive patch side within the unit cell |
| `amc_gap_mm` | separation between adjacent AMC cells |
| `amc_via_radius_mm` | via radius for mushroom-style AMC realizations |
| `amc_via_height_mm` | via height, usually tied to substrate height |
| `amc_ground_size_mm` | recommended total ground size |
| `amc_array_rows` | recommended row count |
| `amc_array_cols` | recommended column count |
| `amc_air_gap_mm` | air-gap recommendation |
| `reflection_phase_target_deg` | target AMC reflection phase center |

---

## Runtime ANN status

The AMC family ANN is now routed in the optimize path and loaded from:

- `models/ann/amc_patch_v1/inverse_ann.pt`
- `models/ann/amc_patch_v1/metadata.json`

The model predicts the AMC lattice terms while the server recipe preserves safe baseline patch dimensions.

---

## Feedback files

- raw: `data/raw/amc_patch_feedback_v1.csv`
- validated: `data/validated/amc_patch_feedback_validated_v1.csv`
- rejected: `data/rejected/amc_patch_feedback_rejected_v1.csv`

Once enough valid AMC rows accumulate, the server can now retrain the AMC family ANN automatically and reload it live.

Schema contract: `schemas/data/amc_patch_feedback.v1.json`
