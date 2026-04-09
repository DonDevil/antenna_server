# Rectangular Patch Rules — Server Reference

## Scope

This file mirrors the client-side rectangular patch rules, but only for the **server responsibilities**:

- compute a stable first-pass geometry
- provide ANN-backed dimensions to the client
- validate CST feedback for retraining

The active implementation lives in `app/antenna/recipes.py` and `app/ann/predictor.py`.

---

## Baseline formulas used by the server

For `microstrip_patch` the server seeds the patch using standard transmission-line equations:

- `W = c / (2 f0) * sqrt(2 / (εr + 1))`
- `εeff = ((εr + 1)/2) + ((εr - 1)/2) / sqrt(1 + 12 h / W)`
- `ΔL = 0.412 h * ((εeff + 0.3)(W/h + 0.264)) / ((εeff - 0.258)(W/h + 0.8))`
- `Leff = c / (2 f0 sqrt(εeff))`
- `L = Leff - 2ΔL`

The server also applies practical heuristics:

- substrate outline ≈ patch size `+ max(6h, 8 mm)`
- `feed_width_mm` starts near a 50 Ω microstrip estimate
- `feed_offset_y_mm` is seeded near `0.28 L`

---

## Parameters returned to the client

These values are returned in `ann_prediction.dimensions`:

| Field | Meaning |
| --- | --- |
| `patch_length_mm` | radiating patch length |
| `patch_width_mm` | radiating patch width |
| `patch_height_mm` | copper thickness / patch thickness |
| `patch_radius_mm` | compatibility field for downstream circular-capable tooling |
| `substrate_length_mm` | substrate Y size |
| `substrate_width_mm` | substrate X size |
| `substrate_height_mm` | dielectric thickness |
| `feed_length_mm` | feedline length |
| `feed_width_mm` | feedline width |
| `feed_offset_x_mm` | feed offset X |
| `feed_offset_y_mm` | feed offset Y |

For rectangular microstrip, `ann_prediction.family_parameters` is typically empty because the standard dimensions are already sufficient.

---

## ANN status

The runtime family model for this path is:

- `models/ann/rect_patch_v1/inverse_ann.pt`
- `models/ann/rect_patch_v1/metadata.json`

If the ANN is unavailable, the server falls back to the analytical recipe and still returns a usable geometry.

---

## Feedback / retraining files

- raw: `data/raw/rect_patch_feedback_v1.csv`
- validated: `data/validated/rect_patch_feedback_validated_v1.csv`
- rejected: `data/rejected/rect_patch_feedback_rejected_v1.csv`

These are the live files used by the current online rectangular retraining flow.
