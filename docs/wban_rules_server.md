# WBAN Rules — Server Reference

## Scope

This file adapts the client WBAN guidance for the server-side design and ANN path. The server now focuses on:

- body-aware geometry priors
- detuning compensation before CST
- returning slot / notch / body parameters to the client
- validating WBAN feedback rows for future retraining

---

## Server-side WBAN rules

For `wban_patch`, the server intentionally designs slightly above the requested frequency to compensate for on-body loading:

- `f_design = f_target × (1.03–1.10)`
- smaller body distance → stronger detuning compensation
- tighter bending radius → stronger detuning compensation

The server also recommends ground-slot and notch geometry using bounded ratios of the radiating patch size.

Typical ranges encoded in the recipe:

- `body_distance_mm`: roughly `2–10 mm`
- `bending_radius_mm`: roughly `30–100 mm`
- `ground_slot_length_mm`: about `0.20–0.50 L`
- `ground_slot_width_mm`: about `0.02–0.10 W`
- `notch_length_mm`: about `0.05–0.20 L`
- `notch_width_mm`: about `0.01–0.06 W`

---

## Parameters returned to the client

Standard dimensions come back in `ann_prediction.dimensions`. WBAN-specific values come back in `ann_prediction.family_parameters`:

| Field | Meaning |
| --- | --- |
| `body_distance_mm` | assumed separation from body / phantom |
| `bending_radius_mm` | assumed curvature radius |
| `design_frequency_ghz` | pre-detuned target frequency used by the server |
| `ground_slot_length_mm` | recommended slot length |
| `ground_slot_width_mm` | recommended slot width |
| `notch_length_mm` | notch length |
| `notch_width_mm` | notch width |
| `detuning_compensation_ratio` | multiplier applied to the requested target frequency |

---

## Runtime ANN status

The WBAN family ANN is now routed in the optimize path and loaded from:

- `models/ann/wban_patch_v1/inverse_ann.pt`
- `models/ann/wban_patch_v1/metadata.json`

The model predicts the body-aware geometry refinements while the server recipe preserves a safe analytical fallback.

---

## Feedback files

- raw: `data/raw/wban_patch_feedback_v1.csv`
- validated: `data/validated/wban_patch_feedback_validated_v1.csv`
- rejected: `data/rejected/wban_patch_feedback_rejected_v1.csv`

Once enough valid WBAN rows accumulate, the server can now retrain the WBAN family ANN automatically and reload it live.

Schema contract: `schemas/data/wban_patch_feedback.v1.json`
