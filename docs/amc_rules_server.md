# AMC Rules — Server Reference

## Scope

This file adapts the client AMC rules for the server-side flow. The server now focuses on:

- generating the base rectangular patch dimensions
- asking the client to build the AMC locally through `implement_amc`
- storing AMC feedback for reference only

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

In the command planner, the server now emits a single `implement_amc` command after the patch/feed geometry is defined. The updated client expands that command locally using its own AMC heuristic, which currently performs better than the retired server-side AMC ANN output.

---

## Parameters returned to the client

For `amc_patch`, the server now returns the standard patch geometry in `ann_prediction.dimensions` and keeps `ann_prediction.family_parameters` empty. The AMC reflector itself is requested through the extra `implement_amc` command in the command package.

---

## Runtime ANN status

The AMC family ANN is no longer used in the optimize path. AMC now follows a **recipe + client-local implementation** path:

- the server sizes the radiating patch safely
- the command package includes `implement_amc`
- the client expands the AMC reflector using its local implementation

---

## Feedback files

- raw: `data/raw/amc_patch_feedback_v1.csv`
- validated: `data/validated/amc_patch_feedback_validated_v1.csv`
- rejected: `data/rejected/amc_patch_feedback_rejected_v1.csv`

AMC feedback is still stored in the family CSVs, but automatic AMC ANN retraining is disabled because the client-local AMC implementation is now the preferred path.

Schema contract: `schemas/data/amc_patch_feedback.v1.json`
