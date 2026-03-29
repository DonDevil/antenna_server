# API and Command Contracts v1

This document locks the first implementation boundary for the server-side AMC antenna optimization system.

## Design Intent

The server accepts natural-language antenna goals, converts them into validated structured intent, calls the ANN model for dimension prediction, emits only high-level whitelisted CST commands, and ingests CST simulation feedback for iterative refinement.

The LLM is not allowed to emit raw VBA, free-form CST code, or inferred values outside validated bounds. When the request is ambiguous or unsupported, the server must return an explicit clarification path or `IDK` outcome.

## Versioning Rules

- API version: `v1`
- Schema version field: `schema_version`
- Command catalog version field: `command_catalog_version`
- ANN model version field: `ann_model_version`
- Prompt/context version field: `prompt_bundle_version`

## Primary Endpoints

### `POST /api/v1/optimize`

Starts a new optimization session or resumes one if `session_id` is provided.

Headers:

- `Content-Type: application/json`
- `X-Trace-ID: <uuid>` optional, server generates if absent
- `X-Idempotency-Key: <string>` recommended for safe retries

Request body fields:

- `schema_version`: string, must be `optimize_request.v1`
- `session_id`: string or null, optional existing session UUID
- `user_request`: string, natural language target
- `target_spec`: object, exact desired antenna targets
- `design_constraints`: object, hard physical and material limits
- `optimization_policy`: object, loop and convergence settings
- `runtime_preferences`: object, execution controls for the session
- `client_capabilities`: object, what the CST client can execute/export
- `context`: object, optional prior design and experiment references

Success response fields:

- `schema_version`: `optimize_response.v1`
- `status`: `accepted` or `completed`
- `session_id`: UUID
- `trace_id`: UUID
- `current_stage`: enum pipeline stage
- `intent_summary`: object, normalized parse of the request
- `ann_prediction`: object or null
- `command_package`: object or null
- `simulation_feedback`: object or null
- `iteration_state`: object
- `artifacts`: object containing generated file paths or artifact identifiers
- `warnings`: array of strings

Error response fields:

- `schema_version`: `optimize_response.v1`
- `status`: `error`
- `trace_id`: UUID
- `error`: object with code, message, retryability, details

Validation error codes returned as HTTP 422 for optimize:

- `SCHEMA_VALIDATION_FAILED`
- `FAMILY_NOT_SUPPORTED`
- `FAMILY_PROFILE_CONSTRAINT_FAILED`

### `GET /api/v1/sessions/{session_id}`

Returns current state, latest results, active iteration, and artifact references.

### `GET /api/v1/health`

Returns status of:

- API service
- Ollama connectivity
- DeepSeek model availability
- ANN model availability
- writable artifact storage

### `WS /api/v1/sessions/{session_id}/stream`

Streams stage changes and iteration feedback.

Event types:

- `session.accepted`
- `stage.changed`
- `llm.intent.parsed`
- `ann.prediction.ready`
- `command.package.ready`
- `client.feedback.received`
- `iteration.completed`
- `session.completed`
- `session.failed`

## Pipeline Stages

- `received`
- `validating_request`
- `parsing_intent`
- `clarification_required`
- `ann_predicting`
- `planning_commands`
- `awaiting_client`
- `evaluating_feedback`
- `refining_design`
- `completed`
- `failed`

## Core Request Contract

## Family Registry Contract

Family handling is configuration-driven via server registry profiles, not hard-coded per endpoint.

Current registered families:

- `amc_patch`
- `microstrip_patch`
- `wban_patch`

Server behavior:

- Normalize requested `target_spec.antenna_family` to a registered profile.
- Apply profile defaults when generic defaults are supplied for materials/substrates.
- Reject unsupported families with `FAMILY_NOT_SUPPORTED`.
- Reject disallowed family constraints with `FAMILY_PROFILE_CONSTRAINT_FAILED`.
- Persist normalized request and family-driven decisions into session artifacts for replay.

### `target_spec`

- `frequency_ghz`: number, required
- `bandwidth_mhz`: number, required
- `antenna_family`: enum, optional, default `amc_patch`
- `center_tolerance_mhz`: number, optional
- `min_return_loss_db`: number, optional
- `max_vswr`: number, optional
- `target_gain_dbi`: number, optional
- `polarization`: enum, optional

### `design_constraints`

- `patch_length_mm`: min and max
- `patch_width_mm`: min and max
- `patch_height_mm`: min and max
- `substrate_length_mm`: min and max
- `substrate_width_mm`: min and max
- `substrate_height_mm`: min and max
- `feed_length_mm`: min and max
- `feed_width_mm`: min and max
- `allowed_materials`: array
- `allowed_substrates`: array
- `manufacturing_notes`: array of strings

### `optimization_policy`

- `mode`: `single_pass` or `auto_iterate`
- `max_iterations`: integer
- `stop_on_first_valid`: boolean
- `acceptance`: object of metric tolerances
- `fallback_behavior`: `best_effort`, `return_error`, or `require_user_confirmation`

### `runtime_preferences`

- `require_explanations`: boolean
- `persist_artifacts`: boolean
- `llm_temperature`: number, default 0
- `timeout_budget_sec`: integer
- `priority`: `normal` or `research`

### `client_capabilities`

- `supports_farfield_export`: boolean
- `supports_current_distribution_export`: boolean
- `supports_parameter_sweep`: boolean
- `max_simulation_timeout_sec`: integer
- `export_formats`: array of enums

## Command Package Contract

The LLM never sends VBA. The server emits a `command_package` with deterministic high-level commands.

Top-level package fields:

- `schema_version`: `cst_command_package.v1`
- `command_catalog_version`: string
- `session_id`: UUID
- `trace_id`: UUID
- `design_id`: string
- `iteration_index`: integer
- `units`: object
- `predicted_dimensions`: object
- `predicted_metrics`: object
- `commands`: ordered array
- `expected_exports`: array
- `safety_checks`: array

Each command has:

- `seq`: integer, strictly increasing
- `command`: enum
- `params`: object, validated by command-specific schema
- `on_failure`: enum `abort`, `retry_once`, `continue`
- `checksum_scope`: string used for reproducibility

## Whitelisted Commands v1

1. `create_project`
2. `set_units`
3. `set_frequency_range`
4. `define_material`
5. `create_substrate`
6. `create_ground_plane`
7. `create_patch`
8. `create_feedline`
9. `create_port`
10. `set_boundary`
11. `set_solver`
12. `run_simulation`
13. `export_s_parameters`
14. `export_farfield`
15. `extract_summary_metrics`

## Required Command Ordering

1. `create_project`
2. `set_units`
3. `set_frequency_range`
4. `define_material` zero or more
5. `create_substrate`
6. `create_ground_plane`
7. `create_patch`
8. `create_feedline`
9. `create_port`
10. `set_boundary`
11. `set_solver`
12. `run_simulation`
13. export and extraction commands

No command may appear before its prerequisites. No geometry modification command may appear after `run_simulation`.

## Clarification and IDK Rules

The server must enter `clarification_required` instead of guessing when any of the following holds:

- target frequency is missing or non-numeric
- bandwidth is missing or physically inconsistent
- request asks for unsupported antenna family or solver behavior
- user constraints conflict with ANN training bounds
- command plan requires a capability not declared by the client

The returned clarification payload must contain:

- `reason`
- `missing_fields`
- `suggested_questions`
- `safe_next_step`

Note: unsupported family is now a deterministic validation error (`FAMILY_NOT_SUPPORTED`) when request validation reaches family profile enforcement.

## Feedback Contract from Client

The CST client should eventually post or stream the following data back to the server:

- `session_id`
- `design_id`
- `iteration_index`
- `simulation_status`
- `actual_center_frequency_ghz`
- `actual_bandwidth_mhz`
- `actual_return_loss_db`
- `actual_vswr`
- `actual_gain_dbi`
- `s11_trace_ref`
- `farfield_ref` optional
- `notes` optional

## Acceptance Policy v1

A design is accepted when all enabled acceptance thresholds pass:

- $|f_{actual} - f_{target}| \le \Delta f$
- $BW_{actual} \ge BW_{target}$
- $VSWR_{actual} \le VSWR_{max}$
- $G_{actual} \ge G_{target}$ when gain target is enabled

If acceptance fails and `max_iterations` is not exhausted, the server generates a refinement cycle.

## Implementation Note

These contracts should remain stable while the implementation is built. Any future incompatible change should produce a new schema version, not a silent edit.