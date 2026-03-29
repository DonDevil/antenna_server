# CST High-Level Command Catalog v1

This file defines the only commands the server may send to the CST client. The language model must reason with these commands only. It must never invent commands, emit VBA, or assume unsupported CST behaviors.

## Mandatory Behavior

- Use only the command names listed here.
- Respect command ordering.
- Use only validated numeric parameters.
- If required information is missing, return `IDK` and request clarification.
- Do not guess material constants or dimensions outside provided bounds.

## Command List

### `create_project`

Purpose: start a new CST project container.

Parameters:

- `project_name`: string

### `set_units`

Purpose: lock project units.

Parameters:

- `geometry`: must be `mm`
- `frequency`: must be `ghz`

### `set_frequency_range`

Purpose: define solver sweep limits.

Parameters:

- `start_ghz`: number
- `stop_ghz`: number

Constraint: `start_ghz < stop_ghz`

### `define_material`

Purpose: register conductor or substrate material before geometry creation.

Parameters:

- `name`: string
- `kind`: `conductor` or `substrate`
- `epsilon_r`: number, required for substrates
- `loss_tangent`: number, optional for substrates
- `conductivity_s_per_m`: number, required for conductors

### `create_substrate`

Purpose: create dielectric block.

Parameters:

- `name`
- `material`
- `length_mm`
- `width_mm`
- `height_mm`
- `origin_mm.x`
- `origin_mm.y`
- `origin_mm.z`

### `create_ground_plane`

Purpose: create conductive ground plane.

Parameters:

- `name`
- `material`
- `length_mm`
- `width_mm`
- `thickness_mm`
- `z_mm`

### `create_patch`

Purpose: create the radiating patch geometry.

Parameters:

- `name`
- `material`
- `length_mm`
- `width_mm`
- `thickness_mm`
- `center_mm.x`
- `center_mm.y`
- `center_mm.z`

### `create_feedline`

Purpose: create the feed geometry.

Parameters:

- `name`
- `material`
- `length_mm`
- `width_mm`
- `thickness_mm`
- `start_mm`
- `end_mm`

### `create_port`

Purpose: define excitation port.

Parameters:

- `port_id`
- `port_type`: `discrete` or `waveguide`
- `impedance_ohm`
- `reference_mm`

### `set_boundary`

Purpose: define the simulation boundary condition.

Parameters:

- `boundary_type`: `open_add_space` or `expanded_open`
- `padding_mm`: optional number

### `set_solver`

Purpose: choose CST solver settings.

Parameters:

- `solver_type`: `time_domain` or `frequency_domain`
- `mesh_cells_per_wavelength`: optional integer

### `run_simulation`

Purpose: execute the solver.

Parameters:

- `timeout_sec`

### `export_s_parameters`

Purpose: export S-parameter results.

Parameters:

- `format`: `json`, `csv`, or `txt`
- `destination_hint`: string

### `export_farfield`

Purpose: export far-field results if supported.

Parameters:

- `format`: `json`, `csv`, or `txt`
- `frequency_ghz`
- `destination_hint`

### `extract_summary_metrics`

Purpose: extract scalar metrics for feedback to the server.

Parameters:

- `metrics`: one or more of `center_frequency_ghz`, `bandwidth_mhz`, `return_loss_db`, `vswr`, `gain_dbi`

## Required Command Order

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
14. `export_farfield` if supported and requested
15. `extract_summary_metrics`

## Forbidden Behavior

- No raw VBA output
- No free-form CST macro text
- No command outside this catalog
- No geometry edits after `run_simulation`
- No guessing hidden parameters
- No silent fallback when required data is absent

## IDK Rule

Return `IDK` when:

- frequency or bandwidth cannot be confidently extracted
- constraints conflict or are incomplete
- a requested action needs a command not listed above
- client capability required by the plan is unavailable