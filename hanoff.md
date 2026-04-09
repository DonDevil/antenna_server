# Rectangular Patch ANN Handoff

I'll give you a concrete handoff plan with exact dataset files, column order, model scope, training flow, and integration rules so Copilot can implement it directly.

## Objective

Build a rectangular microstrip patch ANN pipeline that uses real CST feedback, improves over time, and does not break the current server contract.

This first phase is only for:

antenna_family = microstrip_patch
patch_shape = rectangular
one feed style only at first: preferably edge
one polarization only at first: linear
Do not mix AMC, WBAN, circular patch, or multiple feed styles into this first ANN.

Core design
Use three data layers, not one:

Master feedback log
Purpose: everything collected from client/CST, rich and future-proof.

Inverse training dataset
Purpose: train ANN to predict starting geometry from target specs.

Forward/correction dataset
Purpose: train future forward model and self-improvement residual model.

This avoids confusing the ANN while keeping all useful information.

Files to create
Use these exact dataset files:

data/raw/rect_patch_feedback_v1.csv
Purpose: raw CST-backed feedback rows from the client.

data/validated/rect_patch_feedback_validated_v1.csv
Purpose: cleaned usable rows after validation.

data/validated/rect_patch_inverse_train_v1.csv
Purpose: compact inverse-ANN dataset derived from validated feedback.

data/validated/rect_patch_forward_train_v1.csv
Purpose: compact forward-ANN dataset derived from validated feedback.

models/ann/rect_patch_v1/
Purpose: family-specific ANN artifacts for rectangular patch.

models/ann/rect_patch_v1/metadata.json
Purpose: training schema, scaler stats, feature order, metrics, artifact version.

Master feedback CSV
This is the file your client should generate.

Exact column order
run_id,
timestamp_utc,
antenna_family,
patch_shape,
feed_type,
polarization,
substrate_name,
conductor_name,

target_frequency_ghz,
target_bandwidth_mhz,
target_minimum_gain_dbi,
target_maximum_vswr,
target_minimum_return_loss_db,

substrate_epsilon_r,
substrate_height_mm,

patch_length_mm,
patch_width_mm,
patch_height_mm,
substrate_length_mm,
substrate_width_mm,
feed_length_mm,
feed_width_mm,
feed_offset_x_mm,
feed_offset_y_mm,

actual_center_frequency_ghz,
actual_bandwidth_mhz,
actual_return_loss_db,
actual_vswr,
actual_gain_dbi,
actual_radiation_efficiency_pct,
actual_total_efficiency_pct,
actual_directivity_dbi,
actual_peak_theta_deg,
actual_peak_phi_deg,
actual_front_to_back_db,
actual_axial_ratio_db,

accepted,
solver_status,
simulation_time_sec,
notes,
farfield_artifact_path,
s11_artifact_path

Fixed values for first collection phase
For now, keep these fixed in every row:

antenna_family = microstrip_patch
patch_shape = rectangular
feed_type = edge
polarization = linear
conductor_name = Copper (annealed) if you are not varying conductor
That reduces model confusion.

what the inverse ANN should actually use
Do not train on every logged field.

Inverse ANN input order
Use this exact input feature order for rect_patch_inverse_v1
target_frequency_ghz,
target_bandwidth_mhz,
substrate_epsilon_r,
substrate_height_mm

That is the best first version.

Optional richer input set
Only if the first dataset is large enough and balanced, extend to:
target_frequency_ghz,
target_bandwidth_mhz,
substrate_epsilon_r,
substrate_height_mm,
target_minimum_gain_dbi,
target_maximum_vswr

Do not start with more than this.

Do not use these as inverse ANN inputs
accepted
actual_*
notes
artifact paths
timestamp
run_id
feed_type if fixed
polarization if fixed
conductor_name if fixed
far-field metrics
objective priorities unless you truly vary them in collection

What the inverse ANN should predict
Do not predict every geometry value at first.

Inverse ANN output order
Use this exact output order:

patch_length_mm,
patch_width_mm,
feed_width_mm,
feed_offset_y_mm

Geometry that should remain deterministic
These should still come from recipe logic, not the ANN:

patch_height_mm
substrate_length_mm
substrate_width_mm
feed_length_mm
feed_offset_x_mm
Why
For rectangular patch, the most sensitive starting variables are:

patch length
patch width
feed width
feed location
If the ANN predicts only these, it has a much easier job and generalizes better.

Derived inverse training dataset
Create rect_patch_inverse_train_v1.csv from the validated feedback file.

Exact column order
target_frequency_ghz,
target_bandwidth_mhz,
substrate_epsilon_r,
substrate_height_mm,
patch_length_mm,
patch_width_mm,
feed_width_mm,
feed_offset_y_mm

If using the richer version:
target_frequency_ghz,
target_bandwidth_mhz,
substrate_epsilon_r,
substrate_height_mm,
target_minimum_gain_dbi,
target_maximum_vswr,
patch_length_mm,
patch_width_mm,
feed_width_mm,
feed_offset_y_mm

Derived forward training dataset
Create rect_patch_forward_train_v1.csv.

Exact column order
target_frequency_ghz,
target_bandwidth_mhz,
substrate_epsilon_r,
substrate_height_mm,
patch_length_mm,
patch_width_mm,
feed_width_mm,
feed_offset_y_mm,
actual_center_frequency_ghz,
actual_bandwidth_mhz,
actual_return_loss_db,
actual_vswr,
actual_gain_dbi,
actual_radiation_efficiency_pct,
actual_total_efficiency_pct,
actual_directivity_dbi,
actual_front_to_back_db,
actual_axial_ratio_db

This is for the future forward evaluator and residual correction loop.

Validation rules for the master feedback file
Before a row is accepted into validated datasets, require:

antenna_family == microstrip_patch
patch_shape == rectangular
feed_type == edge
target_frequency_ghz > 0
target_bandwidth_mhz > 0
actual_center_frequency_ghz > 0
actual_bandwidth_mhz > 0
actual_vswr > 0
patch_length_mm > 0
patch_width_mm > 0
feed_width_mm > 0
substrate_height_mm > 0
solver_status == success or equivalent success code
Reject rows where:

CST crashed
fields are missing
dimensions are out of safe bounds
actual frequency or bandwidth is zero/invalid
geometry is clearly corrupted
Recommended safe bounds for rectangular patch collection
Use these initial bounds:

target_frequency_ghz: 2.0 to 7.0

target_bandwidth_mhz: 30.0 to 300.0

substrate_epsilon_r: 2.2 to 4.4

substrate_height_mm: 0.8 to 3.2

Implemented workflow entry points
Once real CST rows are available, use these commands in this repo:

python scripts/validate_rect_patch_feedback.py
python scripts/derive_rect_patch_datasets.py
python scripts/train_rect_patch_ann.py

Or run the whole rectangular path in one command:

python scripts/run_rect_patch_pipeline.py

To compare recipe-only vs family ANN on validated rectangular rows after training:

python scripts/evaluate_rect_patch_ann.py

Client-side logging helper now exists in:

app/data/rect_patch_feedback_logger.py

Use append_rect_patch_feedback_row(...) to write schema-compliant rows into:

data/raw/rect_patch_feedback_v1.csv

patch_length_mm: 5.0 to 80.0

patch_width_mm: 5.0 to 100.0

feed_width_mm: 0.2 to 8.0

feed_offset_y_mm: -50.0 to 0.0

Keep ranges narrower at first to improve stability.

ANN model plan
Model 1: inverse starting-geometry ANN
Name:

rect_patch_inverse_v1
Purpose:

target specs -> starting geometry
Input:

4 features first
Output:

4 geometry values
Recommended architecture
Start simple:

Dense 64
ReLU
Dense 128
ReLU
Dense 64
ReLU
Dense 4
Use:

standardized inputs
standardized outputs
Adam optimizer
MSE or Huber loss
Training split
Use exact split:

70% train
15% validation
15% test
Do not use only training loss. Save:

train loss
validation loss
test MAE per target
test MAPE per target where meaningful
Early stopping
Use early stopping on validation loss:

patience: 20
restore best weights: true
Confidence and fallback behavior
Do not let the ANN fully control the geometry at first.

Runtime integration rule
At inference time:

generate recipe baseline
get inverse ANN prediction for 4 fields
clamp outputs to safe bounds
merge ANN outputs into recipe-generated geometry
keep deterministic fields from recipe
if ANN prediction is outlier-like, fall back to recipe-only
Initial merge rule
Use ANN to override only:

patch_length_mm
patch_width_mm
feed_width_mm
feed_offset_y_mm
Everything else stays recipe-derived.

This keeps the system stable.

Self-improvement logic
You said you liked the retraining logic from the old project. Keep that idea, but make it cleaner.

Phase 1
Do not use online correction immediately in production.

Instead:

append new CST feedback rows continuously
retrain offline when enough new rows accumulate
Retrain trigger
Retrain only when:

at least 100 new validated rows are added
or performance audit says the current model degraded
Phase 2 residual model
After enough data exists, add a residual corrector.

Residual model input:
target_frequency_ghz,
target_bandwidth_mhz,
substrate_epsilon_r,
substrate_height_mm,
recipe_patch_length_mm,
recipe_patch_width_mm,
recipe_feed_width_mm,
recipe_feed_offset_y_mm,
predicted_center_frequency_ghz,
predicted_bandwidth_mhz,
predicted_vswr
Residual model output:
delta_patch_length_mm,
delta_patch_width_mm,
delta_feed_width_mm,
delta_feed_offset_y_mm

But do this only after the base inverse ANN is working well.

Exact client-side collection plan
Give Copilot this collection rule set.

Randomize these inputs
target_frequency_ghz
target_bandwidth_mhz
substrate_name
substrate_epsilon_r
substrate_height_mm
Keep these fixed initially
antenna_family = microstrip_patch
patch_shape = rectangular
feed_type = edge
polarization = linear
conductor_name = Copper (annealed)
Geometry generation for each run
For each random target:

generate recipe baseline geometry
perturb only:
patch length
patch width
feed width
feed offset y
keep the rest deterministic
run CST
log a full row into rect_patch_feedback_v1.csv
This will create a clean supervised dataset instead of a chaotic one.

Recommended perturbation strategy for collection
To avoid narrow data:

patch length perturbation:
±
15
%
±15%
patch width perturbation:
±
15
%
±15%
feed width perturbation:
±
20
%
±20%
feed offset y perturbation:
±
20
%
±20%
Use bounded random exploration around the recipe, not fully random geometry.

That gives useful coverage without too many failed CST runs.

Artifact and metadata plan
For models/ann/rect_patch_v1/metadata.json, store:

model_version
family
patch_shape
feed_type
input_columns
output_columns
train_rows
validation_rows
test_rows
x_mean
x_std
y_mean
y_std
train_loss
val_loss
test_mae
test_mape
safe_output_bounds
This keeps the training reproducible.

Exact implementation order for Copilot
Give Copilot this sequence.

Step 1
Add new raw feedback schema support:

create validator/parser for rect_patch_feedback_v1.csv
Step 2
Create derivation scripts:

raw feedback -> validated feedback
validated feedback -> inverse training CSV
validated feedback -> forward training CSV
Step 3
Create rect_patch_inverse_v1 trainer:

compact 4-input, 4-output ANN
scaler save/load
metadata save
validation/test metrics
Step 4
Update predictor integration:

recipe baseline remains primary
ANN only overrides 4 selected geometry fields
keep fallback to recipe-only
Step 5
Add offline retrain command:

retrain only when new validated rows threshold is met
Step 6
Add evaluation script:

compare recipe-only vs recipe+ANN on held-out real CST rows
Exact short handoff you can give to Copilot
Use this verbatim if you want a tighter prompt:

Implement a new rectangular-patch ANN pipeline for microstrip_patch with rectangular shape and fixed edge feed only. Add a raw CST feedback CSV schema at data/raw/rect_patch_feedback_v1.csv, validate it into data/validated/rect_patch_feedback_validated_v1.csv, derive rect_patch_inverse_train_v1.csv and rect_patch_forward_train_v1.csv, and train a family-specific inverse ANN rect_patch_inverse_v1 using inputs [target_frequency_ghz, target_bandwidth_mhz, substrate_epsilon_r, substrate_height_mm] and outputs [patch_length_mm, patch_width_mm, feed_width_mm, feed_offset_y_mm]. Keep the existing server contract unchanged: recipe logic should still fill the rest of the geometry, and the ANN should only override those 4 fields with bounds checking and recipe fallback.
I've laid out the complete rectangular-patch dataset and ANN plan with exact file layout, column order, feature/output sets, training rules, and integration strategy for a Copilot handoff.
End of handoff.

