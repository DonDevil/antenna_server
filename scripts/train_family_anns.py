from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.ann.family_ann_trainer import FamilyAnnSpec, default_family_specs, evaluate_family_ann_model, train_family_ann_model
from app.data.family_dataset_generators import write_all_family_synth_datasets, write_family_synth_dataset


def _load_client_rect_reference(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()
    if df.empty:
        return df

    for column in (
        "target_frequency_ghz",
        "target_bandwidth_mhz",
        "substrate_epsilon_r",
        "substrate_height_mm",
        "patch_length_mm",
        "patch_width_mm",
        "feed_width_mm",
        "feed_offset_y_mm",
        "actual_bandwidth_mhz",
        "target_minimum_return_loss_db",
        "actual_return_loss_db",
    ):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ("target_minimum_return_loss_db", "actual_return_loss_db"):
        if column in df.columns:
            df[column] = -df[column].abs()
    if "actual_axial_ratio_db" in df.columns:
        df["actual_axial_ratio_db"] = pd.to_numeric(df["actual_axial_ratio_db"], errors="coerce").fillna(0.0)

    mask = pd.Series(True, index=df.index)
    if "antenna_family" in df.columns:
        mask &= df["antenna_family"].astype(str).str.lower().eq("microstrip_patch")
    if "patch_shape" in df.columns:
        mask &= df["patch_shape"].astype(str).str.lower().eq("rectangular")
    if "feed_type" in df.columns:
        mask &= df["feed_type"].astype(str).str.lower().eq("edge")
    if "solver_status" in df.columns:
        mask &= df["solver_status"].astype(str).str.lower().isin({"success", "completed"})
    if "actual_bandwidth_mhz" in df.columns:
        mask &= df["actual_bandwidth_mhz"].fillna(0.0) >= 5.0

    return df.loc[mask].reset_index(drop=True)


def _augment_rect_training_spec(spec: FamilyAnnSpec, client_csv: Path) -> tuple[FamilyAnnSpec, int]:
    if not client_csv.exists():
        return spec, 0

    client_reference = _load_client_rect_reference(client_csv)
    if client_reference.empty:
        return spec, 0

    synth_df = pd.read_csv(spec.dataset_path)
    training_columns = list(dict.fromkeys([*spec.input_columns, *spec.output_columns]))
    repeat_factor = max(1, min(120, max(20, len(synth_df) // max(len(client_reference), 1) // 10)))
    client_augmented = pd.concat([client_reference.loc[:, training_columns]] * repeat_factor, ignore_index=True)
    combined = pd.concat([synth_df.loc[:, training_columns], client_augmented], ignore_index=True)
    combined_path = spec.dataset_path.with_name("rect_patch_formula_plus_client_v1.csv")
    combined.to_csv(combined_path, index=False)
    return replace(spec, dataset_path=combined_path), int(len(client_reference))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate larger family datasets, train the three family ANNs, and report fit quality.")
    parser.add_argument(
        "--family",
        choices=["all", "microstrip_patch", "wban_patch"],
        default="all",
        help="Which family to train. Default: all three.",
    )
    parser.add_argument("--rows", type=int, default=15000, help="Synthetic rows to generate per family.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation.")
    parser.add_argument("--skip-generate", action="store_true", help="Skip dataset regeneration and train from the current CSVs.")
    parser.add_argument(
        "--client-csv",
        default="/home/darkdevil/Desktop/antenna_client/data/raw/rect_patch_feedback_v1.csv",
        help="Optional client-side rectangular feedback CSV used for an external rect-patch check.",
    )
    args = parser.parse_args()

    if not args.skip_generate:
        if args.family == "all":
            generated = write_all_family_synth_datasets(rows_per_family=args.rows, seed=args.seed)
        else:
            generated = [write_family_synth_dataset(args.family, rows=args.rows, seed=args.seed)]
        for artifact in generated:
            print(f"generated[{artifact.family}] rows={artifact.rows} accepted={artifact.accepted_rows} csv={artifact.csv_path}")

    all_specs = default_family_specs()
    selected_specs = all_specs if args.family == "all" else {args.family: all_specs[args.family]}

    client_csv = Path(args.client_csv)
    if "microstrip_patch" in selected_specs:
        augmented_spec, reference_rows = _augment_rect_training_spec(selected_specs["microstrip_patch"], client_csv)
        selected_specs["microstrip_patch"] = augmented_spec
        if reference_rows:
            print(f"rect client calibration rows={reference_rows} merged_into={selected_specs['microstrip_patch'].dataset_path}")

    summary: list[dict[str, object]] = []
    for family, spec in selected_specs.items():
        artifacts = train_family_ann_model(spec)
        summary.append(
            {
                "family": family,
                "model_version": spec.model_version,
                "checkpoint": str(artifacts.checkpoint_path),
                "metadata": str(artifacts.metadata_path),
                "train_rows": artifacts.train_rows,
                "validation_rows": artifacts.validation_rows,
                "test_rows": artifacts.test_rows,
                "best_validation_loss": artifacts.best_validation_loss,
                "train_mape": artifacts.train_mape,
                "validation_mape": artifacts.validation_mape,
                "test_mape": artifacts.test_mape,
                "fit_status": artifacts.fit_status,
            }
        )

    print(json.dumps({"trained_models": summary}, indent=2))

    if "microstrip_patch" in selected_specs and client_csv.exists():
        client_reference = _load_client_rect_reference(client_csv)
        if not client_reference.empty:
            reference_summary = evaluate_family_ann_model(
                selected_specs["microstrip_patch"],
                reference_df=client_reference,
            )
            print(json.dumps({"client_rect_reference": reference_summary}, indent=2))


if __name__ == "__main__":
    main()
