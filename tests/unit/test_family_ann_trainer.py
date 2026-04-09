from __future__ import annotations

import json
from pathlib import Path

from app.data.family_dataset_generators import generate_amc_synth_dataset


def test_assess_fit_status_and_training_artifacts(tmp_path: Path) -> None:
    from app.ann.family_ann_trainer import FamilyAnnSpec, assess_fit_status, train_family_ann_model

    assert assess_fit_status(train_mape=3.0, validation_mape=3.8, test_mape=4.0) == "balanced"
    assert assess_fit_status(train_mape=15.0, validation_mape=16.5, test_mape=17.0) == "underfitting"
    assert assess_fit_status(train_mape=2.0, validation_mape=8.0, test_mape=8.5) == "overfitting"

    csv_path = tmp_path / "amc_synth.csv"
    model_dir = tmp_path / "artifacts"
    generate_amc_synth_dataset(rows=180, seed=9).to_csv(csv_path, index=False)

    spec = FamilyAnnSpec(
        family="amc_patch",
        model_version="amc_test_v1",
        dataset_path=csv_path,
        model_dir=model_dir,
        input_columns=(
            "target_frequency_ghz",
            "target_bandwidth_mhz",
            "substrate_epsilon_r",
            "substrate_height_mm",
        ),
        output_columns=(
            "amc_unit_cell_period_mm",
            "amc_patch_size_mm",
            "amc_gap_mm",
            "amc_via_radius_mm",
            "amc_air_gap_mm",
        ),
        min_rows=120,
    )

    artifacts = train_family_ann_model(spec, epochs=12, batch_size=32)

    assert artifacts.checkpoint_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.train_rows > 0
    assert artifacts.validation_rows > 0
    assert artifacts.test_rows > 0
    assert artifacts.fit_status in {"balanced", "underfitting", "overfitting"}

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["family"] == "amc_patch"
    assert metadata["fit_status"] == artifacts.fit_status
