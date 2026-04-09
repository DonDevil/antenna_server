from __future__ import annotations

from pathlib import Path

from app.data.family_dataset_generators import (
    AMC_SYNTH_COLUMNS,
    RECT_PATCH_SYNTH_COLUMNS,
    WBAN_SYNTH_COLUMNS,
    generate_amc_synth_dataset,
    generate_rect_patch_synth_dataset,
    generate_wban_synth_dataset,
    write_family_synth_dataset,
)


def test_generate_rect_patch_synth_dataset_produces_expected_fields() -> None:
    df = generate_rect_patch_synth_dataset(rows=24, seed=7)

    assert len(df) == 24
    assert set(RECT_PATCH_SYNTH_COLUMNS).issubset(df.columns)
    assert df["antenna_family"].eq("microstrip_patch").all()
    assert df["patch_shape"].eq("rectangular").all()
    assert df["patch_length_mm"].gt(0).all()
    assert df["patch_width_mm"].gt(0).all()
    assert df["target_minimum_return_loss_db"].lt(0).all()
    assert df["actual_return_loss_db"].lt(0).all()


def test_generate_amc_synth_dataset_produces_expected_fields() -> None:
    df = generate_amc_synth_dataset(rows=24, seed=11)

    assert len(df) == 24
    assert set(AMC_SYNTH_COLUMNS).issubset(df.columns)
    assert df["antenna_family"].eq("amc_patch").all()
    assert df["amc_unit_cell_period_mm"].gt(0).all()
    assert df["amc_patch_size_mm"].gt(0).all()
    assert df["amc_gap_mm"].gt(0).all()
    assert df["actual_gain_improvement_dbi"].between(1.0, 8.0).all()


def test_generate_wban_synth_dataset_and_writer_work(tmp_path: Path) -> None:
    csv_path = tmp_path / "wban_synth.csv"

    df = generate_wban_synth_dataset(rows=24, seed=19)
    artifacts = write_family_synth_dataset("wban_patch", rows=24, seed=19, csv_path=csv_path)

    assert len(df) == 24
    assert set(WBAN_SYNTH_COLUMNS).issubset(df.columns)
    assert df["antenna_family"].eq("wban_patch").all()
    assert df["body_distance_mm"].between(2.0, 10.0).all()
    assert df["bending_radius_mm"].between(30.0, 100.0).all()
    assert csv_path.exists()
    assert artifacts.rows == 24
    assert artifacts.csv_path == csv_path
