import pandas as pd

from app.data.validator import validate_dataset


def test_dataset_validator_rejects_out_of_range_rows():
    df = pd.DataFrame(
        [
            {
                "frequency_ghz": 2.4,
                "bandwidth_mhz": 80,
                "patch_length_mm": 30,
                "patch_width_mm": 35,
                "patch_height_mm": 0.035,
                "substrate_length_mm": 45,
                "substrate_width_mm": 50,
                "substrate_height_mm": 1.6,
                "feed_length_mm": 12,
                "feed_width_mm": 2.0,
                "feed_offset_x_mm": 0.0,
                "feed_offset_y_mm": -8.0,
            },
            {
                "frequency_ghz": 42.0,
                "bandwidth_mhz": 80,
                "patch_length_mm": 30,
                "patch_width_mm": 35,
                "patch_height_mm": 0.035,
                "substrate_length_mm": 45,
                "substrate_width_mm": 50,
                "substrate_height_mm": 1.6,
                "feed_length_mm": 12,
                "feed_width_mm": 2.0,
                "feed_offset_x_mm": 0.0,
                "feed_offset_y_mm": -8.0,
            },
        ]
    )

    result = validate_dataset(df)
    assert len(result.valid_df) == 1
    assert len(result.rejected_df) == 1
