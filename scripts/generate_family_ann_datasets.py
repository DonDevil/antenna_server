from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data.family_dataset_generators import write_all_family_synth_datasets, write_family_synth_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate formula-based synthetic datasets for the antenna ANN families.")
    parser.add_argument(
        "--family",
        choices=["all", "microstrip_patch", "wban_patch"],
        default="all",
        help="Family to generate. Use 'all' to create all three datasets.",
    )
    parser.add_argument("--rows", type=int, default=15000, help="Number of rows per family.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    args = parser.parse_args()

    if args.family == "all":
        for artifact in write_all_family_synth_datasets(rows_per_family=args.rows, seed=args.seed):
            print(f"[{artifact.family}] rows={artifact.rows} accepted={artifact.accepted_rows} csv={artifact.csv_path}")
        return

    artifact = write_family_synth_dataset(args.family, rows=args.rows, seed=args.seed)
    print(f"[{artifact.family}] rows={artifact.rows} accepted={artifact.accepted_rows} csv={artifact.csv_path}")


if __name__ == "__main__":
    main()
