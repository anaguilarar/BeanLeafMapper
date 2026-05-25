"""Combine field and lab outputs into a per-plot comparison report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from beanleafmapper.config import PipelineConfig
from beanleafmapper.pipeline.validation import build_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--field-csv",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/field/all_leaves.csv",
    )
    parser.add_argument(
        "--lab-csv",
        type=Path,
        default=None,
        help="Defaults to <output-dir>/lab/all_leaves.csv",
    )
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    field_csv = args.field_csv or cfg.output_dir / "field" / "all_leaves.csv"
    lab_csv = args.lab_csv or cfg.output_dir / "lab" / "all_leaves.csv"
    if not field_csv.exists() or not lab_csv.exists():
        raise SystemExit(
            f"Missing input CSVs.\n"
            f"  field: {field_csv} ({'ok' if field_csv.exists() else 'missing'})\n"
            f"  lab:   {lab_csv} ({'ok' if lab_csv.exists() else 'missing'})\n"
            f"Run scripts/run_field.py and scripts/run_lab.py first."
        )

    field_df = pd.read_csv(field_csv)
    lab_df = pd.read_csv(lab_csv)
    report = build_validation_report(field_df, lab_df, cfg.output_dir / "validation")
    print(report.to_string(index=False))
    if "rmse_cm2" in report.attrs:
        print(f"\nMean error: {report.attrs['mean_error_cm2']:.2f} cm² | RMSE: {report.attrs['rmse_cm2']:.2f} cm²")
    print(f"\nReport saved to {cfg.output_dir / 'validation'}")


if __name__ == "__main__":
    main()
