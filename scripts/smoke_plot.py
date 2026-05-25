"""End-to-end test on every photo of a single plot.

Processes all field + lab photos sharing a plot_key (e.g. A77-B1-S1), runs the
full directory pipelines limited to that plot, then builds the validation report
including the new calibration step.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from beanleafmapper.calibration import (
    ArucoCalibrator,
    GridCalibrator,
    TemplateMatchingCalibrator,
)
from beanleafmapper.config import PipelineConfig
from beanleafmapper.io_utils import list_photos
from beanleafmapper.model import Sam3Model
from beanleafmapper.pipeline.field import _empty_with_warning as field_skip
from beanleafmapper.pipeline.field import process_field_image
from beanleafmapper.pipeline.lab import dedup_lab_sequences
from beanleafmapper.pipeline.lab import _empty_with_warning as lab_skip
from beanleafmapper.pipeline.lab import process_lab_image
from beanleafmapper.pipeline.validation import build_validation_report

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-key", required=True, help="e.g. A77-B1-S1")
    parser.add_argument("--downscale", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cfg.detection.image_downscale = args.downscale

    field_ids = [p for p in list_photos(cfg.photos_dir, kind="field") if p.plot_key == args.plot_key]
    lab_ids = [p for p in list_photos(cfg.photos_dir, kind="lab") if p.plot_key == args.plot_key]
    print(f"plot_key={args.plot_key}: {len(field_ids)} field photos, {len(lab_ids)} lab photos")
    if not field_ids or not lab_ids:
        print("Need at least one field and one lab photo. Exiting.")
        return 2

    print("Loading SAM3...")
    model = Sam3Model(confidence_threshold=cfg.detection.leaf_confidence)
    aruco_cal = ArucoCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    template_cal = TemplateMatchingCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    grid_cal = GridCalibrator(square_size_cm=cfg.grid.square_size_cm)

    field_frames: list[pd.DataFrame] = []
    for image_id in field_ids:
        print(f"  [field] {image_id.stem}")
        try:
            df = process_field_image(image_id, model, aruco_cal, cfg, template_cal)
        except Exception as exc:  # noqa: BLE001
            df = field_skip(image_id, reason=f"error:{type(exc).__name__}:{exc}")
        field_frames.append(df)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    lab_frames: list[pd.DataFrame] = []
    for image_id in lab_ids:
        print(f"  [lab]   {image_id.stem}")
        try:
            df = process_lab_image(image_id, model, aruco_cal, grid_cal, cfg, template_cal)
        except Exception as exc:  # noqa: BLE001
            df = lab_skip(image_id, reason=f"error:{type(exc).__name__}:{exc}")
        lab_frames.append(df)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    field_df = pd.concat(field_frames, ignore_index=True) if field_frames else pd.DataFrame()
    lab_df_raw = pd.concat(lab_frames, ignore_index=True) if lab_frames else pd.DataFrame()
    lab_df = dedup_lab_sequences(lab_df_raw)

    (cfg.output_dir / "field").mkdir(parents=True, exist_ok=True)
    (cfg.output_dir / "lab").mkdir(parents=True, exist_ok=True)
    field_df.to_csv(cfg.output_dir / "field" / f"{args.plot_key}_all.csv", index=False)
    lab_df_raw.to_csv(cfg.output_dir / "lab" / f"{args.plot_key}_raw.csv", index=False)
    lab_df.to_csv(cfg.output_dir / "lab" / f"{args.plot_key}_dedup.csv", index=False)

    print("\n[field summary]")
    print(field_df.groupby("image")["area_cm2"].agg(["count", "mean", "median", "sum"]).to_string())
    print("\n[lab raw]")
    print(lab_df_raw.groupby("image")["area_cm2"].agg(["count", "mean", "median", "sum"]).to_string())
    print("\n[lab deduped — one row per (leaf_no, rank)]")
    print(lab_df.to_string(index=False))

    report = build_validation_report(field_df, lab_df, cfg.output_dir / "validation")
    print("\n[validation]")
    print(report.to_string(index=False))
    print(f"\nOutputs under {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
