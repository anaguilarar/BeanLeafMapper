"""Re-run only A77-B3-S9 field photos from 20260328 and patch the timeseries CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from beanleafmapper.calibration import (
    ArucoCalibrator,
    GridCalibrator,
    TemplateMatchingCalibrator,
)
from beanleafmapper.config import PipelineConfig
from beanleafmapper.io_utils import list_photos
from beanleafmapper.model import build_detector
from beanleafmapper.pipeline.field import process_field_image
from scripts.run_timeseries import plot_growth, summarise_growth


CAPTURE_DATE = "2026-03-28"
PHOTOS_DIR = Path("photos_iaf_20260328")
OUT_DIR = Path("outputs_timeseries")


def main() -> int:
    cfg = PipelineConfig()
    cfg.photos_dir = PHOTOS_DIR
    cfg.output_dir = OUT_DIR
    cfg.detection.image_downscale = 4

    targets = [p for p in list_photos(PHOTOS_DIR, kind="field")
               if p.trial == "A77" and p.block == 3]
    print(f"Patching {len(targets)} A77-B3 field photos from {CAPTURE_DATE}")

    print("Loading SAM3...")
    model = build_detector(cfg.model, cfg.detection.leaf_confidence)
    aruco_cal = ArucoCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    template_cal = TemplateMatchingCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )

    frames = []
    for image_id in targets:
        print(f"  {image_id.stem}")
        df = process_field_image(image_id, model, aruco_cal, cfg, template_cal)
        df = df.assign(capture_date=CAPTURE_DATE, series_key=image_id.series_key)
        frames.append(df)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    patch = pd.concat(frames, ignore_index=True)
    print(f"  -> {patch['area_cm2'].notna().sum()} valid leaf rows")

    field_csv = OUT_DIR / "A77_field_all.csv"
    field_all = pd.read_csv(field_csv)
    # Drop the failed B3-20260328 rows
    drop_mask = (field_all["capture_date"] == CAPTURE_DATE) & (field_all["series_key"] == "A77-B3")
    print(f"  removing {drop_mask.sum()} prior failed rows, adding {len(patch)}")
    field_all = pd.concat([field_all[~drop_mask], patch], ignore_index=True)
    field_all.to_csv(field_csv, index=False)

    lab_all = pd.read_csv(OUT_DIR / "A77_lab_all.csv")
    summary = summarise_growth(field_all, lab_all)
    summary.to_csv(OUT_DIR / "A77_growth_summary.csv", index=False)
    plot_growth(summary, OUT_DIR / "A77_growth.png")

    print("\n=== Updated growth summary ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
