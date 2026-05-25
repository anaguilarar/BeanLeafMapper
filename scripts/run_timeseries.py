"""Process A77 photos across multiple capture dates and compare leaf area growth.

Walks every photos_iaf_<YYYYMMDD> folder, runs the field + lab pipelines on the
photos for a chosen trial (default A77), aggregates per (series_key, date), and
produces a CSV + plot of leaf-area growth over time."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from beanleafmapper.pipeline.field import _empty_with_warning as field_skip
from beanleafmapper.pipeline.field import process_field_image
from beanleafmapper.pipeline.lab import dedup_lab_sequences
from beanleafmapper.pipeline.lab import _empty_with_warning as lab_skip
from beanleafmapper.pipeline.lab import process_lab_image


DATE_RE = re.compile(r"photos_iaf_(\d{8})$")


def find_date_folders(root: Path) -> list[tuple[str, Path]]:
    """Return [(YYYY-MM-DD, folder), ...] sorted by date."""
    out: list[tuple[str, Path]] = []
    for p in sorted(root.glob("photos_iaf_*")):
        if not p.is_dir():
            continue
        m = DATE_RE.match(p.name)
        if not m:
            continue
        iso = datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        out.append((iso, p))
    return sorted(out, key=lambda t: t[0])


def process_one_date(
    capture_date: str,
    photos_dir: Path,
    trial: str,
    cfg: PipelineConfig,
    model,
    aruco_cal: ArucoCalibrator,
    template_cal: TemplateMatchingCalibrator,
    grid_cal: GridCalibrator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (field_df, lab_df_deduped) for the given trial on the given date."""
    field_ids = [p for p in list_photos(photos_dir, kind="field") if p.trial == trial]
    lab_ids = [p for p in list_photos(photos_dir, kind="lab") if p.trial == trial]
    print(f"[{capture_date}] {len(field_ids)} field + {len(lab_ids)} lab photos")

    field_frames: list[pd.DataFrame] = []
    for image_id in field_ids:
        print(f"  field {image_id.stem}")
        try:
            df = process_field_image(image_id, model, aruco_cal, cfg, template_cal)
        except Exception as exc:  # noqa: BLE001
            df = field_skip(image_id, reason=f"error:{type(exc).__name__}:{exc}")
        df = df.assign(capture_date=capture_date, series_key=image_id.series_key)
        field_frames.append(df)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    lab_frames: list[pd.DataFrame] = []
    for image_id in lab_ids:
        print(f"  lab   {image_id.stem}")
        try:
            df = process_lab_image(image_id, model, aruco_cal, grid_cal, cfg, template_cal)
        except Exception as exc:  # noqa: BLE001
            df = lab_skip(image_id, reason=f"error:{type(exc).__name__}:{exc}")
        df = df.assign(capture_date=capture_date, series_key=image_id.series_key)
        lab_frames.append(df)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    field_df = pd.concat(field_frames, ignore_index=True) if field_frames else pd.DataFrame()
    lab_df_raw = pd.concat(lab_frames, ignore_index=True) if lab_frames else pd.DataFrame()
    lab_df = dedup_lab_sequences(lab_df_raw)
    if not lab_df.empty:
        plot_to_series = lab_df_raw.drop_duplicates("plot_key")[["plot_key", "series_key"]]
        lab_df = lab_df.merge(plot_to_series, on="plot_key", how="left")
        lab_df["capture_date"] = capture_date
    return field_df, lab_df


def summarise_growth(field_df: pd.DataFrame, lab_df: pd.DataFrame) -> pd.DataFrame:
    """Per-date per-block summary: mean/median leaf area and total leaf area."""
    rows: list[dict] = []
    if not field_df.empty:
        f = field_df.dropna(subset=["area_cm2"])
        for (date, series), g in f.groupby(["capture_date", "series_key"]):
            rows.append(
                {
                    "capture_date": date,
                    "series_key": series,
                    "source": "field",
                    "n_leaves": int(g.shape[0]),
                    "mean_area_cm2": float(g["area_cm2"].mean()),
                    "median_area_cm2": float(g["area_cm2"].median()),
                    "total_area_cm2": float(g["area_cm2"].sum()),
                    "n_images": int(g["image"].nunique()),
                }
            )
    if not lab_df.empty and "area_cm2" in lab_df.columns:
        l = lab_df.dropna(subset=["area_cm2"])
        for (date, series), g in l.groupby(["capture_date", "series_key"]):
            rows.append(
                {
                    "capture_date": date,
                    "series_key": series,
                    "source": "lab",
                    "n_leaves": int(g.shape[0]),
                    "mean_area_cm2": float(g["area_cm2"].mean()),
                    "median_area_cm2": float(g["area_cm2"].median()),
                    "total_area_cm2": float(g["area_cm2"].sum()),
                    "n_images": int(g["leaf_no"].nunique()) if "leaf_no" in g else 0,
                }
            )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["series_key", "capture_date", "source"]).reset_index(drop=True)
    return summary


def plot_growth(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    sources = ["field", "lab"]
    metrics = [
        ("mean_area_cm2", "Mean leaf area (cm²)"),
        ("total_area_cm2", "Total detected leaf area (cm²)"),
        ("n_leaves", "Number of detected leaves"),
    ]
    if summary.empty:
        for ax, (_, label) in zip(axes, metrics):
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_ylabel(label)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    summary = summary.copy()
    summary["capture_date"] = pd.to_datetime(summary["capture_date"])
    for ax, (col, label) in zip(axes, metrics):
        for source in sources:
            for series, g in summary[summary["source"] == source].groupby("series_key"):
                g = g.sort_values("capture_date")
                style = "-o" if source == "field" else "--s"
                ax.plot(g["capture_date"], g[col], style,
                        label=f"{series} ({source})", alpha=0.85)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.suptitle("A77 leaf-area growth over time")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Project root containing photos_iaf_<YYYYMMDD> folders.")
    parser.add_argument("--trial", type=str, default="A77")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_timeseries"))
    parser.add_argument("--downscale", type=int, default=4)
    args = parser.parse_args()

    date_folders = find_date_folders(args.root)
    if not date_folders:
        print(f"No photos_iaf_<date> folders found in {args.root}.")
        return 2
    print("Capture dates:")
    for date, folder in date_folders:
        print(f"  {date}  ->  {folder}")

    cfg = PipelineConfig()
    cfg.output_dir = args.output_dir
    cfg.detection.image_downscale = args.downscale
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading SAM3...")
    model = build_detector(cfg.model, cfg.detection.leaf_confidence)
    aruco_cal = ArucoCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    template_cal = TemplateMatchingCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    grid_cal = GridCalibrator(square_size_cm=cfg.grid.square_size_cm)

    all_field: list[pd.DataFrame] = []
    all_lab: list[pd.DataFrame] = []
    for capture_date, folder in date_folders:
        cfg.photos_dir = folder
        field_df, lab_df = process_one_date(
            capture_date, folder, args.trial, cfg, model,
            aruco_cal, template_cal, grid_cal,
        )
        all_field.append(field_df)
        all_lab.append(lab_df)

    field_all = pd.concat(all_field, ignore_index=True) if all_field else pd.DataFrame()
    lab_all = pd.concat(all_lab, ignore_index=True) if all_lab else pd.DataFrame()
    field_all.to_csv(args.output_dir / f"{args.trial}_field_all.csv", index=False)
    lab_all.to_csv(args.output_dir / f"{args.trial}_lab_all.csv", index=False)

    summary = summarise_growth(field_all, lab_all)
    summary.to_csv(args.output_dir / f"{args.trial}_growth_summary.csv", index=False)
    plot_growth(summary, args.output_dir / f"{args.trial}_growth.png")

    print("\n=== Growth summary ===")
    if summary.empty:
        print("(no detections)")
    else:
        print(summary.to_string(index=False))
    print(f"\nOutputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
