"""Lab-image pipeline: detached leaves on graph paper. Used as ground truth.

For each lab photo we:
  1. Calibrate pixel size from the ArUco marker.
  2. Cross-check pixel size against the graph-paper grid period.
  3. Detect leaves with SAM3 and report per-leaf area in cm².
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration import ArucoCalibrator, GridCalibrator, TemplateMatchingCalibrator
from ..config import PipelineConfig
from ..detector import LeavesDetector
from ..io_utils import ImageId, list_photos, load_rgb
from ..model import Sam3Model, build_detector
from ..visualization import save_annotated


def process_lab_image(
    image_id: ImageId,
    model: Sam3Model,
    aruco_cal: ArucoCalibrator,
    grid_cal: GridCalibrator,
    cfg: PipelineConfig,
    template_cal: TemplateMatchingCalibrator | None = None,
) -> pd.DataFrame:
    """Process one lab photo. Returns a DataFrame with one row per detected leaf."""
    image = load_rgb(image_id.path, downscale=cfg.detection.image_downscale)
    image_np = np.array(image)

    aruco_scale = aruco_cal.calibrate(image_np)
    if aruco_scale is None and template_cal is not None:
        aruco_scale = template_cal.calibrate(image_np)
    grid_scale = grid_cal.calibrate(image_np)
    scale = aruco_scale or grid_scale
    if scale is None:
        return _empty_with_warning(image_id, reason="no_calibration")

    aruco_cm_per_px = aruco_scale.cm_per_px if aruco_scale else None
    grid_cm_per_px = grid_scale.cm_per_px if grid_scale else None
    calibration_ratio = (
        aruco_cm_per_px / grid_cm_per_px if (aruco_cm_per_px and grid_cm_per_px) else None
    )

    inference = model.detect(
        image,
        text_prompt=cfg.detection.leaf_prompt,
        confidence_threshold=cfg.detection.leaf_confidence,
    )
    leaves = LeavesDetector()
    leaves.set_sam3results(inference)
    if leaves.n_objects == 0:
        return _empty_with_warning(image_id, reason="no_leaves_detected")

    rows: list[dict] = []
    for i in range(leaves.n_objects):
        try:
            m = leaves.calculate_oneobject_metrics(i)
        except ValueError:
            continue
        rows.append(
            {
                "image": image_id.stem,
                "plot_key": image_id.plot_key,
                "trial": image_id.trial,
                "block": image_id.block,
                "plot": image_id.plot,
                "leaf_no": image_id.leaf_no,
                "detection_id": i,
                "area_cm2": scale.px_to_cm2(m.area_px),
                "height_cm": m.height_px * scale.cm_per_px,
                "width_cm": m.width_px * scale.cm_per_px,
                "cm_per_px_aruco": aruco_cm_per_px,
                "cm_per_px_grid": grid_cm_per_px,
                "calibration_ratio_aruco_over_grid": calibration_ratio,
            }
        )
    df = pd.DataFrame(rows)

    out_dir = Path(cfg.output_dir) / "lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = leaves.annotate_image(image_np)
    save_annotated(annotated, out_dir / f"{image_id.stem}_leaves.png", title=image_id.stem)
    df.to_csv(out_dir / f"{image_id.stem}_leaves.csv", index=False)
    return df


def process_lab_directory(cfg: PipelineConfig, model: Sam3Model | None = None) -> pd.DataFrame:
    model = model or build_detector(cfg.model, cfg.detection.leaf_confidence)
    aruco_cal = ArucoCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm,
        dictionary=cfg.aruco.dictionary,
    )
    template_cal = TemplateMatchingCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm,
        dictionary=cfg.aruco.dictionary,
    )
    grid_cal = GridCalibrator(square_size_cm=cfg.grid.square_size_cm)

    frames: list[pd.DataFrame] = []
    for image_id in list_photos(cfg.photos_dir, kind="lab"):
        try:
            df = process_lab_image(image_id, model, aruco_cal, grid_cal, cfg, template_cal)
        except Exception as exc:  # noqa: BLE001
            df = _empty_with_warning(image_id, reason=f"error:{type(exc).__name__}:{exc}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_csv = Path(cfg.output_dir) / "lab" / "all_leaves.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)

    deduped = dedup_lab_sequences(combined)
    deduped.to_csv(out_csv.with_name("all_leaves_dedup.csv"), index=False)
    return deduped


def dedup_lab_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse multiple photos of the same sheet (same plot_key + leaf_no) into one row per leaf.

    Within each photo, leaves are ranked by area (largest=1). Across photos of the
    same sheet, leaves of the same rank are assumed to be the same physical leaf,
    so we average their area/height/width. Standard deviation across photos is
    reported in `*_std` columns as a measurement-consistency check."""
    if df.empty or "area_cm2" not in df.columns:
        return df
    valid = df.dropna(subset=["area_cm2"]).copy()
    if valid.empty:
        return df
    valid["rank"] = (
        valid.groupby("image")["area_cm2"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    agg = (
        valid.groupby(["plot_key", "trial", "block", "plot", "leaf_no", "rank"])
        .agg(
            area_cm2=("area_cm2", "mean"),
            area_cm2_std=("area_cm2", "std"),
            height_cm=("height_cm", "mean"),
            width_cm=("width_cm", "mean"),
            n_observations=("area_cm2", "count"),
        )
        .reset_index()
    )
    return agg


def _empty_with_warning(image_id: ImageId, reason: str) -> pd.DataFrame:
    print(f"[lab] {image_id.stem}: skipped ({reason})")
    return pd.DataFrame(
        [
            {
                "image": image_id.stem,
                "plot_key": image_id.plot_key,
                "trial": image_id.trial,
                "block": image_id.block,
                "plot": image_id.plot,
                "leaf_no": image_id.leaf_no,
                "warning": reason,
            }
        ]
    )
