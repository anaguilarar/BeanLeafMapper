"""Field-image pipeline: calibrate with ArUco, detect leaves, filter to the main plant."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration import ArucoCalibrator, TemplateMatchingCalibrator
from ..detector import LeavesDetector
from ..io_utils import ImageId, list_photos, load_rgb
from ..model import Sam3Model, build_detector
from ..visualization import save_annotated, save_leaf_area_histogram


def process_field_image(
    image_id: ImageId,
    model: Sam3Model,
    calibrator: ArucoCalibrator,
    cfg,
    template_calibrator: TemplateMatchingCalibrator | None = None,
) -> pd.DataFrame:
    """Process one field photo. Returns a DataFrame of per-leaf metrics (may be empty)."""
    image = load_rgb(image_id.path, downscale=cfg.DETECTION.image_downscale)
    image_np = np.array(image)

    scale = calibrator.calibrate(image_np)
    if scale is None and template_calibrator is not None:
        scale = template_calibrator.calibrate(image_np)
    if scale is None:
        return _empty_with_warning(image_id, reason="calibration_failed")

    inference = model.detect(
        image,
        text_prompt=cfg.DETECTION.leaf_prompt,
        confidence_threshold=cfg.DETECTION.leaf_confidence,
    )

    leaves = LeavesDetector()
    leaves.set_sam3results(inference)
    if leaves.n_objects == 0:
        return _empty_with_warning(image_id, reason="no_leaves_detected")

    main_plant = leaves.main_plant_metrics(
        image_shape=image_np.shape,
        pixel_scale=scale,
        max_distance_cm=cfg.DETECTION.main_plant_max_distance_cm,
        cluster_eps_cm=cfg.DETECTION.main_plant_cluster_eps_cm,
        cluster_min_samples=cfg.DETECTION.main_plant_cluster_min_samples,
    )
    main_plant = leaves.filter_by_area_quantile(
        main_plant, quantile=cfg.DETECTION.min_area_quantile
    )
    main_plant = _attach_image_metadata(main_plant, image_id, scale_cm_per_px=scale.cm_per_px)

    out_dir = Path(cfg.GENERAL_INFO.output_dir) / "field"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = leaves.annotate_image(image_np, leaf_ids=main_plant["leaf_id"].tolist())
    save_annotated(annotated, out_dir / f"{image_id.stem}_leaves.png", title=image_id.stem)
    save_leaf_area_histogram(main_plant, out_dir / f"{image_id.stem}_hist.png", title=image_id.stem)
    main_plant.to_csv(out_dir / f"{image_id.stem}_leaves.csv", index=False)
    return main_plant


def process_field_directory(cfg, model: Sam3Model | None = None) -> pd.DataFrame:
    """Process every field image in cfg.photos_dir, return a concatenated DataFrame."""
    model = model or build_detector(cfg.MODEL, cfg.DETECTION.leaf_confidence)
    calibrator = ArucoCalibrator(
        marker_size_cm=cfg.ARUCO.marker_size_cm,
        dictionary=cfg.ARUCO.dictionary,
    )
    template_calibrator = TemplateMatchingCalibrator(
        marker_size_cm=cfg.ARUCO.marker_size_cm,
        dictionary=cfg.ARUCO.dictionary,
    )
    frames: list[pd.DataFrame] = []
    for image_id in list_photos(cfg.GENERAL_INFO.photos_dir, kind="field"):
        try:
            df = process_field_image(image_id, model, calibrator, cfg, template_calibrator)
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            df = _empty_with_warning(image_id, reason=f"error:{type(exc).__name__}:{exc}")
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_csv = Path(cfg.GENERAL_INFO.output_dir) / "field" / "all_leaves.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    return combined


def _attach_image_metadata(df: pd.DataFrame, image_id: ImageId, scale_cm_per_px: float) -> pd.DataFrame:
    if df.empty:
        df = df.assign(image=image_id.stem, plot_key=image_id.plot_key, cm_per_px=scale_cm_per_px)
        return df
    df = df.copy()
    df["image"] = image_id.stem
    df["plot_key"] = image_id.plot_key
    df["trial"] = image_id.trial
    df["block"] = image_id.block
    df["plot"] = image_id.plot
    df["cm_per_px"] = scale_cm_per_px
    return df


def _empty_with_warning(image_id: ImageId, reason: str) -> pd.DataFrame:
    print(f"[field] {image_id.stem}: skipped ({reason})")
    return pd.DataFrame(
        [
            {
                "image": image_id.stem,
                "plot_key": image_id.plot_key,
                "trial": image_id.trial,
                "block": image_id.block,
                "plot": image_id.plot,
                "leaf_id": None,
                "area_cm2": None,
                "warning": reason,
            }
        ]
    )
