"""End-to-end smoke test on ONE field photo and ONE lab photo.

Runs in two modes:

  --stub        Inject a synthetic SAM3-like inference (no model needed).
                Verifies calibration, detector, metrics, CSV/PNG output, and
                validation report. Use this to confirm the wiring is sound
                before pulling SAM3 weights.

  --real        Load the actual SAM3 model and run text-prompt detection.
                Requires `pip install 'git+https://github.com/facebookresearch/sam3.git'`
                and a valid HF token for the gated weights.

Default is --stub.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

from beanleafmapper.calibration import ArucoCalibrator, GridCalibrator, TemplateMatchingCalibrator
from beanleafmapper.config import PipelineConfig
from beanleafmapper.detector import LeavesDetector
from beanleafmapper.io_utils import ImageId, load_rgb, parse_filename
from beanleafmapper.model import Sam3Inference
from beanleafmapper.pipeline.field import _attach_image_metadata
from beanleafmapper.pipeline.validation import build_validation_report
from beanleafmapper.visualization import save_annotated, save_leaf_area_histogram


FIELD_PHOTO = "photos_iaf_20260328/A77-B1-S1-1.1.jpeg"
LAB_PHOTO = "photos_iaf_20260328/A77-B1-S1-H1-1.1.jpeg"


def make_stub_inference(image_shape: tuple[int, int], n_leaves: int = 5) -> Sam3Inference:
    """Generate a fake Sam3Inference with elliptical 'leaves' near the image centre."""
    h, w = image_shape[:2]
    cx, cy = w / 2, h / 2

    boxes = []
    masks = []
    scores = []
    rng = np.random.default_rng(42)
    for i in range(n_leaves):
        radius = rng.uniform(min(h, w) * 0.05, min(h, w) * 0.10)
        offset_x = rng.uniform(-min(h, w) * 0.15, min(h, w) * 0.15)
        offset_y = rng.uniform(-min(h, w) * 0.15, min(h, w) * 0.15)
        ecx, ecy = cx + offset_x, cy + offset_y
        ax = radius * 1.4
        ay = radius * 0.9
        mask = np.zeros((h, w), dtype=np.float32)
        yy, xx = np.ogrid[:h, :w]
        ellipse = ((xx - ecx) ** 2) / (ax ** 2) + ((yy - ecy) ** 2) / (ay ** 2) <= 1
        mask[ellipse] = 1.0
        masks.append(mask)
        x1, y1 = ecx - ax, ecy - ay
        x2, y2 = ecx + ax, ecy + ay
        boxes.append([x1, y1, x2, y2])
        scores.append(0.9 - i * 0.05)

    return Sam3Inference(
        scores=torch.tensor(scores),
        boxes=torch.tensor(boxes),
        masks=torch.tensor(np.stack(masks))[:, None, :, :],
        raw={},
    )


def smoke_field(cfg: PipelineConfig, inference: Sam3Inference | None) -> pd.DataFrame:
    image_id = parse_filename(FIELD_PHOTO)
    print(f"\n[field] {image_id.stem} (plot_key={image_id.plot_key})")

    image = load_rgb(image_id.path, downscale=cfg.detection.image_downscale)
    image_np = np.array(image)
    print(f"        loaded image, shape={image_np.shape}")

    aruco = ArucoCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    ).calibrate(image_np)
    if aruco is None:
        template = TemplateMatchingCalibrator(
            marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
        ).calibrate(image_np)
        scale = template
    else:
        scale = aruco
    if scale is None:
        print("        no calibration found, falling back to stub 0.05 cm/px")
        from beanleafmapper.calibration import PixelScale

        scale = PixelScale(cm_per_px=0.05, source="fallback_stub")
    print(f"        calibration: {scale.cm_per_px:.5f} cm/px ({scale.source})")

    if inference is None:
        inference = make_stub_inference(image_np.shape, n_leaves=8)
        print(f"        using stub inference ({inference.boxes.shape[0]} synthetic leaves)")

    leaves = LeavesDetector()
    leaves.set_sam3results(inference)
    main_plant = leaves.main_plant_metrics(
        image_shape=image_np.shape,
        pixel_scale=scale,
        max_distance_cm=cfg.detection.main_plant_max_distance_cm,
        cluster_eps_cm=cfg.detection.main_plant_cluster_eps_cm,
        cluster_min_samples=cfg.detection.main_plant_cluster_min_samples,
    )
    filter_desc = (
        f"DBSCAN(eps={cfg.detection.main_plant_cluster_eps_cm} cm)"
        if cfg.detection.main_plant_cluster_eps_cm is not None
        else f"radius {cfg.detection.main_plant_max_distance_cm} cm"
    )
    print(f"        leaves in main-plant cluster ({filter_desc}): {len(main_plant)}")
    main_plant = leaves.filter_by_area_quantile(main_plant, quantile=cfg.detection.min_area_quantile)
    print(f"        after area-quantile filter (q={cfg.detection.min_area_quantile}): {len(main_plant)}")
    main_plant = _attach_image_metadata(main_plant, image_id, scale_cm_per_px=scale.cm_per_px)

    out_dir = Path(cfg.output_dir) / "field"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = leaves.annotate_image(image_np, leaf_ids=main_plant["leaf_id"].dropna().astype(int).tolist())
    save_annotated(annotated, out_dir / f"{image_id.stem}_leaves.png", title=image_id.stem)
    save_leaf_area_histogram(main_plant, out_dir / f"{image_id.stem}_hist.png", title=image_id.stem)
    main_plant.to_csv(out_dir / f"{image_id.stem}_leaves.csv", index=False)
    print(f"        wrote {out_dir / (image_id.stem + '_leaves.csv')}")
    return main_plant


def smoke_lab(cfg: PipelineConfig, inference: Sam3Inference | None) -> pd.DataFrame:
    image_id = parse_filename(LAB_PHOTO)
    print(f"\n[lab] {image_id.stem} (plot_key={image_id.plot_key}, leaf_no={image_id.leaf_no})")

    image = load_rgb(image_id.path, downscale=cfg.detection.image_downscale)
    image_np = np.array(image)
    print(f"      loaded image, shape={image_np.shape}")

    aruco_cal = ArucoCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    template_cal = TemplateMatchingCalibrator(
        marker_size_cm=cfg.aruco.marker_size_cm, dictionary=cfg.aruco.dictionary
    )
    grid_cal = GridCalibrator(square_size_cm=cfg.grid.square_size_cm)
    aruco_scale = aruco_cal.calibrate(image_np) or template_cal.calibrate(image_np)
    grid_scale = grid_cal.calibrate(image_np)
    print(f"      aruco/template: {aruco_scale}")
    print(f"      grid:           {grid_scale}")
    scale = aruco_scale or grid_scale
    if scale is None:
        print("      no calibration → using fallback 0.05 cm/px")
        from beanleafmapper.calibration import PixelScale

        scale = PixelScale(cm_per_px=0.05, source="fallback_stub")

    if inference is None:
        inference = make_stub_inference(image_np.shape, n_leaves=3)
        print(f"      using stub inference ({inference.boxes.shape[0]} synthetic leaves)")

    leaves = LeavesDetector()
    leaves.set_sam3results(inference)
    print(f"      detected {leaves.n_objects} objects")

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
                "cm_per_px_aruco": aruco_scale.cm_per_px if aruco_scale else None,
                "cm_per_px_grid": grid_scale.cm_per_px if grid_scale else None,
                "calibration_ratio_aruco_over_grid": (
                    aruco_scale.cm_per_px / grid_scale.cm_per_px
                    if (aruco_scale and grid_scale) else None
                ),
            }
        )
    df = pd.DataFrame(rows)
    out_dir = Path(cfg.output_dir) / "lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    annotated = leaves.annotate_image(image_np)
    save_annotated(annotated, out_dir / f"{image_id.stem}_leaves.png", title=image_id.stem)
    df.to_csv(out_dir / f"{image_id.stem}_leaves.csv", index=False)
    print(f"      wrote {out_dir / (image_id.stem + '_leaves.csv')} ({len(df)} rows)")
    return df


def real_sam3_inference(image: Image.Image, prompt: str, confidence: float) -> Sam3Inference:
    from beanleafmapper.model import Sam3Model

    model = Sam3Model(confidence_threshold=confidence)
    return model.detect(image, text_prompt=prompt, confidence_threshold=confidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="Run the actual SAM3 model.")
    parser.add_argument("--downscale", type=int, default=None, help="Override image downscale.")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.downscale is not None:
        cfg.detection.image_downscale = args.downscale

    if args.real:
        try:
            import sam3  # noqa: F401
        except ImportError:
            print("SAM3 is not installed. Install with:")
            print("  pip install 'git+https://github.com/facebookresearch/sam3.git'")
            return 2

        field_image = load_rgb(FIELD_PHOTO, downscale=cfg.detection.image_downscale)
        lab_image = load_rgb(LAB_PHOTO, downscale=cfg.detection.image_downscale)
        from beanleafmapper.model import Sam3Model

        model = Sam3Model(confidence_threshold=cfg.detection.leaf_confidence)
        print("Running real SAM3 on field image…")
        field_inf = model.detect(field_image, "leaf", cfg.detection.leaf_confidence)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("Running real SAM3 on lab image…")
        lab_inf = model.detect(lab_image, "leaf", cfg.detection.leaf_confidence)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        field_inf = None
        lab_inf = None

    field_df = smoke_field(cfg, field_inf)
    lab_df = smoke_lab(cfg, lab_inf)

    print("\n[validation]")
    report = build_validation_report(field_df, lab_df, cfg.output_dir / "validation")
    print(report.to_string(index=False))
    print(f"\nAll outputs under: {cfg.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
