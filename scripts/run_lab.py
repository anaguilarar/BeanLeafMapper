"""Process every lab photo: ArUco + grid calibration, SAM3 leaf detection, per-leaf area."""

from __future__ import annotations

import argparse
from pathlib import Path

from beanleafmapper.config import PipelineConfig
from beanleafmapper.pipeline.lab import process_lab_directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--marker-size-cm", type=float, default=None)
    parser.add_argument("--aruco-dict", type=str, default=None)
    parser.add_argument("--grid-square-cm", type=float, default=None)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--downscale", type=int, default=None)
    parser.add_argument("--model-backend", type=str, default=None,
                        help="Detector backend (default: sam3_image; see README).")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Path to a local SAM3 checkpoint .pt; default downloads from HF.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--compile", action="store_true",
                        help="Enable torch.compile (~2x on Ampere+).")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.photos_dir:
        cfg.photos_dir = args.photos_dir
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.marker_size_cm is not None:
        cfg.aruco.marker_size_cm = args.marker_size_cm
    if args.aruco_dict:
        cfg.aruco.dictionary = args.aruco_dict
    if args.grid_square_cm is not None:
        cfg.grid.square_size_cm = args.grid_square_cm
    if args.confidence is not None:
        cfg.detection.leaf_confidence = args.confidence
    if args.downscale is not None:
        cfg.detection.image_downscale = args.downscale
    if args.model_backend:
        cfg.model.backend = args.model_backend
    if args.checkpoint_path:
        cfg.model.checkpoint_path = args.checkpoint_path
    if args.device:
        cfg.model.device = args.device
    if args.compile:
        cfg.model.compile = True

    df = process_lab_directory(cfg)
    print(f"Processed {df['image'].nunique() if not df.empty else 0} lab images.")
    print(f"Wrote outputs to {cfg.output_dir / 'lab'}")


if __name__ == "__main__":
    main()
