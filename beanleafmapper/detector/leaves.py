"""Leaf-specific detector built on top of the base Detector.

Adds:
- Distance-from-image-centre to identify leaves belonging to the *main* plant
  (the one centred in the frame), which is the plant we actually want to phenotype.
- Per-leaf area in cm² using a provided pixel-to-cm scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

from ..calibration.aruco import PixelScale
from .base import Detector


def _keep_centre_cluster(
    df: pd.DataFrame,
    image_shape: tuple[int, int],
    pixel_scale: PixelScale,
    eps_cm: float,
    min_samples: int,
) -> pd.DataFrame:
    """DBSCAN-cluster leaf centroids in cm-space, keep cluster closest to image centre."""
    from sklearn.cluster import DBSCAN

    if df.empty:
        return df
    coords_px = df[["center_x_px", "center_y_px"]].to_numpy()
    coords_cm = coords_px * pixel_scale.cm_per_px
    eps_px = eps_cm / pixel_scale.cm_per_px

    labels = DBSCAN(eps=eps_px, min_samples=min_samples).fit_predict(coords_px)
    df = df.assign(cluster=labels)

    valid = df[df["cluster"] != -1]
    if valid.empty:
        return df.head(0)

    h, w = image_shape[:2]
    img_centre = np.array([w / 2.0, h / 2.0])
    cluster_distances: dict[int, float] = {}
    for cid in valid["cluster"].unique():
        members = valid[valid["cluster"] == cid][["center_x_px", "center_y_px"]].to_numpy()
        centroid = members.mean(axis=0)
        cluster_distances[int(cid)] = float(np.linalg.norm(centroid - img_centre))
    best_cluster = min(cluster_distances, key=cluster_distances.get)
    return valid[valid["cluster"] == best_cluster].drop(columns=["cluster"])


@dataclass(frozen=True)
class LeafMetric:
    leaf_id: int
    area_cm2: float
    height_cm: float
    width_cm: float
    center_x_px: float
    center_y_px: float
    distance_to_center_cm: float


class LeavesDetector(Detector):
    """Detects leaves and filters to the main (centre-most) plant."""

    def distances_to_center(
        self,
        image_shape: tuple[int, int],
        pixel_scale: PixelScale,
    ) -> np.ndarray:
        """Distance (cm) from each detected leaf's centre to the image centre."""
        h, w = image_shape[:2]
        img_center = np.array([w / 2.0, h / 2.0])
        distances = np.zeros(self.n_objects, dtype=np.float64)
        for i in range(self.n_objects):
            metrics = self.calculate_oneobject_metrics(i)
            distances[i] = np.linalg.norm(np.array(metrics.center_px) - img_center)
        return distances * pixel_scale.cm_per_px

    def main_plant_metrics(
        self,
        image_shape: tuple[int, int],
        pixel_scale: PixelScale,
        max_distance_cm: float | None = None,
        cluster_eps_cm: float | None = None,
        cluster_min_samples: int = 3,
    ) -> pd.DataFrame:
        """Per-leaf metrics, restricted to leaves on the main plant.

        Filter strategy:
          - If `cluster_eps_cm` is given, DBSCAN-cluster leaf centroids and keep
            only the cluster whose centroid is closest to the image centre.
          - Else if `max_distance_cm` is given, keep leaves within that radius
            of the image centre.
          - Otherwise, return all detections.
        """
        rows: list[dict] = []
        distances = self.distances_to_center(image_shape, pixel_scale)
        for i in range(self.n_objects):
            try:
                m = self.calculate_oneobject_metrics(i)
            except ValueError:
                continue
            rows.append(
                {
                    "leaf_id": i,
                    "area_cm2": pixel_scale.px_to_cm2(m.area_px),
                    "height_cm": m.height_px * pixel_scale.cm_per_px,
                    "width_cm": m.width_px * pixel_scale.cm_per_px,
                    "center_x_px": m.center_px[0],
                    "center_y_px": m.center_px[1],
                    "distance_to_center_cm": float(distances[i]),
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if cluster_eps_cm is not None:
            df = _keep_centre_cluster(
                df, image_shape, pixel_scale, eps_cm=cluster_eps_cm, min_samples=cluster_min_samples
            )
            return df.reset_index(drop=True)
        if max_distance_cm is not None:
            return df[df["distance_to_center_cm"] <= max_distance_cm].reset_index(drop=True)
        return df

    def filter_by_area_quantile(self, df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
        """Keep leaves whose area is above the given quantile of the set.

        Used to drop tiny fragmentary detections from a focused plant of interest."""
        if df.empty:
            return df
        threshold = float(df["area_cm2"].quantile(quantile))
        return df[df["area_cm2"] >= threshold].reset_index(drop=True)

    def annotate_image(
        self,
        image: np.ndarray,
        leaf_ids: list[int] | None = None,
        color: tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Draw mask contours and bboxes on a copy of the image."""
        out = image.copy()
        ids = leaf_ids if leaf_ids is not None else list(range(self.n_objects))
        for i in ids:
            contour = self.find_contour(i, pad=None)
            if contour.size:
                cv2.drawContours(out, [contour], -1, color, 2)
            x1, y1, x2, y2 = self.get_bbox(i)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                out,
                f"#{i}",
                (x1, max(y1 - 5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return out
