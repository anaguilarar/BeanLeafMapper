"""Template-matching fallback for ArUco/AprilTag markers printed without a quiet zone.

We generate the canonical ArUco/AprilTag marker bitmap for a given (dictionary, id)
and slide it across the photo at several scales and rotations, picking the best
normalized cross-correlation peak. The detected side length in pixels divides the
known marker side in cm to give cm/px.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .aruco import PixelScale, _DICT_LOOKUP


@dataclass(frozen=True)
class TemplateMatch:
    score: float
    side_px: float
    top_left: tuple[int, int]
    angle_deg: float


class TemplateMatchingCalibrator:
    """Fallback detector that matches a known marker template against the photo."""

    def __init__(
        self,
        marker_size_cm: float,
        dictionary: str = "DICT_APRILTAG_16h5",
        marker_id: int = 0,
        min_side_px: int = 60,
        max_side_px: int = 400,
        n_scales: int = 18,
        angle_step_deg: float = 15.0,
        score_threshold: float = 0.50,
    ):
        if dictionary not in _DICT_LOOKUP:
            raise ValueError(f"Unknown dictionary {dictionary!r}")
        self.marker_size_cm = marker_size_cm
        self.marker_id = marker_id
        self._dict = cv2.aruco.getPredefinedDictionary(_DICT_LOOKUP[dictionary])
        self.min_side_px = min_side_px
        self.max_side_px = max_side_px
        self.n_scales = n_scales
        self.angle_step_deg = angle_step_deg
        self.score_threshold = score_threshold
        # Reference at high res, no quiet zone (matches markers printed without border)
        self._reference = cv2.aruco.generateImageMarker(self._dict, marker_id, 256)

    def calibrate(self, image: np.ndarray) -> PixelScale | None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        match = self._best_match(gray)
        if match is None or match.score < self.score_threshold:
            return None
        return PixelScale(
            cm_per_px=self.marker_size_cm / match.side_px,
            source=f"template(score={match.score:.2f})",
        )

    def find(self, image: np.ndarray) -> TemplateMatch | None:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        return self._best_match(gray)

    def _best_match(self, gray: np.ndarray) -> TemplateMatch | None:
        sizes = np.linspace(self.min_side_px, self.max_side_px, self.n_scales).astype(int)
        angles = np.arange(0.0, 360.0, self.angle_step_deg) if self.angle_step_deg > 0 else [0.0]

        best: TemplateMatch | None = None
        for side in sizes:
            template_base = cv2.resize(self._reference, (side, side), interpolation=cv2.INTER_AREA)
            if template_base.shape[0] > gray.shape[0] or template_base.shape[1] > gray.shape[1]:
                continue
            for angle in angles:
                rotated = _rotate_with_padding(template_base, float(angle))
                if rotated.shape[0] > gray.shape[0] or rotated.shape[1] > gray.shape[1]:
                    continue
                res = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if best is None or max_val > best.score:
                    best = TemplateMatch(
                        score=float(max_val),
                        side_px=float(side),
                        top_left=(int(max_loc[0]), int(max_loc[1])),
                        angle_deg=float(angle),
                    )
        return best


def _rotate_with_padding(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around its centre and expand the canvas so corners aren't cropped."""
    if abs(angle_deg) < 1e-6:
        return image
    h, w = image.shape[:2]
    centre = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos = abs(rot[0, 0])
    sin = abs(rot[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    rot[0, 2] += (new_w - w) / 2.0
    rot[1, 2] += (new_h - h) / 2.0
    return cv2.warpAffine(image, rot, (new_w, new_h), borderValue=255)
