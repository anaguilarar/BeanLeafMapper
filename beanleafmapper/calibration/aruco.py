"""ArUco-based pixel size calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_DICT_LOOKUP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


@dataclass(frozen=True)
class PixelScale:
    """cm per pixel, derived from an object of known physical size."""

    cm_per_px: float
    source: str

    def px_to_cm2(self, area_px: float) -> float:
        return area_px * (self.cm_per_px ** 2)


class ArucoCalibrator:
    """Detects an ArUco marker and computes cm-per-pixel from its known side length."""

    def __init__(self, marker_size_cm: float, dictionary: str = "DICT_4X4_50"):
        if dictionary not in _DICT_LOOKUP:
            raise ValueError(
                f"Unknown ArUco dictionary {dictionary!r}; supported: {sorted(_DICT_LOOKUP)}"
            )
        self.marker_size_cm = marker_size_cm
        self._dict = cv2.aruco.getPredefinedDictionary(_DICT_LOOKUP[dictionary])
        self._params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

    def calibrate(self, image: np.ndarray) -> PixelScale | None:
        """Return the median cm-per-pixel across detected markers, or None if none found."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return None

        side_lengths_px = []
        for c in corners:
            pts = c.reshape(4, 2)
            for i in range(4):
                side_lengths_px.append(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))
        median_side = float(np.median(side_lengths_px))
        if median_side <= 0:
            return None
        return PixelScale(
            cm_per_px=self.marker_size_cm / median_side,
            source="aruco",
        )
