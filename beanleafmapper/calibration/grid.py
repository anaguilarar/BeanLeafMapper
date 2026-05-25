"""Graph-paper grid calibration (lab photos only).

Estimates pixel size by finding the periodic spacing of the small dark grid lines on
the lab paper. Uses an FFT of horizontal/vertical gradient projections to recover the
dominant period in pixels, then divides by the known physical square size."""

from __future__ import annotations

import cv2
import numpy as np

from .aruco import PixelScale


class GridCalibrator:
    """Detect graph-paper grid spacing and compute cm-per-pixel."""

    def __init__(self, square_size_cm: float = 0.5, min_period_px: int = 4, max_period_px: int = 60):
        self.square_size_cm = square_size_cm
        self.min_period_px = min_period_px
        self.max_period_px = max_period_px

    def calibrate(self, image: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> PixelScale | None:
        """Return PixelScale from grid period detection.

        roi: (x, y, w, h) crop of the graph-paper region. If None, uses the full image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        if roi is not None:
            x, y, w, h = roi
            gray = gray[y : y + h, x : x + w]

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

        period_x = self._dominant_period(gx.sum(axis=0))
        period_y = self._dominant_period(gy.sum(axis=1))
        if period_x is None and period_y is None:
            return None
        periods = [p for p in (period_x, period_y) if p is not None]
        median_period = float(np.median(periods))
        return PixelScale(
            cm_per_px=self.square_size_cm / median_period,
            source="grid",
        )

    def _dominant_period(self, signal: np.ndarray) -> float | None:
        signal = signal - signal.mean()
        if signal.size < 2 * self.max_period_px:
            return None
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size)
        with np.errstate(divide="ignore"):
            periods = np.where(freqs > 0, 1.0 / freqs, np.inf)
        mask = (periods >= self.min_period_px) & (periods <= self.max_period_px)
        if not np.any(mask):
            return None
        idx = np.argmax(spectrum * mask)
        return float(periods[idx])
