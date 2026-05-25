"""Base detector: wraps SAM3 inference state into per-object mask/bbox accessors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..model import Sam3Inference


def _pad_mask(mask: np.ndarray, padding: int = 2) -> np.ndarray:
    padded = np.zeros(
        (mask.shape[0] + padding, mask.shape[1] + padding), dtype=np.uint8
    )
    padded[padding // 2 : -padding // 2, padding // 2 : -padding // 2] = mask
    return padded


def _euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


@dataclass(frozen=True)
class ObjectMetrics:
    object_id: int
    height_px: float
    width_px: float
    area_px: float
    center_px: tuple[float, float]


class Detector:
    """Holds SAM3 results and exposes per-object geometry.

    bbs: (N, 4) array of XYXY boxes (in image-pixel coords of the inference image).
    masks: list of binary masks per object.
    """

    def __init__(self) -> None:
        self.bbs: np.ndarray = np.empty((0, 4))
        self.masks: list[np.ndarray] = []
        self.scores: np.ndarray = np.empty((0,))

    def set_sam3results(self, inference: Sam3Inference) -> None:
        self.bbs = inference.boxes.detach().cpu().numpy().reshape(-1, 4)
        self.scores = inference.scores.detach().cpu().numpy().reshape(-1)
        masks_t = inference.masks.detach().cpu().numpy()
        if masks_t.ndim == 4:
            masks_t = masks_t.squeeze(1)
        elif masks_t.ndim == 2:
            masks_t = masks_t[None, :, :]
        self.masks = [(m > 0.5).astype(np.uint8) for m in masks_t]

    @property
    def n_objects(self) -> int:
        return self.bbs.shape[0]

    def get_bbox(self, idx: int) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbs[idx]
        return int(x1), int(y1), int(x2), int(y2)

    def get_mask(self, idx: int, pad: int | None = 2) -> np.ndarray:
        mask = self.masks[idx]
        if pad:
            return _pad_mask(mask, padding=pad)
        return mask

    def clip_image(self, image: np.ndarray, idx: int) -> np.ndarray:
        x1, y1, x2, y2 = self.get_bbox(idx)
        return image[max(y1, 0) : y2, max(x1, 0) : x2].copy()

    def find_contour(self, idx: int, pad: int | None = 2) -> np.ndarray:
        mask = self.get_mask(idx, pad=pad)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.empty((0, 1, 2), dtype=np.int32)
        return max(contours, key=cv2.contourArea)

    def calculate_oneobject_metrics(self, idx: int) -> ObjectMetrics:
        contour = self.find_contour(idx)
        if contour.size == 0:
            raise ValueError(f"No contour for object {idx}")
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w_px, h_px), _ = rect
        area_px = float(cv2.contourArea(contour))
        return ObjectMetrics(
            object_id=idx,
            height_px=float(max(w_px, h_px)),
            width_px=float(min(w_px, h_px)),
            area_px=area_px,
            center_px=(float(cx), float(cy)),
        )

    def largest_object_index(self) -> int:
        areas = [float(cv2.contourArea(self.find_contour(i))) for i in range(self.n_objects)]
        if not areas:
            raise ValueError("No objects detected")
        return int(np.argmax(areas))
