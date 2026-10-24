"""Run the C06 court-area segmentation model on a frame (court chapter, C06).

The model predicts instance masks of class ``court``. A frame has one court, so
all predicted instances above ``conf`` are merged. Two optional clean-ups,
both following from the geometry of a court seen by a pinhole camera:

* ``convex``: the image of a rectangle on a plane is convex. Players standing
  on the boundary, or a logo the model hesitated on, punch holes / notches in
  the raw mask; the convex hull of the largest region removes them.
* ``smooth``: the mask moves smoothly with the camera; an exponential average
  over frames (``alpha``) removes flicker. Off by default (it lags on cuts).
"""

from __future__ import annotations

import cv2
import numpy as np


class CourtSegmenter:
    def __init__(self, weights: str, imgsz: int = 512, conf: float = 0.25, device: str = "cpu",
                 convex: bool = True, smooth: float = 0.0) -> None:
        # imported here: the rest of the chapter does not need torch
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz, self.conf, self.device = imgsz, conf, device
        self.convex, self.smooth = convex, smooth
        self._avg: np.ndarray | None = None

    def reset(self) -> None:
        self._avg = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """uint8 mask (h, w): 1 = court area."""
        h, w = frame.shape[:2]
        r = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False,
                               retina_masks=True)[0]
        mask = np.zeros((h, w), np.uint8)
        if r.masks is not None and len(r.masks):
            m = r.masks.data.cpu().numpy() > 0.5
            mask = (m.any(0)).astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if self.convex:
            mask = convex_largest(mask)
        if self.smooth > 0:
            f = mask.astype(np.float32)
            self._avg = f if self._avg is None or self._avg.shape != f.shape else \
                self.smooth * self._avg + (1 - self.smooth) * f
            mask = (self._avg >= 0.5).astype(np.uint8)
        return mask


def convex_largest(mask: np.ndarray) -> np.ndarray:
    """Convex hull of the largest connected region (empty mask stays empty)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n < 2:
        return mask.astype(np.uint8)
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask, np.uint8)
    cv2.fillPoly(out, [cv2.convexHull(max(cnts, key=cv2.contourArea))], 1)
    return out


def feet_inside(mask: np.ndarray, boxes: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """True for boxes whose feet (bottom centre) are inside ``mask``, allowing ``margin`` x box height outside."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return np.zeros(0, bool)
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros(len(boxes), bool)
    out = np.zeros(len(boxes), bool)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        foot = (float((x1 + x2) / 2), float(min(y2, h - 1)))
        d = max(cv2.pointPolygonTest(c, foot, True) for c in cnts)
        out[i] = d >= -margin * (y2 - y1)
    return out
