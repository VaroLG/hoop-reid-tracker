"""The court AREA as an image mask, from a homography (court chapter, C06).

C06 trains a model to segment the playing area — everything inside the
sidelines and baselines — directly from the image, so that the mask follows
the real court in any arena without first finding a homography. Its training
labels come from homographies (calibrated cameras, verified keypoints, verified
propagated chains) and synthetic renders; this module turns a homography into
that label.

One subtlety: with a wide or side camera, part of the court can lie BEHIND the
camera. Projecting such corners gives nonsense (they flip to the other side of
the image). So the court rectangle is first clipped, in court coordinates, to
the half-plane in front of the camera, and only then projected and clipped to
the image frame.
"""

from __future__ import annotations

import cv2
import numpy as np

from .leagues import CourtSpec


def _clip_halfplane(poly: np.ndarray, a: np.ndarray, eps: float) -> np.ndarray:
    """Sutherland–Hodgman: keep the part of ``poly`` (N, 2) where a·(x, y, 1) >= eps."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        dp, dq = a @ (*p, 1.0) - eps, a @ (*q, 1.0) - eps
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append(p + t * (q - p))
    return np.array(out, float).reshape(-1, 2)


def court_polygon(H: np.ndarray, spec: CourtSpec, image_size, margin: float = 0.0) -> np.ndarray:
    """Image polygon (K, 2) of the court area (+ ``margin`` metres), clipped to the frame. Empty if not visible."""
    w, h = image_size
    m = margin
    court = np.array([[-m, -m], [spec.length + m, -m], [spec.length + m, spec.width + m], [-m, spec.width + m]])
    # Depth of a court point = third row of H, up to a scale whose SIGN is unknown (H is only
    # defined up to scale). The floor point seen at the bottom centre of the image (always floor
    # in a broadcast) is certainly in front of the camera: its sign is "in front".
    c = np.linalg.inv(H) @ np.array([w / 2, h - 1, 1.0])
    sign = np.sign(H[2] @ (c / c[2])) or 1.0
    poly = _clip_halfplane(court, sign * H[2] / np.linalg.norm(H[2]), 1e-6)
    if len(poly) < 3:
        return np.zeros((0, 2))
    p = cv2.perspectiveTransform(poly[None].astype(np.float64), H)[0]
    p = np.clip(p, -50 * w, 50 * w).astype(np.float32)
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    hull = cv2.convexHull(p)                       # the projected court is convex: guards against vertex order
    area, inter = cv2.intersectConvexConvex(hull, frame)
    return np.zeros((0, 2)) if inter is None or area <= 1 else inter.reshape(-1, 2).astype(float)


def court_mask(H: np.ndarray, spec: CourtSpec, image_size, margin: float = 0.0) -> np.ndarray:
    """uint8 mask (h, w): 1 inside the court area, 0 outside."""
    w, h = image_size
    mask = np.zeros((h, w), np.uint8)
    poly = court_polygon(H, spec, image_size, margin)
    if len(poly) >= 3:
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask


def yolo_seg_line(poly: np.ndarray, image_size, cls: int = 0) -> str:
    """One line of a YOLO segmentation label: ``cls x1 y1 x2 y2 ...`` normalised to [0, 1]."""
    w, h = image_size
    p = np.clip(poly / [w, h], 0, 1)
    return f"{cls} " + " ".join(f"{v:.6f}" for v in p.ravel())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    return float(inter / union) if union else 1.0
