"""Refine a rough homography by aligning the court's lines with the painted lines (C03).

Keypoints give a homography from a few points; if one is off by a metre, the
whole court is. The painted lines are a much richer signal: thousands of pixels
along the boundary, the lanes, the circles. This module

1. finds **line evidence** in the image: thin bright or dark strokes on the floor
   (morphological top-hat / black-hat), kept only if elongated and on the floor;
2. turns it into a **distance map** — for every pixel, the distance to the
   nearest line pixel (distance transform);
3. **projects the court's lines** through a candidate homography, samples them
   every few centimetres and measures how far each sample is from a painted line
   (a truncated chamfer distance);
4. **optimises** the homography to minimise that distance, coarse to fine.

The same measure answers the question C02 left open — *is this homography right?* —
as the share of projected line samples that land within a few pixels of a
painted line (``agreement``). A wrong homography cannot score high: its lines
fall on wood, on players, on the crowd.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize


@dataclass
class LineEvidence:
    mask: np.ndarray        # uint8 0/1, painted-line pixels
    dist: np.ndarray        # float32, distance (px) to the nearest line pixel
    valid: np.ndarray       # uint8 0/1, where line samples are allowed to count (the floor)
    angle: np.ndarray       # float32, direction (rad) of the line at the NEAREST line pixel


def line_evidence(img: np.ndarray, floor_hull: np.ndarray | None = None, people: np.ndarray | None = None,
                  kernel: int = 11, thresh: int = 28, min_len: int = 40) -> LineEvidence:
    """Painted lines on the floor, as a binary mask and a distance map.

    ``floor_hull``: polygon of the floor (e.g. from the colour CourtMask) — line
    pixels outside it (crowd, ads, score bar) are dropped. ``people``: (N, 4)
    boxes; their pixels are dropped too (jerseys and shoes are full of edges).
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    # top-hat = bright thin structures, black-hat = dark thin structures (lines narrower than the kernel)
    resp = np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k), cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k))
    m = (resp > thresh).astype(np.uint8)
    # keep elongated, sparse components (lines), drop compact blobs (heads, logos, numbers)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        w, h, a = st[i, 2], st[i, 3], st[i, 4]
        keep[i] = max(w, h) >= min_len and a / (w * h + 1e-6) < 0.5
    m = keep[lab].astype(np.uint8)

    valid = np.ones_like(m)
    if floor_hull is not None:
        valid = np.zeros_like(m)
        cv2.fillPoly(valid, [np.asarray(floor_hull, np.int32)], 1)
        valid = cv2.dilate(valid, np.ones((25, 25), np.uint8))
    if people is not None:
        for x1, y1, x2, y2 in np.asarray(people, int).reshape(-1, 4):
            valid[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    m &= valid
    # Direction of each line pixel: the intensity gradient is perpendicular to a thin line.
    gb = cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0)
    theta = np.arctan2(cv2.Sobel(gb, cv2.CV_32F, 0, 1), cv2.Sobel(gb, cv2.CV_32F, 1, 0)) + np.pi / 2
    dist, lab = cv2.distanceTransformWithLabels(1 - m, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(m)
    # labels index line pixels in raster order; map every pixel to its nearest line pixel's angle
    lut = np.zeros(len(xs) + 1, np.float32)
    order = np.argsort(lab[ys, xs])
    lut[lab[ys, xs][order]] = theta[ys, xs][order]
    return LineEvidence(m, dist, valid, lut[lab])


def sample_lines(polylines: list[np.ndarray], step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` metres along the court's polylines and their unit directions, (N, 2) each."""
    pts, dirs = [], []
    for pl in polylines:
        for a, b in zip(pl[:-1], pl[1:], strict=True):
            seg = b - a
            n = max(int(np.linalg.norm(seg) / step), 1)
            t = np.linspace(0, 1, n, endpoint=False)[:, None]
            pts.append(a + t * seg)
            dirs.append(np.repeat((seg / (np.linalg.norm(seg) + 1e-9))[None], n, axis=0))
    return np.vstack(pts), np.vstack(dirs)


def _project(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(1, -1, 2).astype(np.float64), H)[0]


def _visible(p, ev):
    h, w = ev.dist.shape
    ok = (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
    idx = np.where(ok)[0]
    q = p[idx].astype(int)
    keep = ev.valid[q[:, 1], q[:, 0]] > 0
    return idx[keep], q[keep]


def score(H: np.ndarray, samples: np.ndarray, ev: LineEvidence, tol: float = 4.0,
          tangents: np.ndarray | None = None, max_sin: float = 0.35) -> tuple[float, int]:
    """(agreement, n): share of visible projected samples on a painted line.

    A sample agrees when it is within ``tol`` px of a line pixel AND, if
    ``tangents`` (court-space direction of each sample) are given, the painted
    line runs in the same direction (|sin(angle difference)| < ``max_sin``).
    The direction test is what makes the score hard to fool: random clutter
    (players, wood grain, the crowd) is close to *some* stroke almost everywhere,
    but rarely to one pointing the same way.
    """
    p = _project(H, samples)
    idx, q = _visible(p, ev)
    if len(q) == 0:
        return 0.0, 0
    good = ev.dist[q[:, 1], q[:, 0]] <= tol
    if tangents is not None:
        t = _project(H, samples[idx] + 0.05 * tangents[idx]) - p[idx]
        a_proj = np.arctan2(t[:, 1], t[:, 0])
        good &= np.abs(np.sin(a_proj - ev.angle[q[:, 1], q[:, 0]])) < max_sin
    return float(good.mean()), int(len(q))


def _cost(H, samples, tangents, ev, tau, min_samples, max_sin=0.5):
    """Mean truncated distance to a painted line running in the same direction."""
    p = _project(H, samples)
    idx, q = _visible(p, ev)
    if len(q) < min_samples:
        return tau * 2.0                             # the court left the image: worse than any fit
    d = np.minimum(ev.dist[q[:, 1], q[:, 0]], tau)
    if tangents is not None:
        t = _project(H, samples[idx] + 0.05 * tangents[idx]) - p[idx]
        wrong_way = np.abs(np.sin(np.arctan2(t[:, 1], t[:, 0]) - ev.angle[q[:, 1], q[:, 0]])) >= max_sin
        d[wrong_way] = tau
    return float(d.mean())


def refine(H0: np.ndarray, samples: np.ndarray, ev: LineEvidence, tangents: np.ndarray | None = None,
           taus=(30.0, 12.0, 5.0), min_samples: int = 200) -> np.ndarray:
    """Move H so that the projected court lines sit on the painted lines.

    Parametrisation: the image positions of 4 court points spread over the
    VISIBLE part of the court (8 numbers, all in pixels — well conditioned,
    unlike the 9 raw entries of H). Optimised coarse to fine: a large truncation
    ``tau`` first lets distant lines pull the fit, a small one then polishes it.
    """
    h, w = ev.dist.shape
    p = _project(H0, samples)
    vis = samples[(p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)]
    if len(vis) < min_samples:
        return H0
    (x0, y0), (x1, y1) = vis.min(0), vis.max(0)
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)          # court (m)
    img0 = _project(H0, quad)                                                        # image (px)

    def homography_of(delta):
        dst = (img0 + delta.reshape(4, 2)).astype(np.float32)
        return cv2.getPerspectiveTransform(quad.astype(np.float32), dst)

    delta = np.zeros(8)
    for tau in taus:
        res = minimize(lambda d, t=tau: _cost(homography_of(d), samples, tangents, ev, t, min_samples), delta,
                       method="Powell", options={"xtol": 0.5, "ftol": 1e-4, "maxfev": 6000,
                                                 "direc": np.eye(8) * tau})
        delta = res.x
    return homography_of(delta)
