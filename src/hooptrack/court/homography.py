"""Image <-> court homographies from detected keypoints (court chapter, C02+).

A homography H (3 x 3) maps points of one plane to another plane seen in
perspective. Here: court plane (metres) -> image (pixels):

    [u, v, w]^T = H [x, y, 1]^T ,   pixel = (u / w, v / w)

Four point pairs (no three on one line) determine H exactly; with more we fit
it robustly with RANSAC, which ignores keypoints the detector got wrong.
Only FLOOR keypoints are used: the rims (09, 33) are 3 m above the plane.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .template import CourtTemplate


@dataclass
class CourtHomography:
    H: np.ndarray                   # court (m) -> image (px)
    inliers: list[str]              # keypoints RANSAC agreed with
    reproj_px: float                # median reprojection error of the inliers

    def to_image(self, court_xy: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(court_xy, float).reshape(1, -1, 2), self.H)[0]

    def to_court(self, image_xy: np.ndarray) -> np.ndarray:
        return cv2.perspectiveTransform(np.asarray(image_xy, float).reshape(1, -1, 2), np.linalg.inv(self.H))[0]


def fit_homography(names: list[str], xy: np.ndarray, conf: np.ndarray, template: CourtTemplate,
                   conf_thr: float = 0.5, ransac_px: float = 8.0, min_points: int = 4) -> CourtHomography | None:
    """Fit court -> image from keypoints detected with confidence >= ``conf_thr``.

    Returns None when fewer than ``min_points`` floor keypoints are available or
    the points are degenerate (e.g. all on one line).
    """
    keep = [i for i, n in enumerate(names)
            if conf[i] >= conf_thr and n in template.points and n not in template.non_floor]
    if len(keep) < min_points:
        return None
    court = np.array([template.points[names[i]] for i in keep], float)
    img = np.asarray(xy, float)[keep]
    H, mask = cv2.findHomography(court, img, cv2.RANSAC, ransac_px)
    if H is None or mask is None or mask.sum() < min_points:
        return None
    mask = mask.ravel().astype(bool)
    proj = cv2.perspectiveTransform(court[None], H)[0]
    err = np.linalg.norm(proj - img, axis=1)
    return CourtHomography(H, [names[keep[i]] for i in np.where(mask)[0]], float(np.median(err[mask])))
