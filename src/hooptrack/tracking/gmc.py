"""Global Motion Compensation (GMC): estimate how the CAMERA moved.

Broadcast basketball cameras pan to follow the ball and zoom on plays. If the
camera pans 40 px to the right, every player appears to move 40 px to the
left even when standing still. A Kalman filter that ignores this predicts
boxes in the wrong place, IoU drops to zero, and tracks are lost.

Method (sparse, fast, CPU-friendly):
  1. Detect good corner features in the previous frame.
  2. Mask out the players: they move on their own and would bias the estimate.
     What remains is mostly court, ads and crowd = the static background.
  3. Track the corners into the current frame with pyramidal Lucas-Kanade.
  4. Fit a similarity transform (rotation + uniform scale + translation)
     with RANSAC to reject outliers.

The resulting 2x3 matrix is applied to every track's Kalman state before
association (see ``KalmanFilterXYWH.apply_affine``).
"""

from __future__ import annotations

import cv2
import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class SparseOpticalFlowGMC:
    def __init__(self, downscale: int = 2) -> None:
        self.downscale = max(1, int(downscale))
        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray, self._prev_pts = None, None

    def _prepare(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self.downscale > 1:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        return gray

    def _features(self, gray: np.ndarray, boxes: np.ndarray | None) -> np.ndarray | None:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        # Ignore a thin border (logos, letterboxing) ...
        h, w = gray.shape
        mask[: int(0.02 * h), :] = 0
        mask[int(0.98 * h) :, :] = 0
        # ... and every detected player.
        if boxes is not None and len(boxes):
            for x1, y1, x2, y2 in (np.asarray(boxes) / self.downscale).astype(int):
                mask[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)] = 0
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3, mask=mask
        )

    def apply(self, frame: np.ndarray, boxes: np.ndarray | None = None) -> np.ndarray:
        """Return the 2x3 affine mapping previous-frame pixels to this frame."""
        gray = self._prepare(frame)
        affine = IDENTITY.copy()

        if self._prev_gray is not None and self._prev_pts is not None and len(self._prev_pts) > 10:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self._prev_gray, gray, self._prev_pts, None
            )
            ok = status.reshape(-1) == 1
            prev_ok, next_ok = self._prev_pts[ok], next_pts[ok]
            if len(prev_ok) > 10:
                est, _ = cv2.estimateAffinePartial2D(
                    prev_ok, next_ok, method=cv2.RANSAC, ransacReprojThreshold=3.0
                )
                if est is not None:
                    affine = est
                    # The matrix was estimated on a downscaled image: only the
                    # translation needs to be scaled back (R is scale-free).
                    affine[:, 2] *= self.downscale

        # Features for the NEXT call are taken on the current frame.
        self._prev_gray = gray
        self._prev_pts = self._features(gray, boxes)
        return affine
