"""Constant-velocity Kalman filter in (cx, cy, w, h) space.

Why this parametrisation?
-------------------------
SORT/ByteTrack use (cx, cy, aspect_ratio, h). In sports the aspect ratio of a
player changes a lot (running, jumping, crouching), so we follow BoT-SORT and
track width and height independently: the filter can then follow a player who
suddenly stretches to block a shot without "fighting" a constant-ratio prior.

State vector (8-D):   x = [cx, cy, w, h, vcx, vcy, vw, vh]
Measurement (4-D):    z = [cx, cy, w, h]

All noise terms are proportional to the box size: a 1-pixel error matters a lot
for a 20-px far-away player and nothing for a 300-px close-up. This makes the
filter scale-invariant without per-video tuning.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg


class KalmanFilterXYWH:
    """Minimal, dependency-free Kalman filter used by every track."""

    # Relative std-devs (fractions of box size). Values from BoT-SORT.
    std_weight_position = 1.0 / 20
    std_weight_velocity = 1.0 / 160

    def __init__(self) -> None:
        ndim, dt = 4, 1.0  # dt = 1 frame

        # F: transition matrix. Position_{t+1} = position_t + velocity_t * dt.
        #     [ I  dt*I ]
        #     [ 0    I  ]
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # H: observation matrix. We only measure the first 4 components.
        self._update_mat = np.eye(ndim, 2 * ndim)

    # ------------------------------------------------------------------ init
    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a new state from a first detection.

        Velocity is unknown, so it starts at zero with a LARGE variance:
        the filter will learn it from the next 2-3 observations.
        """
        mean_pos = measurement.astype(float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        w, h = measurement[2], measurement[3]
        std = [
            2 * self.std_weight_position * w,
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * w,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * w,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * w,
            10 * self.std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    # --------------------------------------------------------------- predict
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Propagate the state one frame forward:  x' = F x,  P' = F P F^T + Q."""
        w, h = mean[2], mean[3]
        std_pos = [self.std_weight_position * w, self.std_weight_position * h] * 2
        std_vel = [self.std_weight_velocity * w, self.std_weight_velocity * h] * 2
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))  # Q

        mean = self._motion_mat @ mean
        covariance = self._motion_mat @ covariance @ self._motion_mat.T + motion_cov
        return mean, covariance

    # --------------------------------------------------------------- project
    def project(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map the state into measurement space:  z_hat = H x,  S = H P H^T + R."""
        w, h = mean[2], mean[3]
        std = [self.std_weight_position * w, self.std_weight_position * h] * 2
        innovation_cov = np.diag(np.square(std))  # R

        mean = self._update_mat @ mean
        covariance = self._update_mat @ covariance @ self._update_mat.T + innovation_cov
        return mean, covariance

    # ---------------------------------------------------------------- update
    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct the prediction with a matched detection.

        K = P H^T S^-1 is computed with a Cholesky solve instead of an explicit
        inverse: S is symmetric positive-definite, so this is faster and
        numerically safer.
        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol, lower), (covariance @ self._update_mat.T).T, check_finite=False
        ).T

        innovation = measurement - projected_mean  # how wrong the prediction was
        new_mean = mean + innovation @ kalman_gain.T
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return new_mean, new_covariance

    # -------------------------------------------------- camera compensation
    @staticmethod
    def apply_affine(
        mean: np.ndarray, covariance: np.ndarray, affine: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Warp a state with the camera's frame-to-frame affine motion.

        ``affine`` is the 2x3 matrix [R | t] that maps pixel coordinates of the
        previous frame onto the current one (estimated by :mod:`.gmc`).

        * The centre is fully transformed:      c' = R c + t
        * The velocity only rotates/scales:     v' = R v
        * Width/height scale with the zoom:     s  = sqrt(|det R|)
        """
        R = affine[:, :2]
        t = affine[:, 2]
        scale = float(np.sqrt(abs(np.linalg.det(R))))

        # Block transform T acting on [cx, cy, w, h, vcx, vcy, vw, vh].
        T = np.zeros((8, 8))
        T[0:2, 0:2] = R
        T[2, 2] = T[3, 3] = scale
        T[4:6, 4:6] = R
        T[6, 6] = T[7, 7] = scale

        mean = T @ mean
        mean[0:2] += t
        covariance = T @ covariance @ T.T
        return mean, covariance
