"""A broadcast camera looking at the court, and the homography it produces (court chapter, C05).

A homography has 8 free numbers, but a real camera has far fewer *plausible*
values. A TV camera filming basketball sits behind one sideline, a few metres up,
and turns (pan / tilt) and zooms (focal length). Writing the homography from
those physical parameters does two things:

* it turns "find H" into a small, bounded search (where is the camera, where is
  it pointing, how much zoom) that needs **no annotation** at all;
* it rules out homographies no camera could produce (mirrored, sheared,
  looking at the court from below…) — exactly the wrong fits that fooled the
  line score in C03.

Conventions (metres): court x along the length [0, L], y across [0, W], z up.
y = 0 is the FAR sideline (top of the broadcast image), y = W the near one; the
camera stands at y > W. Image: x right, y down, principal point at the centre.

For a point on the floor (z = 0) a pinhole camera gives

    p ~ K · [r1  r2  t] · (x, y, 1)       (r1, r2 = first two columns of R, t = −R·C)

so the floor homography is H = K [r1 r2 t]. ``camera_from_homography`` inverts
that (focal length from the orthogonality of r1 and r2), which lets us read the
camera of any homography — e.g. the hand-checked ones of C04 — and check that
the search space covers them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    position: tuple[float, float, float]    # (x, y, z) in court metres
    target: tuple[float, float]             # floor point at the image centre (x, y)
    focal: float                            # pixels
    image_size: tuple[int, int] = (1280, 720)

    def rotation(self) -> np.ndarray:
        """World -> camera rotation (rows: camera x = right, y = down, z = forward)."""
        c = np.asarray(self.position, float)
        fwd = np.array([*self.target, 0.0]) - c
        fwd /= np.linalg.norm(fwd)
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, fwd)                 # x grows to the right when looking at the far side
        right /= np.linalg.norm(right)
        down = np.cross(right, fwd)
        return np.stack([right, down, fwd])

    def homography(self) -> np.ndarray:
        """Court floor (m) -> image (px)."""
        return look_at_homographies(np.array([self.position]), np.array([self.target]),
                                    np.array([self.focal]), self.image_size)[0]


def look_at_homographies(pos: np.ndarray, target: np.ndarray, focal: np.ndarray,
                         image_size=(1280, 720)) -> np.ndarray:
    """Vectorised ``Camera.homography`` for N cameras: (N, 3), (N, 2), (N,) -> (N, 3, 3)."""
    n = len(pos)
    fwd = np.c_[target, np.zeros(n)] - pos
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    right = np.cross(np.array([0.0, 0.0, 1.0]), fwd)
    right /= np.linalg.norm(right, axis=1, keepdims=True)
    down = np.cross(right, fwd)
    R = np.stack([right, down, fwd], axis=1)                     # (N, 3, 3)
    t = -np.einsum("nij,nj->ni", R, pos)
    K = np.zeros((n, 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = focal
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = image_size[0] / 2, image_size[1] / 2, 1.0
    M = np.stack([R[:, :, 0], R[:, :, 1], t], axis=2)          # [r1 r2 t]
    H = K @ M
    return H / H[:, 2:3, 2:3]


def camera_from_homography(H: np.ndarray, image_size=(1280, 720)) -> dict | None:
    """Focal length and camera position implied by a floor homography (None if not a valid camera)."""
    w, h = image_size
    A = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]]) @ H     # move principal point to 0
    A = np.diag([1 / w, 1 / w, 1.0]) @ A                                # pixels -> image widths (conditioning)
    h1, h2 = A[:, 0], A[:, 1]
    # With K = diag(f, f, 1), r1 = K^-1 h1 and r2 = K^-1 h2 must be orthogonal and of equal length.
    # In u = 1/f^2 each condition is linear:
    #   u (x1 x2 + y1 y2) + z1 z2 = 0          and     u (x1² + y1² − x2² − y2²) + (z1² − z2²) = 0
    # One of them degenerates when the camera looks straight across the court (z1 ≈ 0: the
    # sidelines are parallel in the image), so both are solved together, by least squares.
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    if a @ a < 1e-18:
        return None
    u = -(a @ b) / (a @ a)
    if u <= 0:
        return None
    f = float(1 / np.sqrt(u))
    Kinv = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Kinv @ h1, Kinv @ h2, Kinv @ A[:, 2]
    s = 1 / np.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))
    r1, r2, t = r1 * s, r2 * s, t * s
    if t[2] < 0:                                # the court must be in front of the camera
        r1, r2, t = -r1, -r2, -t
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    C = -R.T @ t
    C[2] = abs(C[2])                # a camera and its mirror image below the floor give the same floor homography
    ortho = float(abs(r1 @ r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))
    return {"focal": f * w, "position": C.tolist(), "orthogonality": ortho}
