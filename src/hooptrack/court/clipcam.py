"""One camera per clip: a correctness check for court homographies (court chapter, C07).

C05 confirms an automatic homography when two chains started from different
candidates agree. That proves *consistency*, not correctness: in 009, on new
FIBA arenas, neighbouring candidates were wrong in the same way and confirmed
each other. This module adds a test that does not depend on the chains at all.

The main broadcast camera does not move during a clip: it pans, tilts and
zooms on a head bolted to the stand. So every correct homography of the clip
must be produced by ONE camera centre ``C`` (3 numbers for the whole clip) and
3 + 1 numbers per frame (rotation, focal length):

    H_t  ~  K(f_t) · R_t · [e1  e2  −C]

A homography that no such camera can produce, within ``tol_m`` on the visible
court, belongs to another camera — i.e. it is wrong (or the director cut to
another camera, which the map must not use either). Measured on C04/C05
(experiment C07): correct clips are explained to 5–11 cm, the wrong ones of
009 to 0.9–2.1 m, and the camera they imply sits 2–5 m above the floor next
to the sideline, which no broadcast camera does.

The clip camera is chosen RANSAC-style among *plausible* positions only
(``plausible``): the wrong frames can be the majority (Tenerife, 009), so the
consensus alone would pick them. If even the chosen camera explains only a
minority of the frames, the clip is sent to review instead of guessing:
a consistent-but-wrong camera can still be plausible (Istanbul, 009).

Coordinates: court metres, x along the length, y across (0 = far sideline),
z up — a LEFT-handed frame (x × y points down). Rotations act on the
right-handed copy with z flipped, hence ``_FLIP``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from .camera import Camera, camera_from_homography
from .leagues import CourtSpec

_FLIP = np.array([1.0, 1.0, -1.0])


@dataclass
class ClipCameraConfig:
    tol_m: float = 0.5               # a frame is explained by the clip camera if within this, on the visible court
    min_share: float = 0.6           # the camera must explain this share of the frames, or the clip goes to review
    min_height: float = 5.0          # plausible broadcast camera: at least this high (m)
    min_behind: float = 6.0          # ... and at least this far behind the near sideline (m)
    max_height: float = 40.0
    hypotheses: int = 30             # frames whose own camera is tried as the clip camera
    sample: int = 40                 # frames used to score / refine a hypothesis


@dataclass
class PTZ:
    rotvec: np.ndarray               # rotation of the (z-flipped) world into the camera
    focal: float                     # pixels
    residual_m: float                # median distance to the input homography on the visible court


@dataclass
class ClipCamera:
    position: np.ndarray | None                      # (x, y, z) court metres; None if no plausible camera
    plausible: bool
    ptz: dict[int, PTZ] = field(default_factory=dict)
    accepted: list[int] = field(default_factory=list)
    share: float = 0.0
    verdict: str = "review"                          # "ok" | "review"


def plausible(C: np.ndarray, spec: CourtSpec, cfg: ClipCameraConfig) -> bool:
    return bool(cfg.min_height <= C[2] <= cfg.max_height and C[1] >= spec.width + cfg.min_behind
                and -spec.length <= C[0] <= 2 * spec.length)


def ptz_homography(C: np.ndarray, ptz: PTZ, image_size=(1280, 720)) -> np.ndarray:
    """Court floor (m) -> image (px) of the camera at ``C`` with this pan / tilt / zoom."""
    w, h = image_size
    R = Rotation.from_rotvec(ptz.rotvec).as_matrix()
    M = np.array([[1, 0, -C[0]], [0, 1, -C[1]], [0, 0, C[2]]], float)      # [e1 e2 −C], z flipped
    K = np.array([[ptz.focal, 0, w / 2], [0, ptz.focal, h / 2], [0, 0, 1.0]])
    H = K @ R @ M
    return H / H[2, 2]


def _visible(H: np.ndarray, spec: CourtSpec, image_size, visible_rows: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.array([[x, y] for x in np.linspace(0, spec.length, 29) for y in np.linspace(0, spec.width, 16)], float)
    p = cv2.perspectiveTransform(g[None], H)[0]
    v = (p[:, 0] >= 0) & (p[:, 0] < image_size[0]) & (p[:, 1] >= 0) & (p[:, 1] < visible_rows)
    return g[v], p[v]


def fit_ptz(H: np.ndarray, C: np.ndarray, spec: CourtSpec, image_size=(1280, 720), visible_rows: int | None = None,
            x0: np.ndarray | None = None) -> PTZ | None:
    """Pan / tilt / roll / zoom of a camera at ``C`` that best reproduces ``H`` (pixel least squares)."""
    w, h = image_size
    g, target = _visible(H, spec, image_size, visible_rows or h)
    if len(g) < 8:
        return None
    if x0 is None:                                   # look at the floor point under the image centre
        centre = cv2.perspectiveTransform(np.array([[[w / 2, h / 2]]], float), np.linalg.inv(H))[0, 0]
        cam = camera_from_homography(H, image_size)
        f0 = cam["focal"] if cam else w * 2.5
        if np.allclose(centre, C[:2]):
            return None
        R0 = Camera(tuple(C), tuple(centre), f0, image_size).rotation() @ np.diag(_FLIP)
        x0 = np.r_[Rotation.from_matrix(R0).as_rotvec(), np.log(f0)]
    P = np.c_[g, np.zeros(len(g))] - C * _FLIP

    def res(x):
        c = P @ Rotation.from_rotvec(x[:3]).as_matrix().T
        f = np.exp(x[3])
        return np.r_[f * c[:, 0] / c[:, 2] + w / 2 - target[:, 0], f * c[:, 1] / c[:, 2] + h / 2 - target[:, 1]]

    x = least_squares(res, x0, method="lm").x
    ptz = PTZ(x[:3], float(np.exp(x[3])), 0.0)
    Hp = ptz_homography(C, ptz, image_size)
    back = cv2.perspectiveTransform(cv2.perspectiveTransform(g[None], Hp), np.linalg.inv(H))[0]
    ptz.residual_m = float(np.median(np.linalg.norm(back - g, axis=1)))
    return ptz


def _residuals(Hs, C, spec, image_size, visible_rows, cap: float = 3.0) -> np.ndarray:
    out = []
    for H in Hs:
        p = fit_ptz(H, C, spec, image_size, visible_rows)
        out.append(cap if p is None or not np.isfinite(p.residual_m) else min(p.residual_m, cap))
    return np.array(out)


def fit_clip_camera(frames: np.ndarray, Hs: np.ndarray, spec: CourtSpec, cfg: ClipCameraConfig | None = None,
                    image_size=(1280, 720), visible_rows: int | None = None) -> ClipCamera:
    """The clip's camera centre, each frame's pan / tilt / zoom, and which frames it explains."""
    cfg = cfg or ClipCameraConfig()
    n = len(frames)
    if n == 0:
        return ClipCamera(None, False)
    pick = np.unique(np.linspace(0, n - 1, min(cfg.hypotheses, n)).astype(int))
    hyps = [np.array(c["position"]) for c in (camera_from_homography(Hs[i], image_size) for i in pick) if c]
    hyps = [C for C in hyps if plausible(C, spec, cfg)]
    if not hyps:
        return ClipCamera(None, False)
    sample = Hs[np.unique(np.linspace(0, n - 1, min(cfg.sample, n)).astype(int))]
    scores = [(np.mean(_residuals(sample, C, spec, image_size, visible_rows) < cfg.tol_m), i)
              for i, C in enumerate(hyps)]
    C0 = hyps[max(scores)[1]]
    inl = sample[_residuals(sample, C0, spec, image_size, visible_rows) < cfg.tol_m]
    if len(inl):                                     # refine the centre on the frames it explains
        C0 = minimize(lambda C: _residuals(inl, C, spec, image_size, visible_rows).mean(), C0,
                      method="Nelder-Mead", options={"xatol": 0.05, "fatol": 1e-3, "maxiter": 200}).x
    out = ClipCamera(C0, plausible(C0, spec, cfg))
    prev = None
    for f, H in zip(frames, Hs, strict=True):
        p = fit_ptz(H, C0, spec, image_size, visible_rows, prev)
        if p is not None and p.residual_m > cfg.tol_m and prev is not None:
            p2 = fit_ptz(H, C0, spec, image_size, visible_rows)      # warm start may have gone astray
            p = p2 if p2 is not None and p2.residual_m < p.residual_m else p
        if p is None:
            continue
        out.ptz[int(f)] = p
        if p.residual_m <= cfg.tol_m:
            out.accepted.append(int(f))
            prev = np.r_[p.rotvec, np.log(p.focal)]
    out.share = len(out.accepted) / n
    out.verdict = "ok" if out.plausible and out.share >= cfg.min_share else "review"
    return out
