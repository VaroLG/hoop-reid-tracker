"""Carry a court homography through a video (court chapter, C04).

An annotated (or detected) homography is valid for one frame. Between two
consecutive frames the camera moves a little, so the court moves a little:

    H_t  =  M_{t-1 -> t} · H_{t-1}

``M`` is the image-to-image homography of the FLOOR between the two frames,
measured with optical flow on corners found on the court (players masked out:
they move on their own). Errors would accumulate, so after every step the
homography is re-aligned with the painted lines (``refine.refine``) — the lines
pull it back to the truth before it can drift.

Photographers' flashes (one over-exposed frame, seen at frame 974 of the NCAA
clip) break optical flow, which assumes constant brightness. Two defences: flow
runs on contrast-normalised images, and a flash frame is never used as the
reference for the next step — the next frame is matched to the last normal one.

The line ``agreement`` of every frame is kept: it drops when the view has few
lines (midcourt) or when tracking fails, and the annotation tool uses it to
choose between chains started from different anchors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import cv2
import numpy as np

from .leagues import CourtSpec, lines
from .refine import line_evidence, refine, sample_lines, score


@dataclass
class ChainStep:
    frame: int
    H: np.ndarray            # court (m) -> image (px)
    agreement: float
    flow_inliers: int


def court_polygon(spec: CourtSpec, H: np.ndarray) -> np.ndarray:
    outline = np.array([[[0, 0], [spec.length, 0], [spec.length, spec.width], [0, spec.width]]], float)
    return cv2.perspectiveTransform(outline, H)[0]


def _normalised_gray(img: np.ndarray) -> np.ndarray:
    """Grey image stretched to zero mean / fixed contrast: a global brightness change cancels out."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = (g - g.mean()) / (g.std() + 1e-6)
    return np.clip(g * 40 + 128, 0, 255).astype(np.uint8)


def is_flash(prev: np.ndarray, cur: np.ndarray, jump: float = 25.0) -> bool:
    """A sudden jump of mean brightness between consecutive frames (camera flash)."""
    return abs(float(cur.mean()) - float(prev.mean())) > jump


def floor_motion(prev: np.ndarray, cur: np.ndarray, mask: np.ndarray, max_corners: int = 800):
    """Image homography prev -> cur from optical flow on ``mask`` (uint8). Returns (M, inliers)."""
    g0 = _normalised_gray(prev)
    g1 = _normalised_gray(cur)
    p0 = cv2.goodFeaturesToTrack(g0, max_corners, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 30:                       # too little texture on the court: use the whole frame
        p0 = cv2.goodFeaturesToTrack(g0, max_corners, 0.01, 8)
    if p0 is None or len(p0) < 4:
        return np.eye(3), 0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
    ok = st.ravel() == 1
    if ok.sum() < 4:
        return np.eye(3), 0
    M, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
    if M is None:
        return np.eye(3), 0
    return M, int(inl.sum())


def track_homography(frames: Iterable[tuple[int, np.ndarray]], H0: np.ndarray, spec: CourtSpec,
                     people: Callable[[int], np.ndarray] | None = None, ignore_below: int | None = None,
                     taus=(8.0, 4.0), evidence_kw: dict | None = None) -> list[ChainStep]:
    """Propagate ``H0`` (valid for the FIRST frame of ``frames``) through the others.

    ``frames``: (index, BGR image) in the order to follow (forwards or backwards).
    ``people``: frame index -> (N, 4) boxes to mask (optional).
    ``ignore_below``: rows from this y down are ignored (score bar), optional.
    ``evidence_kw``: extra options for ``refine.line_evidence`` (e.g. ``edges=True``).
    """
    S, T = sample_lines(lines(spec))
    it = iter(frames)
    f_prev, prev = next(it)
    H = H0.copy()
    out = []
    skipped = 0
    for f, cur in it:
        mask = np.zeros(prev.shape[:2], np.uint8)
        cv2.fillPoly(mask, [court_polygon(spec, H).astype(np.int32)], 255)
        boxes = people(f_prev) if people else np.zeros((0, 4))
        for x1, y1, x2, y2 in np.asarray(boxes, int).reshape(-1, 4):
            mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
        if ignore_below is not None:
            mask[ignore_below:, :] = 0
        M, n = floor_motion(prev, cur, mask)
        ev = line_evidence(cur, None, people(f) if people else None, **(evidence_kw or {}))
        if ignore_below is not None:
            ev.valid[ignore_below:, :] = 0
        H_new = refine(M @ H, S, ev, T, taus=taus)
        a, _ = score(H_new, S, ev, tangents=T)
        out.append(ChainStep(f, H_new.copy(), a, n))
        if is_flash(prev, cur) and skipped < 2:
            skipped += 1                 # keep the last normal frame (and its H) as the reference
            continue
        skipped = 0                      # (a lasting change after 2 frames is a new normal, not a flash)
        H = H_new
        f_prev, prev = f, cur
    return out


@dataclass
class Chain:
    """One propagation run: the anchor and every frame reached from it."""
    name: str
    anchor_frame: int
    steps: list[ChainStep]          # includes the anchor itself as the first step


def trusted(chain: Chain, lost_below: float = 0.3, patience: int = 5) -> Chain:
    """The part of ``chain`` before it got lost.

    A chain is declared lost at the first run of ``patience`` consecutive
    frames with line agreement below ``lost_below`` (a cut, a whip pan, motion
    blur): from there on its homography may be anywhere, and a lost chain can
    later lock onto a WRONG alignment that happens to score high (seen on the
    NCAA clip: 0.99 agreement, 9 m off). So everything after it is dropped,
    whatever it scores.
    """
    run = 0
    for i, st in enumerate(chain.steps):
        run = run + 1 if st.agreement < lost_below else 0
        if run == patience:
            return Chain(chain.name, chain.anchor_frame, chain.steps[:i - patience + 1])
    return chain


def merge_chains(chains: list[Chain], nearest_ranges: list[tuple[int, int]] = ()) -> dict[int, ChainStep]:
    """One homography per frame, chosen among the chains that reached it.

    Pass chains already cut with ``trusted``. Default: the chain with the
    highest line agreement. Inside ``nearest_ranges``
    (views with few painted lines, e.g. midcourt during a fast pan, where the
    agreement is noisy) the chain closest to its anchor wins instead: fewer
    steps, less accumulated drift.
    """
    cand: dict[int, list[tuple[Chain, ChainStep]]] = {}
    for ch in chains:
        for st in ch.steps:
            cand.setdefault(st.frame, []).append((ch, st))
    out = {}
    for f, cs in cand.items():
        if any(a <= f <= b for a, b in nearest_ranges):
            _, best = min(cs, key=lambda c: abs(f - c[0].anchor_frame))
        else:
            _, best = max(cs, key=lambda c: c[1].agreement)
        out[f] = best
    return dict(sorted(out.items()))


def disagreement(Ha: np.ndarray, Hb: np.ndarray, spec: CourtSpec, image_size=(1280, 720)) -> tuple[float, float]:
    """Median distance between two homographies over the visible court: (pixels, metres).

    Used to check that two chains started from DIFFERENT anchors meet at the
    same answer — an independent test of the annotation.
    """
    grid = np.array([[x, y] for x in np.linspace(0, spec.length, 15) for y in np.linspace(0, spec.width, 8)], float)
    pa = cv2.perspectiveTransform(grid[None], Ha)[0]
    pb = cv2.perspectiveTransform(grid[None], Hb)[0]
    w, h = image_size
    vis = (pa[:, 0] >= 0) & (pa[:, 0] < w) & (pa[:, 1] >= 0) & (pa[:, 1] < h)
    if not vis.any():
        return float("nan"), float("nan")
    px = float(np.median(np.linalg.norm(pa[vis] - pb[vis], axis=1)))
    back = cv2.perspectiveTransform(pa[vis][None], np.linalg.inv(Hb))[0]
    return px, float(np.median(np.linalg.norm(back - grid[vis], axis=1)))
