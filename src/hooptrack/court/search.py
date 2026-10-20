"""Find the court in a single frame with no annotation: search over broadcast cameras (court chapter, C05).

C03 showed that line refinement only works from a close start. This module
produces the start by *searching the camera*, not the homography:

1. **Grid.** A few thousand plausible broadcast cameras (``camera.Camera``):
   position behind the near sideline, the floor point it looks at, the zoom.
   Each one is a homography; the court lines are projected through all of them
   at once (vectorised) and scored against the painted-line evidence.
2. **Local search.** The best few grid cells (kept apart from each other, so
   one wrong peak cannot fill the list) are polished in camera space (Powell
   over the 6 camera numbers), then in free homography space (``refine.refine``).
3. **Ranking.** The final score rewards projected line samples that land on a
   painted line running in the same direction, and penalises those that do not:

       score = matched − ``miss_weight`` · unmatched      (counts of samples)

   A ratio (C03's ``agreement``) favours tiny views where a handful of samples
   happen to match; a count with a penalty favours explaining *many* lines.

The output is a ranked list of candidates — not a verdict. Whether a candidate
is right is decided later by temporal consistency (``autoinit``), the check
that a single frame cannot fake.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from .camera import look_at_homographies
from .leagues import CourtSpec, lines
from .refine import LineEvidence, refine, sample_lines, score


@dataclass
class Candidate:
    H: np.ndarray
    score: float              # matched - miss_weight * unmatched (samples)
    agreement: float          # share of visible samples matched (C03)
    visible: int              # visible samples
    camera: np.ndarray        # (cx, cy, cz, tx, ty, log f) after the camera-space search


@dataclass(frozen=True)
class SearchSpace:
    """Ranges of plausible broadcast cameras, relative to the court (metres / pixels)."""
    dx: tuple[float, ...] = (-4.0, 0.0, 4.0)              # camera x from the halfway line
    back: tuple[float, ...] = (8.0, 18.0, 30.0)           # distance behind the near sideline
    height: tuple[float, ...] = (6.0, 12.0, 20.0)
    target_step: float = 1.5                              # grid of floor points looked at (x)
    target_y: tuple[float, ...] = (3.0, 7.5, 12.0)        # as a share of 15 m, rescaled to the court width
    focal: tuple[float, ...] = (900, 1300, 1900, 2700, 3900, 5600)


DEFAULT_SPACE = SearchSpace()


def grid_cameras(spec: CourtSpec, space: SearchSpace = DEFAULT_SPACE) -> np.ndarray:
    """(N, 6) cameras: cx, cy, cz, tx, ty, log(focal)."""
    tx = np.arange(-2.0, spec.length + 2.0 + 1e-6, space.target_step)
    ty = np.array(space.target_y) * spec.width / 15.0
    g = np.array(np.meshgrid(space.dx, space.back, space.height, tx, ty, np.log(space.focal),
                             indexing="ij")).reshape(6, -1).T
    g[:, 0] += spec.length / 2
    g[:, 1] += spec.width
    return g


def homographies_of(cams: np.ndarray, image_size) -> np.ndarray:
    return look_at_homographies(cams[:, :3], cams[:, 3:5], np.exp(cams[:, 5]), image_size)


def batch_scores(Hs: np.ndarray, pts: np.ndarray, dirs: np.ndarray, ev: LineEvidence, tol: float = 6.0,
                 max_sin: float = 0.5, miss_weight: float = 0.5, soft: bool = False) -> np.ndarray:
    """Score of N homographies at once, (N,).

    Hard (``soft=False``): matched − w·unmatched, a sample being matched when it
    lies within ``tol`` px of a painted line running the same way.
    Soft: each visible sample scores (1+w)·(1 − d/tol) − w, from +1 on a line
    to −w at ``tol`` px or more — the same idea, but it changes smoothly as the
    camera moves, so a coarse grid cell near the truth already scores well and a
    local optimiser can climb towards it. ``tol`` is then a capture radius:
    large on the grid, smaller as the search narrows.
    """
    h, w = ev.dist.shape
    P = np.c_[pts, np.ones(len(pts))]
    Q = np.c_[pts + 0.05 * dirs, np.ones(len(pts))]
    a = np.einsum("nij,pj->npi", Hs, P)
    b = np.einsum("nij,pj->npi", Hs, Q)
    front = (a[..., 2] > 1e-6) & (b[..., 2] > 1e-6)
    a = a[..., :2] / np.where(front, a[..., 2], 1)[..., None]
    b = b[..., :2] / np.where(front, b[..., 2], 1)[..., None]
    x, y = a[..., 0], a[..., 1]
    vis = front & (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    xi = np.clip(x, 0, w - 1).astype(np.int32)
    yi = np.clip(y, 0, h - 1).astype(np.int32)
    vis &= ev.valid[yi, xi] > 0
    t = b - a
    ang = np.arctan2(t[..., 1], t[..., 0])
    same_way = np.abs(np.sin(ang - ev.angle[yi, xi])) < max_sin
    d = np.where(same_way, ev.dist[yi, xi], np.inf)          # wrong direction = no line
    if soft:
        g = (1 + miss_weight) * (1 - np.minimum(d, tol) / tol) - miss_weight
        return np.where(vis, g, 0.0).sum(1)
    good = vis & (d <= tol)
    return good.sum(1) - miss_weight * (vis & ~good).sum(1)


def _distinct(cams: np.ndarray, Hs: np.ndarray, order: np.ndarray, k: int, spec: CourtSpec,
              image_size, min_px: float = 60.0) -> list[int]:
    """Top-k indices whose projected court corners differ by at least ``min_px`` (non-maximum suppression)."""
    corners = np.array([[0, 0], [spec.length, 0], [spec.length, spec.width], [0, spec.width],
                        [spec.length / 2, 0], [spec.length / 2, spec.width]], float)
    keep, kept_pts = [], []
    for i in order:
        p = cv2.perspectiveTransform(corners[None], Hs[i])[0]
        p = np.clip(p, -2 * image_size[0], 3 * image_size[0])
        if all(np.median(np.linalg.norm(p - q, axis=1)) > min_px for q in kept_pts):
            keep.append(int(i))
            kept_pts.append(p)
            if len(keep) == k:
                break
    return keep


def search(ev: LineEvidence, spec: CourtSpec, image_size=(1280, 720), space: SearchSpace = DEFAULT_SPACE,
           top_k: int = 8, miss_weight: float = 0.5, grid_tol: float = 30.0, local_tols=(30.0, 15.0, 8.0),
           refine_taus=(12.0, 5.0)) -> list[Candidate]:
    """Ranked court candidates for one frame's line evidence."""
    pts, dirs = sample_lines(lines(spec), step=0.5)               # coarse samples for the search
    S, T = sample_lines(lines(spec))                              # fine samples for refinement and ranking
    cams = grid_cameras(spec, space)
    s = np.concatenate([batch_scores(homographies_of(cams[i:i + 2000], image_size), pts, dirs, ev,
                                     tol=grid_tol, max_sin=0.7, miss_weight=miss_weight, soft=True)
                        for i in range(0, len(cams), 2000)])
    order = np.argsort(-s)[:400]
    Hs = homographies_of(cams[order], image_size)
    top = _distinct(cams[order], Hs, np.arange(len(order)), top_k, spec, image_size)

    out = []
    for i in top:
        cam = cams[order[i]].copy()
        for tol in local_tols:                                    # shrink the capture radius as we close in
            def cost(c, tol=tol):
                return -float(batch_scores(homographies_of(c[None], image_size), pts, dirs, ev, tol=tol,
                                           max_sin=0.6, miss_weight=miss_weight, soft=True)[0])

            cam = minimize(cost, cam, method="Powell",
                           options={"xtol": 1e-2, "ftol": 1e-3, "maxfev": 400,
                                    "direc": np.diag([1.0, 2.0, 1.0, 0.8, 0.8, 0.08])}).x
        H = refine(homographies_of(cam[None], image_size)[0], S, ev, T, taus=refine_taus)
        sc = float(batch_scores(H[None], S, T, ev, tol=4.0, max_sin=0.35, miss_weight=miss_weight)[0])
        a, n = score(H, S, ev, tangents=T)
        out.append(Candidate(H, sc, a, n, cam))
    return sorted(out, key=lambda c: -c.score)


# ---------------------------------------------------------------------------------------------
# Second method: hypotheses from straight lines (C05b)
#
# The camera grid above fails on real footage (see the C05 README): the true
# homography is the best-scoring one, but its basin is a needle — a grid cell
# 1 m away already scores like clutter. So instead of guessing the camera and
# checking the lines, start FROM the lines:
#
# * The court's straight lines come in two families: lines along the court
#   (sidelines, lane edges: y = const) and across it (baselines, halfway,
#   free-throw lines: x = const). In a broadcast view the first family is close
#   to horizontal, the second is steep.
# * Two image lines from each family meet in 4 points. Saying WHICH court lines
#   they are (e.g. "far sideline and upper lane edge", "baseline and free-throw
#   line") gives 4 court points -> one homography. The camera stands on the near
#   side, so the order is preserved: the upper image line is the smaller y,
#   the left one the smaller x. That leaves a few thousand hypotheses, each
#   checked against ALL the painted lines with the strict score.
# * Hypotheses no real camera could produce are dropped first
#   (``camera.camera_from_homography``).
# ---------------------------------------------------------------------------------------------

def image_lines(ev: LineEvidence, min_len: int = 120, max_lines: int = 6,
                ambiguous: tuple[float, float] = (25.0, 25.0)) -> tuple[list, list]:
    """Long straight lines of the evidence, split into (along-court, across-court) families.

    Each line is (a, b, c) with a·x + b·y + c = 0, a² + b² = 1, plus its support
    (pixels). Collinear segments are merged; the longest ``max_lines`` per family are kept.
    """
    m = (ev.mask * 255).astype(np.uint8)
    segs = cv2.HoughLinesP(m, 1, np.pi / 360, threshold=60, minLineLength=min_len, maxLineGap=12)
    if segs is None:
        return [], []
    segs = segs.reshape(-1, 4).astype(float)
    lines_, used = [], np.zeros(len(segs), bool)
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    for i in np.argsort(-length):
        if used[i]:
            continue
        x1, y1, x2, y2 = segs[i]
        n = np.array([y2 - y1, x1 - x2])
        n /= np.linalg.norm(n)
        c = -n @ (x1, y1)
        # merge segments lying on (almost) the same line
        mids = np.c_[(segs[:, 0] + segs[:, 2]) / 2, (segs[:, 1] + segs[:, 3]) / 2]
        dirs = np.c_[segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]] / length[:, None]
        same = (~used) & (np.abs(mids @ n + c) < 6) & (np.abs(dirs @ n) < 0.05)
        used |= same
        lines_.append((np.array([n[0], n[1], c]), float(length[same].sum())))
    # Which family a line belongs to is clear for near-horizontal (sidelines) and steep
    # (baselines) lines, less so in between: a lane edge seen from a corner camera runs at ~30°.
    # Lines with an angle inside ``ambiguous`` go to BOTH families (a hypothesis using the same
    # line twice has no intersection and is skipped). Tried with (15°, 40°): no gain on the FIBA
    # clip it was meant for, and fewer confirmed anchors on the NBA clip (905 vs 1,048 frames),
    # so the default is a plain 25° split (C05 README).
    along, across = [], []
    for ln, sup in lines_:
        angle = np.degrees(np.arctan2(abs(ln[0]), abs(ln[1])))      # 0 = horizontal line
        if angle < ambiguous[1]:
            along.append((ln, sup))
        if angle > ambiguous[0]:
            across.append((ln, sup))
    along = sorted(along, key=lambda t: -t[1])[:max_lines]
    across = sorted(across, key=lambda t: -t[1])[:max_lines]
    return along, across


def _court_line_values(spec: CourtSpec) -> tuple[np.ndarray, np.ndarray]:
    """y of the straight lines along the court, x of those across it."""
    c, lw = spec.width / 2, spec.lane_w / 2
    ys = {0.0, spec.width, c - lw, c + lw}
    if spec.three_x > 0:
        ys |= {spec.three_side, spec.width - spec.three_side}
    xs = {0.0, spec.length, spec.length / 2, spec.ft_x, spec.length - spec.ft_x}
    return np.array(sorted(ys)), np.array(sorted(xs))


def _meet(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    p = np.cross(l1, l2)
    return p[:2] / p[2] if abs(p[2]) > 1e-9 else np.array([np.nan, np.nan])


def plausible(H: np.ndarray, spec: CourtSpec, image_size, min_cover: float = 0.3, check_camera: bool = True) -> bool:
    """Could a broadcast camera produce H while filming this game?

    * filming the game: every court corner is in front of the camera (the
      homography keeps its sign) and the court covers at least ``min_cover`` of
      the image. This rules out the degenerate fits that squash the whole court
      onto one strong image line — they explain that line and nothing else;
    * a real camera (``check_camera``): positive focal length, above the floor,
      behind the near sideline (``camera.camera_from_homography``). Only applied
      AFTER refinement: a raw 4-line hypothesis can fit the visible part well
      and still extrapolate to an impossible camera; refinement on all the
      visible lines fixes that, and the camera test then judges the result.
    """
    w, h = image_size
    court = np.array([[0, 0, 1], [spec.length, 0, 1], [spec.length, spec.width, 1], [0, spec.width, 1]], float)
    q = court @ H.T
    if not ((q[:, 2] > 0).all() or (q[:, 2] < 0).all()):
        return False
    poly = (q[:, :2] / q[:, 2:]).astype(np.float32)
    if np.abs(poly).max() > 1e6:
        return False
    frame = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    area, _ = cv2.intersectConvexConvex(poly, frame)
    if area < min_cover * w * h:
        return False
    if not check_camera:
        return True
    from .camera import camera_from_homography
    cam = camera_from_homography(H, image_size)
    if cam is None:
        return False
    _, y, z = cam["position"]
    # (x along the court is poorly determined when the camera looks straight across: not checked)
    return 300 < cam["focal"] < 15000 and 2 < z < 60 and spec.width + 1 < y < spec.width + 120


def line_hypotheses(along: list, across: list, spec: CourtSpec, image_size) -> np.ndarray:
    """All order-preserving 2 + 2 line assignments -> homographies (N, 3, 3)."""
    ys, xs = _court_line_values(spec)
    w, h = image_size

    def order_y(ln):                     # where the line crosses the image's vertical centre line
        a, b, c = ln
        return -(a * w / 2 + c) / b if abs(b) > 1e-9 else np.inf

    def order_x(ln):                     # where the line crosses the image's horizontal centre line
        a, b, c = ln
        return -(b * h / 2 + c) / a if abs(a) > 1e-9 else np.inf

    A = sorted([ln for ln, _ in along], key=order_y)       # top to bottom  = increasing court y
    B = sorted([ln for ln, _ in across], key=order_x)      # left to right  = increasing court x
    out = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            for k in range(len(B)):
                for m in range(k + 1, len(B)):
                    img = np.array([_meet(A[i], B[k]), _meet(A[i], B[m]), _meet(A[j], B[k]), _meet(A[j], B[m])])
                    if not np.isfinite(img).all() or np.abs(img).max() > 20 * w:
                        continue
                    for yi in range(len(ys)):
                        for yj in range(yi + 1, len(ys)):
                            for xk in range(len(xs)):
                                for xm in range(xk + 1, len(xs)):
                                    court = np.array([[xs[xk], ys[yi]], [xs[xm], ys[yi]],
                                                      [xs[xk], ys[yj]], [xs[xm], ys[yj]]], np.float32)
                                    out.append(cv2.getPerspectiveTransform(court, img.astype(np.float32)))
    return np.array(out) if out else np.zeros((0, 3, 3))


def explained(H: np.ndarray, samples: np.ndarray, tangents: np.ndarray, ev: LineEvidence, tol: float = 4.0,
              max_sin: float = 0.35, miss_weight: float = 0.5, cell: int = 3) -> float:
    """Like the hard score, but counting IMAGE cells (``cell`` px) instead of court samples.

    A degenerate homography can squash the whole court onto one strong image
    line (seen on the NCAA clip: every court line piled on the baseline): each
    court sample then "matches", and a per-sample count rewards it. Counting how
    many distinct image cells are explained — matched cells minus w · unmatched
    cells — caps what squashing can earn at the one line it lands on. (When that
    line is most of what is visible, squashing can still come close: the
    coverage test in ``plausible`` is what rules it out.)
    """
    h, w = ev.dist.shape
    p = cv2.perspectiveTransform(samples[None].astype(np.float64), H)[0]
    q = cv2.perspectiveTransform((samples + 0.05 * tangents)[None].astype(np.float64), H)[0]
    ok = np.isfinite(p).all(1) & (p[:, 0] >= 0) & (p[:, 0] < w - 1) & (p[:, 1] >= 0) & (p[:, 1] < h - 1)
    p, q = p[ok], q[ok]
    xi, yi = p[:, 0].astype(int), p[:, 1].astype(int)
    vis = ev.valid[yi, xi] > 0
    t = q - p
    good = vis & (ev.dist[yi, xi] <= tol) & \
        (np.abs(np.sin(np.arctan2(t[:, 1], t[:, 0]) - ev.angle[yi, xi])) < max_sin)
    key = (yi // cell) * (w // cell + 1) + xi // cell
    good_cells = np.unique(key[good])
    bad_cells = np.setdiff1d(np.unique(key[vis & ~good]), good_cells)
    return float(len(good_cells) - miss_weight * len(bad_cells))


def search_lines(ev: LineEvidence, spec: CourtSpec, image_size=(1280, 720), top_k: int = 6,
                 miss_weight: float = 0.5, refine_taus=(12.0, 5.0), min_len: int = 80) -> list[Candidate]:
    """Ranked court candidates from straight-line hypotheses (C05b)."""
    along, across = image_lines(ev, min_len=min_len)
    Hs = line_hypotheses(along, across, spec, image_size)
    if len(Hs) == 0:
        return []
    Hs = Hs[[plausible(H, spec, image_size, check_camera=False) for H in Hs]]   # no squashed courts
    if len(Hs) == 0:
        return []
    S, T = sample_lines(lines(spec))
    pts, dirs = sample_lines(lines(spec), step=0.25)
    s = np.concatenate([batch_scores(Hs[i:i + 1000], pts, dirs, ev, tol=5.0, max_sin=0.4,
                                     miss_weight=miss_weight) for i in range(0, len(Hs), 1000)])
    keep = list(np.argsort(-s)[:300])
    ex = np.array([explained(Hs[i], pts, dirs, ev, tol=5.0, max_sin=0.4, miss_weight=miss_weight) for i in keep])
    keep = [keep[i] for i in np.argsort(-ex)]
    top = _distinct(None, Hs[keep], np.arange(len(keep)), 2 * top_k, spec, image_size)
    out = []
    for i in top:
        H = refine(Hs[keep[i]], S, ev, T, taus=refine_taus)
        if not plausible(H, spec, image_size):
            continue
        sc = explained(H, S, T, ev, miss_weight=miss_weight)
        a, n = score(H, S, ev, tangents=T)
        out.append(Candidate(H, sc, a, n, np.zeros(6)))
    return sorted(out, key=lambda c: -c.score)
