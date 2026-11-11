"""One ball through a clip, offline, from the detector's candidates of every frame (B05).

Why
---
The ball detector of B03 alone finds 58 % of the balls at 76 % precision (B04): it misses balls in the hands, half
hidden or blurred in a pass, and now and then answers on a yellow shoe or a sign in the stands. One frame cannot
tell, but a clip can: there is exactly one ball, and between two frames it can only move so far. A false alarm is
usually alone (it does not move like a ball), and a missed ball sits between two detections that agree.

How
---
Every detection of every frame (confidence >= ``min_conf``) is a node. The best sequence of nodes over the clip is
found by dynamic programming in time order (a longest path in a DAG, exact, no learning):

* a node is worth ``log(conf / tau)``: positive for a confident detection, negative for a weak one, which is kept
  only when its neighbours in time support it;
* two nodes can follow each other (same segment) if the second is at most ``max_gap`` frames later and within
  ``max_speed`` ball diameters per frame (plus one diameter of slack); each frame skipped costs ``gap_cost``;
* the path may also stop and start again later (the ball left the picture, a long occlusion) at ``restart_cost``.

The chosen path gives the ball's position in its frames; inside a segment, the frames in between get a linear
interpolation (the ball moved from one detection to the next), marked ``interp``. Outside the segments: no ball.

All values below are fixed by physics and the detector's scale, not tuned on the test clips: a pass reaches
~12 m/s, i.e. ~2 ball diameters (0.24 m) per frame at 25 fps, camera motion adds to it -> 3 diameters per frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BallTrackConfig:
    min_conf: float = 0.05        # detections kept as candidates
    tau: float = 0.1              # a node is worth log(conf / tau): > 0 above tau
    max_speed: float = 3.0        # ball diameters per frame
    max_gap: int = 15             # frames (0.6 s) a segment can skip
    gap_cost: float = 0.05        # per skipped frame inside a segment
    restart_cost: float = 1.5     # end a segment and start another one later
    # B05c: a segment (a run of linked detections) whose detections are worth less than this in total is dropped:
    # the tracker answers "no ball" there instead of following weak evidence. None = keep every segment (B05).
    min_segment_score: float | None = None


def track_ball(dets: dict[int, np.ndarray],
               cfg: BallTrackConfig | None = None) -> dict[int, tuple[float, float, float, str]]:
    """``{frame: (N, 5) [cx, cy, diameter, conf, _]}`` -> ``{frame: (cx, cy, diameter, "det" | "interp")}``."""
    cfg = cfg or BallTrackConfig()
    nodes = [(f, *d[:4]) for f in sorted(dets) for d in np.asarray(dets[f]).reshape(-1, 5) if d[3] >= cfg.min_conf]
    if not nodes:
        return {}
    N = np.array(nodes, float)                     # columns: frame, cx, cy, diameter, conf
    F, X, Y, D, C = N.T
    reward = np.log(C / cfg.tau)
    best = np.empty(len(N))
    prev = np.full(len(N), -1)
    # best path value ending at each node; prefix maximum over earlier frames for restarts
    order = np.arange(len(N))                      # nodes are already sorted by frame
    frame_best_idx = -1                            # index of the best node among strictly earlier frames
    k = 0
    for j in order:
        while k < j and F[k] < F[j]:               # nodes of earlier frames become available for a restart
            if frame_best_idx < 0 or best[k] > best[frame_best_idx]:
                frame_best_idx = k
            k += 1
        cand, cand_prev = reward[j], -1            # start here
        if frame_best_idx >= 0 and best[frame_best_idx] - cfg.restart_cost + reward[j] > cand:
            cand, cand_prev = best[frame_best_idx] - cfg.restart_cost + reward[j], frame_best_idx
        lo = np.searchsorted(F, F[j] - cfg.max_gap, side="left")
        for i in range(lo, j):
            gap = F[j] - F[i]
            if gap < 1:
                continue
            reach = cfg.max_speed * gap * max(D[i], D[j]) + max(D[i], D[j])
            if np.hypot(X[j] - X[i], Y[j] - Y[i]) > reach:
                continue
            v = best[i] - cfg.gap_cost * (gap - 1) + reward[j]
            if v > cand:
                cand, cand_prev = v, i
        best[j], prev[j] = cand, cand_prev
    # backtrack from the best end; a link is "continuous" if it passed the motion gate (not a restart)
    path = []
    j = int(np.argmax(best))
    while j >= 0:
        path.append(j)
        j = int(prev[j])
    path = path[::-1]
    # B05c: cut the path into segments (a restart, or a link that fails the motion gate, ends a segment) and drop the
    # weak ones
    def linked(a: int, b: int) -> bool:
        gap = F[b] - F[a]
        reach = cfg.max_speed * gap * max(D[a], D[b]) + max(D[a], D[b])
        return 1 <= gap <= cfg.max_gap and np.hypot(X[b] - X[a], Y[b] - Y[a]) <= reach

    if cfg.min_segment_score is not None and path:
        segments, cur = [], [path[0]]
        for a, b in zip(path, path[1:], strict=False):
            if linked(a, b):
                cur.append(b)
            else:
                segments.append(cur)
                cur = [b]
        segments.append(cur)
        path = [n for seg in segments if reward[seg].sum() >= cfg.min_segment_score for n in seg]
    out: dict[int, tuple[float, float, float, str]] = {}
    for a, b in zip(path, path[1:] + [None], strict=True):
        out[int(F[a])] = (float(X[a]), float(Y[a]), float(D[a]), "det")
        if b is None:
            continue
        gap = int(F[b] - F[a])
        reach = cfg.max_speed * gap * max(D[a], D[b]) + max(D[a], D[b])
        if 1 < gap <= cfg.max_gap and np.hypot(X[b] - X[a], Y[b] - Y[a]) <= reach:
            for t in range(1, gap):                # same segment: fill the frames in between
                w = t / gap
                out[int(F[a]) + t] = (float(X[a] + w * (X[b] - X[a])), float(Y[a] + w * (Y[b] - Y[a])),
                                      float(D[a] + w * (D[b] - D[a])), "interp")
    return out
