"""Static look-alikes of the ball, found on the court plan (ball chapter B05b, through the adapter).

Why
---
B05's tracker followed objects that look like a ball and never move: an orange circle painted on the court, a
scorer's table, a car in an advert, a face in the stands. A still object passes any speed limit, so the tracker is
happy to stay on it for seconds. In the image such an object moves whenever the camera pans; on the court plan it
does not: projected with the frame's homography (court chapter, C07/C09), it lands on the same spot frame after
frame. The real ball never stays on one spot of the plan for long (and when it flies, its projection is not even on
the floor, it moves fast).

How
---
Each candidate's centre is projected to the plan (metres) with its frame's homography. The plan is cut into
``cell`` x ``cell`` metre cells; a cell (with its 8 neighbours, so an object on a border is not split) is
**background** if candidates were seen there in at least ``min_frames`` different frames of the clip. Candidates in
background cells are removed. Frames without a homography keep all their candidates (no evidence either way).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..court.clipcam import court_homographies


def load_court_h(dirs: list[str | Path], clip: str) -> dict[int, np.ndarray]:
    """{frame: image <- court homography} from the first ``<dir>/<clip>/homographies.npz`` found, else {}."""
    for d in dirs:
        p = Path(d) / clip / "homographies.npz"
        if p.is_file():
            return court_homographies(np.load(p), use_ptz=True, smooth=True)
    return {}


def remove_static(dets: dict[int, np.ndarray], H: dict[int, np.ndarray], min_frames: int | None,
                  cell: float = 0.5) -> tuple[dict[int, np.ndarray], int]:
    """``{frame: (N, 5) [cx, cy, d, conf, _]}`` -> (same without the background candidates, number removed)."""
    if not min_frames or not H:
        return dets, 0
    keys: dict[tuple[int, int], set[int]] = {}
    where: dict[tuple[int, int], tuple[int, int]] = {}              # (frame, row) -> cell
    for f, d in dets.items():
        if f not in H or not len(d):
            continue
        pts = cv2.perspectiveTransform(np.asarray(d[:, :2], np.float64)[None], np.linalg.inv(H[f]))[0]
        for k, (x, y) in enumerate(pts):
            c = (int(np.floor(x / cell)), int(np.floor(y / cell)))
            keys.setdefault(c, set()).add(f)
            where[(f, k)] = c

    def frames_near(c):
        s = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                s |= keys.get((c[0] + dx, c[1] + dy), set())
        return len(s)

    background = {c for c in keys if frames_near(c) >= min_frames}
    out, removed = {}, 0
    for f, d in dets.items():
        keep = [k for k in range(len(d)) if where.get((f, k)) not in background]
        removed += len(d) - len(keep)
        out[f] = d[keep]
    return out, removed
