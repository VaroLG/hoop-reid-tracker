"""Anchor homographies: a few hand-placed court points on a few frames (court chapter, C04).

Neither the public keypoint model (C02) nor line refinement from a bad start
(C03) gives a correct homography on SportsMOT footage. An *anchor* fixes that
at the source: on one frame, 6–7 points whose court position is known (lane
corners, free-throw line ends, where a sideline meets the halfway line…) are
read off the image. A homography is fitted to them (exact, no RANSAC: the
points are checked by eye) and then polished with the painted lines
(``refine.refine``), which corrects the few pixels of error of a hand click.

Anchors are stored as JSON so that every number in the experiment can be
rebuilt::

    {"sequence": "v_00HRwkvvjtQ_c001", "league": "NCAA",
     "anchors": [{"frame": 300, "points": {"35": [843, 212], ...},
                  "taus": [12, 5]}, ...]}

Point names are the 33 keypoint names of ``leagues.keypoints``; the league
decides where each one is on the court.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .leagues import SPECS, CourtSpec, keypoints, lines
from .refine import line_evidence, refine, sample_lines, score


@dataclass
class Anchor:
    frame: int
    points: dict[str, tuple[float, float]]     # keypoint name -> image (px)
    taus: tuple[float, ...] = (12.0, 5.0)      # refinement schedule (smaller = start trusted more)
    note: str = ""


@dataclass
class AnchorFile:
    sequence: str
    league: str
    anchors: list[Anchor] = field(default_factory=list)

    @property
    def spec(self) -> CourtSpec:
        return SPECS[self.league]

    @classmethod
    def load(cls, path: str | Path) -> AnchorFile:
        d = json.loads(Path(path).read_text())
        anchors = [Anchor(a["frame"], {k: tuple(v) for k, v in a["points"].items()},
                          tuple(a.get("taus", (12.0, 5.0))), a.get("note", "")) for a in d["anchors"]]
        return cls(d["sequence"], d["league"], anchors)


def homography_from_points(points: dict[str, tuple[float, float]], spec: CourtSpec) -> np.ndarray:
    """Court (m) -> image (px) from >= 4 named points (least squares over all of them)."""
    kp = keypoints(spec)
    unknown = set(points) - set(kp)
    if unknown:
        raise KeyError(f"unknown keypoint names: {sorted(unknown)}")
    if len(points) < 4:
        raise ValueError("a homography needs at least 4 points")
    src = np.array([kp[k] for k in points], float)
    dst = np.array([points[k] for k in points], float)
    H, _ = cv2.findHomography(src, dst, 0)
    return H


def homography_from_court_points(points: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Court (m) -> image (px) from explicit correspondences ``(x_m, y_m, u_px, v_px)``.

    For points that have no keypoint name: where a line crosses the halfway line,
    the leftmost point of a circle, a point along a sideline... (C06 evaluation labels).
    """
    if len(points) < 4:
        raise ValueError("a homography needs at least 4 points")
    p = np.asarray(points, float)
    H, _ = cv2.findHomography(p[:, :2], p[:, 2:], 0)
    return H


def fit_anchor(img: np.ndarray, anchor: Anchor, spec: CourtSpec, people: np.ndarray | None = None,
               ignore_below: int | None = None) -> tuple[np.ndarray, float, float]:
    """(H, agreement before, agreement after) for one anchor frame."""
    S, T = sample_lines(lines(spec))
    H0 = homography_from_points(anchor.points, spec)
    ev = line_evidence(img, None, people)
    if ignore_below is not None:
        ev.valid[ignore_below:, :] = 0
    a0, _ = score(H0, S, ev, tangents=T)
    H = refine(H0, S, ev, T, taus=anchor.taus)
    a1, _ = score(H, S, ev, tangents=T)
    return H, a0, a1


def line_intersection(a, b, c, d) -> np.ndarray:
    """Point where line a-b meets line c-d (all image px).

    Annotation helper: a court point hidden by a player (or outside the
    frame) can still be placed exactly by extending two visible lines.
    """
    a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
    da, dc = b - a, d - c
    den = da[0] * dc[1] - da[1] * dc[0]
    if abs(den) < 1e-9:
        raise ValueError("parallel lines")
    t = ((c - a)[0] * dc[1] - (c - a)[1] * dc[0]) / den
    return a + t * da
