"""Court templates for several leagues, generated from a handful of official dimensions (C03).

Every basketball court is the same drawing with different numbers: a rectangle,
a halfway line, a centre circle and, at each end, a lane, a free-throw line and
a three-point line. ``CourtSpec`` holds those numbers; from it we generate

* the 33 named keypoints (same names as the NBA dataset, so the detector's
  outputs can be paired with ANY league's positions), and
* the painted lines, as polylines in metres (for drawing and for line-based
  refinement, see ``refine.py``).

The detector recognises *what* a point is ("corner of the lane on the
baseline"); the spec says *where* it is in this league. That separation is what
lets a model trained on NBA footage be used on NCAA and FIBA courts.

Dimensions (metres; NBA and NCAA are defined in feet):

=========  ========  =======  ======  ======  ======  ===========  ========
league     court     hoop*    lane    FT**    3-pt r  3-pt corner  mark***
=========  ========  =======  ======  ======  ======  ===========  ========
NBA        28.65 x   1.60     4.88    5.79    7.24    0.91 from    8.53
           15.24                                      sideline
FIBA       28 x 15   1.575    4.90    5.80    6.75    0.90         8.325
NCAA (men, 28.65 x   1.60     3.66    5.79    6.32    arc reaches  8.53
pre-2019)  15.24                                      the corner
=========  ========  =======  ======  ======  ======  ===========  ========

\\* hoop centre from the baseline · \\** free-throw line from the baseline ·
\\*** sideline mark (coach box / throw-in line) from the baseline.

The NCAA lane and 3-pt line changed over time (3-pt moved to 6.75 m in 2019);
SportsMOT's NCAA clip is from 2014. More variants can be added as specs; the
league of a video is chosen from the data (best line agreement), not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .template import FT_TO_M, NON_FLOOR_KEYPOINTS, CourtTemplate

F = FT_TO_M


@dataclass(frozen=True)
class CourtSpec:
    name: str
    length: float
    width: float
    hoop_x: float          # hoop centre, from the baseline
    lane_w: float          # lane width
    ft_x: float            # free-throw line, from the baseline
    three_r: float         # 3-pt arc radius (from the hoop centre)
    three_side: float      # 3-pt corner line, distance from the sideline
    mark_x: float          # sideline mark, from the baseline
    circle_r: float        # centre circle and free-throw circle radius

    @property
    def three_x(self) -> float:
        """x where the straight corner line meets the arc."""
        dy = self.width / 2 - self.three_side
        return self.hoop_x + (np.sqrt(self.three_r ** 2 - dy ** 2) if self.three_r > dy else 0.0)


NBA = CourtSpec("NBA", 94 * F, 50 * F, 5.25 * F, 16 * F, 19 * F, 23.75 * F, 3 * F, 28 * F, 6 * F)
FIBA = CourtSpec("FIBA", 28.0, 15.0, 1.575, 4.9, 5.8, 6.75, 0.9, 8.325, 1.8)
NCAA_PRE2019 = CourtSpec("NCAA", 94 * F, 50 * F, 5.25 * F, 12 * F, 19 * F, 20.75 * F, 4.25 * F, 28 * F, 6 * F)
SPECS = {s.name: s for s in (NBA, FIBA, NCAA_PRE2019)}


def keypoints(spec: CourtSpec) -> dict[str, tuple[float, float]]:
    """The 33 named keypoints of ``spec`` (names of the NBA keypoint dataset)."""
    L, W, c = spec.length, spec.width, spec.width / 2
    s, lw, x3 = spec.three_side, spec.lane_w / 2, spec.three_x
    left = {
        "01": (0, 0), "02": (0, s), "04": (0, c - lw), "05": (0, c + lw), "07": (0, W - s), "08": (0, W),
        "09": (spec.hoop_x, c),
        "10": (x3, s), "11": (x3, W - s),
        "12": (spec.ft_x, c - lw), "13": (spec.ft_x, c), "14": (spec.ft_x, c + lw),
        "15": (spec.mark_x, 0), "17": (spec.mark_x, W),
        "16": (spec.hoop_x + spec.three_r, c),
    }
    mirror = {"01": "34", "02": "35", "04": "37", "05": "38", "07": "40", "08": "41", "09": "33",
              "10": "31", "11": "32", "12": "28", "13": "29", "14": "30", "15": "25", "17": "27", "16": "26"}
    pts = dict(left)
    pts.update({mirror[k]: (L - x, y) for k, (x, y) in left.items()})
    pts.update({"19": (L / 2, 0), "21": (L / 2, c), "23": (L / 2, W)})
    return {k: (float(x), float(y)) for k, (x, y) in pts.items()}


def lines(spec: CourtSpec, n_arc: int = 48) -> list[np.ndarray]:
    """Painted lines as polylines in metres.

    The free-throw circle is drawn only on the side away from the baseline: the
    half inside the lane is dashed on most courts and would mislead line fitting.
    """
    L, W, c = spec.length, spec.width, spec.width / 2

    def arc(cx, cy, r, a0, a1):
        t = np.linspace(np.radians(a0), np.radians(a1), n_arc)
        return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]

    out = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float),
           np.array([[L / 2, 0], [L / 2, W]], float),
           arc(L / 2, c, spec.circle_r, 0, 360)]
    lw, s, x3 = spec.lane_w / 2, spec.three_side, spec.three_x
    a = np.degrees(np.arctan2(c - s, x3 - spec.hoop_x)) if x3 > spec.hoop_x else 90.0
    for side in (0, 1):
        def m(p, side=side):
            return np.c_[L - p[:, 0], p[:, 1]] if side else p
        out += [
            m(np.array([[0, c - lw], [spec.ft_x, c - lw], [spec.ft_x, c + lw], [0, c + lw]], float)),
            m(arc(spec.ft_x, c, spec.circle_r, -90, 90)),
            m(arc(spec.hoop_x, c, spec.three_r, -a, a)),
        ]
        if x3 > 0:
            out += [m(np.array([[0, s], [x3, s]], float)), m(np.array([[0, W - s], [x3, W - s]], float))]
    return out


def template(spec: CourtSpec) -> CourtTemplate:
    return CourtTemplate(spec.name, spec.length, spec.width, keypoints(spec), NON_FLOOR_KEYPOINTS)
