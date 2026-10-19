"""Court templates: where each named keypoint lies on the court, in metres.

Why a template
--------------
A homography maps the image to a flat plane. To compute one we need pairs
(pixel in the image) <-> (position on the court). The keypoint detector gives
the pixels; this module gives the positions. With the homography we can then

* project the court rectangle into the image  -> an exact court mask (court chapter, C04)
* project players' feet onto the court        -> a top-down map (court chapter, C05)

Coordinate system (top-down view, as seen from the main broadcast camera)
-------------------------------------------------------------------------
    x: along the court, 0 = LEFT baseline  ->  28.65 m = right baseline
    y: across the court, 0 = FAR sideline   ->  15.24 m = near sideline (camera side)

          y=0  far sideline
     01 ─────────15──────────19──────────25──────────── 34
     02            (28 ft)   │  (midcourt)   (66 ft)     35
     04 ┐                    │                         ┌ 37
      09│ lane    12 ┐       │                    ┌ 28 │33
     05 ┘          13│ 16    21  centre      26  │29   └ 38
     07            14┘       │                    └ 30   40
     08 ─────────17──────────23──────────27──────────── 41
          y=15.24 near sideline        (10/11 and 31/32: ends of the 3-pt straight lines)

The 33 names come from the public dataset ``roboflow-jvuqo/basketball-court-detection-2``
(NBA broadcasts). The dataset ships the names but NOT their court positions:
the positions below were derived from official NBA dimensions and then
**verified on all 850 labelled images** (scripts/court/audit_dataset.py):
fitting a homography per image, the labels of every floor point reproject
within ~1-2 px (576 px images).

``09`` and ``33`` are the RIMS, 10 ft above the floor: a floor homography cannot
map them (they show a constant ~9 px error in the audit), so they are excluded
from homography fitting but kept as labels (they help the detector).

League differences (important for SportsMOT, which mixes NBA, NCAA and FIBA):
lane width, 3-point distance and court size differ. Only the NBA template is
defined here; points that are the same in every league (baseline corners,
midcourt, centre) are listed in ``LEAGUE_INVARIANT``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FT_TO_M = 0.3048

# NBA court: 94 x 50 ft. Positions in FEET (official dimensions), converted below.
#   hoop centre 5.25 ft from the baseline · lane 16 ft wide, free-throw line 19 ft
#   3-pt line: straight 3 ft from each sideline up to 14 ft, arc r = 23.75 ft (top at 29 ft)
#   sideline marks (coach box) at 28 ft from each baseline · centre circle at 47 ft
NBA_KEYPOINTS_FT: dict[str, tuple[float, float]] = {
    # --- left baseline (x = 0) ---
    "01": (0, 0), "02": (0, 3), "04": (0, 17), "05": (0, 33), "07": (0, 47), "08": (0, 50),
    "09": (5.25, 25),                                   # left rim (NOT on the floor)
    # --- left half ---
    "10": (14, 3), "11": (14, 47),                      # ends of the 3-pt straight lines
    "12": (19, 17), "13": (19, 25), "14": (19, 33),     # free-throw line: far end, centre, near end
    "15": (28, 0), "17": (28, 50),                      # sideline marks at 28 ft
    "16": (29, 25),                                     # top of the 3-pt arc
    # --- midcourt ---
    "19": (47, 0), "21": (47, 25), "23": (47, 50),
    # --- right half (mirror) ---
    "25": (66, 0), "27": (66, 50),
    "26": (65, 25),
    "28": (75, 17), "29": (75, 25), "30": (75, 33),
    "31": (80, 3), "32": (80, 47),
    "33": (88.75, 25),                                  # right rim (NOT on the floor)
    # --- right baseline (x = 94 ft) ---
    "34": (94, 0), "35": (94, 3), "37": (94, 17), "38": (94, 33), "40": (94, 47), "41": (94, 50),
}
NBA_KEYPOINTS: dict[str, tuple[float, float]] = {
    k: (x * FT_TO_M, y * FT_TO_M) for k, (x, y) in NBA_KEYPOINTS_FT.items()
}
NON_FLOOR_KEYPOINTS = frozenset({"09", "33"})
# Same position in NBA and NCAA (94 x 50 ft); FIBA (28 x 15 m) differs by < 0.7 m.
LEAGUE_INVARIANT = frozenset({"01", "08", "19", "21", "23", "34", "41"})


@dataclass
class CourtTemplate:
    """Named keypoints of one court type, in metres."""

    name: str
    length: float                                    # metres, along x
    width: float                                     # metres, along y
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    non_floor: frozenset[str] = frozenset()

    def floor_points(self, names) -> tuple[list[str], np.ndarray]:
        """The subset of ``names`` usable for a floor homography, and their (N, 2) positions."""
        keep = [n for n in names if n in self.points and n not in self.non_floor]
        return keep, np.array([self.points[n] for n in keep], dtype=np.float64).reshape(-1, 2)

    def outline(self) -> np.ndarray:
        """Court rectangle (4, 2) in metres: the playing area, lines included."""
        return np.array([[0, 0], [self.length, 0], [self.length, self.width], [0, self.width]], float)


def nba_template() -> CourtTemplate:
    return CourtTemplate("NBA", 94 * FT_TO_M, 50 * FT_TO_M, dict(NBA_KEYPOINTS), NON_FLOOR_KEYPOINTS)
