"""Post-detection filters: remove people who are not players, without training.

Why a filter and not (only) a fine-tuned detector?
--------------------------------------------------
Experiment 001 showed that a COCO ``person`` detector finds 97 % of the players
but half of its confident boxes are spectators, bench players, photographers
and referees. SportsMOT only annotates players on the court. Those extra boxes
create ~110 spurious identities and fill the identity memory with strangers.

A fine-tuned detector learns this annotation policy, but needs a GPU and
labelled data. Most of the false positives, however, share one simple
geometric property: **their feet are not on the playing surface**. This module
exploits that with classic image processing, on CPU, in a few milliseconds.

CourtMask — how it works (per frame)
------------------------------------
1. Estimate the floor colour as the median HSV of a central band of the image,
   ignoring detected people (the band is almost always hardwood).
2. Threshold pixels close to that colour  ->  binary "floor" mask.
3. Clean it (opening removes speckles, closing fills court lines and logos)
   and keep the largest connected component.
4. Take its convex hull: painted areas (the key, centre logo) and players
   standing on the floor punch holes in the colour mask; the hull fills them.
5. A detection is kept if its foot point (bottom-centre of the box) is inside
   the hull, with a tolerance proportional to the box height (players step on
   or just over the lines).

Safety nets (it must never make things worse)
---------------------------------------------
* If the floor covers too little of the frame (close-up, replay, crowd shot),
  the mask is not trusted: the last good hull is reused for a few frames, and
  after that the filter lets everything through.
* The filter only removes boxes; it never adds or moves them.

Known limitation: referees walk on the court, so this filter cannot remove
them. They need an appearance cue (experiment 006).

Experiment 008: the playing court sits INSIDE the visible wood (a strip of
apron runs along the sidelines). On the train split, players' feet are never
within a few percent of their height of the floor edge, while referees, bench
and staff often are — hence a negative (inward) ``margin``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CourtMaskConfig:
    downscale: int = 2             # work on a half-resolution image (speed)
    band: tuple[float, float, float, float] = (0.45, 0.80, 0.15, 0.85)  # y0, y1, x0, x1 (fractions)
    hue_tol: int = 12              # max hue distance to the floor colour (OpenCV hue: 0-179)
    sat_tol: int = 70              # max saturation distance
    min_value: int = 60            # ignore very dark pixels (shadows, black ads)
    min_area_frac: float = 0.15    # floor must cover >= 15 % of the frame to be trusted
    # Foot-point tolerance, as a fraction of box height. Positive = feet may be
    # OUTSIDE the floor by that much (003). Negative = feet must be that far
    # INSIDE it: the court is inset from the visible wood by the apron, where
    # referees, bench and staff stand (008, calibrated on the train split with
    # scripts/calibrate_court_margin.py).
    margin: float = 0.15
    # --- 008: per-video, self-calibrated margin -----------------------------------
    # A single inward margin does not transfer between arenas (train split: players'
    # feet p1 ranges from -1.2 to +0.2 box heights per video). Instead, each video
    # measures where ITS players stand: boxes labelled team A/B by the colour
    # classifier (006) report their foot distance; after ``adaptive_after`` frames
    # the margin becomes  min(margin, -(p_q - slack))  and is frozen. It can only
    # get stricter than ``margin`` when this video's players clearly stay inside.
    adaptive_margin: bool = False
    adaptive_after: int = 150        # frames of evidence before the margin is set
    adaptive_min_samples: int = 200  # labelled player boxes needed
    adaptive_quantile: float = 1.0   # percentile of players' foot distance (1 = keep 99 %)
    adaptive_slack: float = 0.05     # extra tolerance below that percentile
    reuse_frames: int = 25         # reuse the last good hull for up to N failed frames

    @classmethod
    def from_dict(cls, d: dict | None) -> CourtMaskConfig:
        d = d or {}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "band" in known:
            known["band"] = tuple(known["band"])
        return cls(**known)


class CourtMask:
    def __init__(self, cfg: CourtMaskConfig | None = None) -> None:
        self.cfg = cfg or CourtMaskConfig()
        self._last_hull: np.ndarray | None = None
        self._failed = 0
        # Counters for reports / debugging.
        self.last_hull: np.ndarray | None = None   # for visualisation
        self.frames_trusted = 0
        self.frames_passthrough = 0
        self.removed = 0
        self.seen = 0
        # 008: self-calibration state
        self.frame = 0
        self.margin = self.cfg.margin      # effective margin (changes once if adaptive)
        self._player_d: list[float] = []
        self.adapted = False

    def reset(self) -> None:
        self.__init__(self.cfg)

    # ------------------------------------------------------------------ mask
    def estimate_hull(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray | None:
        """Convex hull of the playing surface in FULL-resolution pixels, or None."""
        c = self.cfg
        s = max(1, int(c.downscale))
        small = cv2.resize(frame, (frame.shape[1] // s, frame.shape[0] // s)) if s > 1 else frame
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # 1. Reference floor colour from the central band, people excluded.
        y0, y1, x0, x1 = c.band
        ref_mask = np.zeros((h, w), dtype=bool)
        ref_mask[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)] = True
        for bx1, by1, bx2, by2 in (np.asarray(boxes, dtype=float).reshape(-1, 4) / s).astype(int):
            ref_mask[max(by1, 0) : max(by2, 0), max(bx1, 0) : max(bx2, 0)] = False
        if ref_mask.sum() < 100:
            return None
        ref = np.median(hsv[ref_mask], axis=0)

        # 2. Pixels close to that colour (hue is circular: 0 and 179 are neighbours).
        dh = np.abs(hsv[..., 0].astype(int) - int(ref[0]))
        dh = np.minimum(dh, 180 - dh)
        floor = (
            (dh < c.hue_tol)
            & (np.abs(hsv[..., 1].astype(int) - int(ref[1])) < c.sat_tol)
            & (hsv[..., 2] > c.min_value)
        ).astype(np.uint8) * 255

        # 3. Clean + largest component.
        floor = cv2.morphologyEx(floor, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        floor = cv2.morphologyEx(floor, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(floor)
        if n < 2:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < c.min_area_frac * h * w:
            return None  # not enough floor visible: do not trust the mask

        # 4. Convex hull, back to full resolution.
        contours, _ = cv2.findContours(
            (labels == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        return (hull.astype(np.float32) * s).astype(np.int32)

    # ---------------------------------------------------------------- filter
    def keep_mask(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Boolean mask over ``boxes``: True = keep (feet on the court)."""
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.seen += len(boxes)
        self.frame += 1
        hull = self.estimate_hull(frame, boxes)
        if hull is not None:
            self._last_hull, self._failed = hull, 0
        else:
            self._failed += 1
            if self._last_hull is not None and self._failed <= self.cfg.reuse_frames:
                hull = self._last_hull

        self.last_hull = hull
        if hull is None:
            self.frames_passthrough += 1
            return np.ones(len(boxes), dtype=bool)
        if len(boxes) == 0:
            return np.ones(0, dtype=bool)

        self.frames_trusted += 1
        keep = np.empty(len(boxes), dtype=bool)
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            foot = (float((x1 + x2) / 2), float(y2))
            # signed distance: > 0 inside, < 0 outside (pixels)
            dist = cv2.pointPolygonTest(hull, foot, True)
            keep[i] = dist >= -self.margin * (y2 - y1)
        self.removed += int((~keep).sum())
        return keep

    # ------------------------------------------------------- 008 self-calibration
    def observe_players(self, boxes: np.ndarray) -> None:
        """Record the foot distance of boxes known to be players (team-labelled).

        Called by the pipeline after the colour classifier; a no-op unless
        ``adaptive_margin`` is on and the margin is not frozen yet.
        """
        c = self.cfg
        if not c.adaptive_margin or self.adapted or self.last_hull is None:
            return
        for x1, y1, x2, y2 in np.asarray(boxes, dtype=float).reshape(-1, 4):
            d = cv2.pointPolygonTest(self.last_hull, (float((x1 + x2) / 2), float(y2)), True)
            self._player_d.append(d / max(y2 - y1, 1.0))
        if self.frame >= c.adaptive_after and len(self._player_d) >= c.adaptive_min_samples:
            q = float(np.percentile(self._player_d, c.adaptive_quantile))
            self.margin = min(c.margin, -(q - c.adaptive_slack))
            self.adapted = True

    def __call__(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
        keep = self.keep_mask(frame, boxes)
        return np.asarray(boxes).reshape(-1, 4)[keep], np.asarray(scores).reshape(-1)[keep]

    def stats(self) -> dict:
        return {
            "court_frames_trusted": self.frames_trusted,
            "court_frames_passthrough": self.frames_passthrough,
            "court_boxes_removed_%": 100.0 * self.removed / max(self.seen, 1),
            "court_margin": self.margin,
        }


def build_filters(cfg: dict | None) -> list:
    """Filters enabled in the ``filters:`` config section (empty list = none).

    Absent section or ``court_mask: false`` -> no filtering, i.e. exactly the
    behaviour of experiment 001.
    """
    cfg = cfg or {}
    out = []
    if cfg.get("court_mask", False):
        out.append(CourtMask(CourtMaskConfig.from_dict(cfg.get("court_mask_params"))))
    return out
