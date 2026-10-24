"""Court mask from the court homography, as a tracker filter (experiment 009).

Experiment 003's ``CourtMask`` guesses the playing surface from the floor's
colour, frame by frame, as an image polygon. Chapter 2 gives something better:
a homography per frame, i.e. where the REAL court (28 x 15 m, or 28.65 x 15.24 m)
is in the image. This filter keeps a person when their feet — the bottom
centre of the box — projected onto the court plan fall inside the court,
plus a margin in METRES (players step on the lines, inbound from outside).

Homographies are computed beforehand, per video, by
``scripts/court/auto_homographies.py`` (no annotation; chapter 2, C05) and read
from ``<dir>/<sequence>/homographies.npz``. Frames without a homography (no
court lines visible: close-ups, some fast pans) fall back to the colour mask,
so this filter is never worse-informed than 003.

Switch: ``filters.court_source: homography`` (default ``colour`` = experiment 003).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..court.leagues import SPECS
from ..detection.filters import CourtMask, CourtMaskConfig


@dataclass
class HomographyCourtConfig:
    dir: str = "outputs/court_auto_edges"     # <dir>/<sequence>/homographies.npz
    margin_m: float = 0.25                    # C04/C05: keeps 99.9 % of players
    fallback_colour: bool = True              # frames without a homography -> colour mask (003)
    guard_colour: bool = False                # 009b: distrust a homography that drops a box deep inside the colour hull
    guard_depth: float = 0.25                 # "deep": feet inside the colour hull by >= this share of the box height
    guard_min_score: float = 0.5

    @classmethod
    def from_dict(cls, d: dict | None) -> HomographyCourtConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class HomographyCourtFilter:
    def __init__(self, cfg: HomographyCourtConfig | None = None, colour: CourtMask | None = None) -> None:
        self.cfg = cfg or HomographyCourtConfig()
        self.colour = colour
        self.reset()

    def reset(self) -> None:
        self.H: dict[int, np.ndarray] = {}
        self.spec = None
        self.frame_idx = 0
        self.last_hull: np.ndarray | None = None
        self.frames_h = self.frames_fallback = self.frames_none = self.frames_guarded = 0
        self.scores: np.ndarray | None = None
        self.seen = self.removed = 0
        if self.colour is not None:
            self.colour.reset()

    def bind(self, seq) -> None:
        """Load this video's homographies (called by the pipeline before the first frame)."""
        self.reset()
        path = Path(self.cfg.dir) / seq.name / "homographies.npz"
        if path.is_file():
            z = np.load(path)
            self.H = {int(f): H for f, H in zip(z["frames"], z["H"], strict=True)}
            self.spec = SPECS[str(z["league"])]

    # ------------------------------------------------------------------ filter
    def keep_mask(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        self.frame_idx += 1                       # frames arrive in order, 1..N
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        H = self.H.get(self.frame_idx)
        if H is not None and self.cfg.guard_colour and self.colour is not None and self._contradicted(frame, boxes, H):
            self.frames_guarded += 1
            H = None                              # two court estimates disagree on the obvious: use the colour one
        if H is None:
            if self.colour is not None:
                self.frames_fallback += 1
                keep = self.colour.keep_mask(frame, boxes)
                self.last_hull = self.colour.last_hull
            else:
                self.frames_none += 1
                keep = np.ones(len(boxes), bool)
                self.last_hull = None
        else:
            self.frames_h += 1
            L, W, m = self.spec.length, self.spec.width, self.cfg.margin_m
            keep = self._on_court(boxes, H) if len(boxes) else np.ones(0, bool)
            court = np.array([[[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m]]], float)
            self.last_hull = np.clip(cv2.perspectiveTransform(court, H)[0], -1e4, 1e4).astype(np.int32)
        self.seen += len(boxes)
        self.removed += int((~keep).sum())
        return keep

    def _on_court(self, boxes: np.ndarray, H: np.ndarray) -> np.ndarray:
        L, W, m = self.spec.length, self.spec.width, self.cfg.margin_m
        feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
        c = cv2.perspectiveTransform(feet[None], np.linalg.inv(H))[0]
        return (c[:, 0] >= -m) & (c[:, 0] <= L + m) & (c[:, 1] >= -m) & (c[:, 1] <= W + m)

    def _contradicted(self, frame: np.ndarray, boxes: np.ndarray, H: np.ndarray) -> bool:
        """009b guard: does H throw out a confident person standing well inside the colour floor?

        A wrong homography (C05 can be confidently wrong on some FIBA corner views)
        typically puts the court boundary through the middle of the floor. The
        colour mask is crude at the BORDER of the court, but a person whose feet
        are deep inside the floor colour is on the court — if H says otherwise,
        H is the one to distrust for this frame.
        """
        if not len(boxes) or self.scores is None:
            return False
        conf = self.scores >= self.cfg.guard_min_score
        if not conf.any():
            return False
        hull = self.colour.estimate_hull(frame, boxes)
        if hull is None:
            return False
        b = boxes[conf]
        depth = np.array([cv2.pointPolygonTest(hull, (float((x1 + x2) / 2), float(y2)), True) for x1, _, x2, y2 in b])
        deep = depth >= self.cfg.guard_depth * (b[:, 3] - b[:, 1])
        return bool((deep & ~self._on_court(b, H)).any())

    def __call__(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
        self.scores = np.asarray(scores).reshape(-1)
        keep = self.keep_mask(frame, boxes)
        return np.asarray(boxes).reshape(-1, 4)[keep], np.asarray(scores).reshape(-1)[keep]

    def stats(self) -> dict:
        return {
            "court_frames_homography": self.frames_h,
            "court_frames_colour_fallback": self.frames_fallback,
            "court_frames_unfiltered": self.frames_none,
            "court_frames_guarded": self.frames_guarded,
            "court_boxes_removed_%": 100.0 * self.removed / max(self.seen, 1),
        }


@dataclass
class SegmentationCourtConfig:
    weights: str = "outputs/court_runs/c06_yolo26n_seg/weights/best.pt"
    imgsz: int = 512
    conf: float = 0.25
    margin: float = 0.05            # feet may be outside the mask by this share of the box height

    @classmethod
    def from_dict(cls, d: dict | None) -> SegmentationCourtConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class SegmentationCourtFilter:
    """C06: the court area predicted by a segmentation model, frame by frame (no homography needed)."""

    def __init__(self, cfg: SegmentationCourtConfig | None = None) -> None:
        from ..court.segment import CourtSegmenter
        self.cfg = cfg or SegmentationCourtConfig()
        self.seg = CourtSegmenter(self.cfg.weights, self.cfg.imgsz, self.cfg.conf)
        self.reset()

    def reset(self) -> None:
        self.seg.reset()
        self.last_hull: np.ndarray | None = None
        self.seen = self.removed = self.frames_empty = 0

    def keep_mask(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        from ..court.segment import feet_inside
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        mask = self.seg(frame)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_hull = max(cnts, key=cv2.contourArea).reshape(-1, 2) if cnts else None
        if self.last_hull is None:                    # no court in view (close-up): do not filter
            self.frames_empty += 1
            keep = np.ones(len(boxes), bool)
        else:
            keep = feet_inside(mask, boxes, self.cfg.margin)
        self.seen += len(boxes)
        self.removed += int((~keep).sum())
        return keep

    def __call__(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray):
        keep = self.keep_mask(frame, boxes)
        return np.asarray(boxes).reshape(-1, 4)[keep], np.asarray(scores).reshape(-1)[keep]

    def stats(self) -> dict:
        return {"court_frames_no_court": self.frames_empty,
                "court_boxes_removed_%": 100.0 * self.removed / max(self.seen, 1)}


def build_court_filters(cfg: dict | None) -> list | None:
    """Filters for ``filters.court_source: homography | segmentation``; None if colour (caller keeps 003)."""
    cfg = cfg or {}
    src = cfg.get("court_source", "colour")
    if src == "segmentation":
        return [SegmentationCourtFilter(SegmentationCourtConfig.from_dict(cfg.get("segmentation")))]
    if src != "homography":
        return None
    hcfg = HomographyCourtConfig.from_dict(cfg.get("homography"))
    colour = CourtMask(CourtMaskConfig.from_dict(cfg.get("court_mask_params"))) if hcfg.fallback_colour else None
    return [HomographyCourtFilter(hcfg, colour)]
