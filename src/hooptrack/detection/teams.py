"""Unsupervised team / referee classification from jersey colours (experiment 006).

Why
---
Experiments 004a and 005 showed that 14 of the remaining ID switches are swaps
between players of DIFFERENT teams, and that no component (camera compensation,
ReID, memory) removes a single one of them. Referees are also still tracked:
they walk on the court, so the court mask (003) keeps them.

Two jersey colours + referees' stripes are an easy visual cue, available on CPU
and without labels. This module turns it into a label per detection:

    0, 1   -> team A / team B
    OTHER  -> neither team (referees, staff)
    -1     -> unknown (warm-up, occluded crop, or not confident)

How
---
1. Feature per box = three pixel fractions of the torso (20–50 % of the height,
   central half of the width), in HSV:
       white     (bright, unsaturated)      high for white jerseys
       black     (dark)                     high for referees' stripes
       saturated (any vivid colour)         high for coloured jerseys
   Plain Lab colour was tried first: referees' grey average merges with the
   darker team. Fractions keep "half black, half white, no colour" apart.
2. Warm-up: collect features of clean, confident boxes for the first frames,
   standardise them and run k-means with k = 3.
3. Which cluster is "other"? The one with the fewest boxes per frame: there are
   ~5 players of each team on court but only ~2–3 referees. If that is not
   clear-cut, the classifier declares itself unreliable and labels nothing
   (the tracker then behaves exactly as without it).
4. After warm-up, each box gets the nearest cluster **only if** it is clearly
   closer to it than to the second nearest (distance ratio < ``confidence_ratio``).

Measured on SportsMOT ``v_00HRwkvvjtQ_c001`` (prototype, ratio 0.7): 81 % of
player boxes labelled, 2.7 % of them with the wrong team; 80 % of non-player
boxes labelled "other". Per-box noise is why decisions are taken per TRACK, by
majority vote (see ``Track.team``).

006b (``label_occluded``): after warm-up, occluded boxes are labelled too, from
the torso pixels not hidden by a player closer to the camera.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..tracking.matching import coverage_matrix

OTHER = 2      # label for "neither team" (referees, staff)
UNKNOWN = -1


@dataclass
class TeamClassifierConfig:
    warmup_frames: int = 60          # frames used to fit the colour clusters
    min_samples: int = 60            # need at least this many clean boxes to fit
    confidence_ratio: float = 0.7    # d(nearest) / d(2nd nearest) must be below this
    min_score: float = 0.5           # only confident detections are labelled
    max_coverage: float = 0.2        # ... and only clean crops (not hidden by another box)
    other_count_ratio: float = 0.8   # "other" must have < 80 % of the boxes/frame of the next cluster
    # 006b: also label OCCLUDED boxes from their visible torso pixels. Off = 006a
    # (only clean crops are labelled), which turned out to be blind exactly where
    # cross-team swaps happen: every one of them involved an occluded box.
    label_occluded: bool = False
    min_visible: float = 0.4         # visible share of the torso needed to label an occluded box
    # 007a: fit a first model after this many frames, then refit at ``warmup_frames``
    # with all the samples. 0 = off (006: labels only start after ``warmup_frames``,
    # so referees visible at the start of a clip are shown for ~2-3 s).
    early_fit_frames: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> TeamClassifierConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def torso_colour_features(
    frame: np.ndarray, boxes: np.ndarray, occluders: np.ndarray | None = None, min_visible: float = 0.4,
) -> np.ndarray:
    """(N, 3) = fraction of white, black and saturated pixels on each torso.

    ``occluders`` (optional, M boxes): pixels of a torso that fall inside an
    occluder box **closer to the camera** (its bottom edge is lower in the image,
    so its feet are nearer) are ignored. A torso with less than ``min_visible``
    of its pixels left is returned as NaN (= cannot be labelled).
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    occ = np.zeros((0, 4)) if occluders is None else np.asarray(occluders, dtype=float).reshape(-1, 4)
    feats = np.full((len(boxes), 3), np.nan, dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        h, w = y2 - y1, x2 - x1
        ty1, tx1 = max(y1 + int(0.2 * h), 0), max(x1 + int(0.25 * w), 0)
        ty2, tx2 = max(y1 + int(0.5 * h), 0), max(x1 + int(0.75 * w), 0)
        patch = hsv[ty1:ty2, tx1:tx2]
        if patch.shape[0] * patch.shape[1] < 20:   # box too small or outside the image
            continue
        visible = np.ones(patch.shape[:2], dtype=bool)
        for o in occ:
            ox1, oy1, ox2, oy2 = o
            if oy2 <= box[3] or np.allclose(o, box):
                continue                               # behind this player (or itself)
            visible[max(int(oy1) - ty1, 0): max(int(oy2) - ty1, 0),
                    max(int(ox1) - tx1, 0): max(int(ox2) - tx1, 0)] = False
        if visible.mean() < min_visible:
            continue
        px = patch[visible]
        if len(px) < 20:
            continue
        s, v = px[:, 1].astype(np.float32), px[:, 2].astype(np.float32)
        feats[i] = (
            ((v > 170) & (s < 60)).mean(),   # white
            (v < 70).mean(),                 # black
            ((s > 90) & (v >= 70)).mean(),   # saturated colour
        )
    return feats


class TeamClassifier:
    def __init__(self, cfg: TeamClassifierConfig | None = None) -> None:
        self.cfg = cfg or TeamClassifierConfig()
        self.reset()

    def reset(self) -> None:
        """Per-video state: every clip has its own two teams."""
        self.frame = 0
        self._samples: list[np.ndarray] = []
        self._sample_frames: list[int] = []
        self.fitted = False
        self.reliable = False
        self._early_done = False      # 007a: first (early) fit already attempted
        self._final = False           # final fit done: stop collecting samples
        self._mu = self._sd = self._centres = None
        self._cluster_to_label: dict[int, int] = {}
        self.labelled = self.seen = 0
        self.counts_per_frame: dict[int, float] = {}

    # ---------------------------------------------------------------- fitting
    def _usable(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros(0, dtype=bool)
        clean = (coverage_matrix(boxes).max(axis=1) < self.cfg.max_coverage
                 if len(boxes) > 1 else np.ones(1, dtype=bool))
        return clean & (np.asarray(scores) >= self.cfg.min_score)

    def _fit(self) -> None:
        self.fitted = True
        self.reliable = False                        # a failed refit discards the early model
        X = np.array(self._samples, dtype=np.float32)
        if len(X) < self.cfg.min_samples:
            return                                   # not enough evidence: stay unreliable
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-6
        Z = (X - self._mu) / self._sd
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
        _, lbl, centres = cv2.kmeans(Z, 3, None, crit, 10, cv2.KMEANS_PP_CENTERS)
        lbl = lbl.ravel()
        frames = np.array(self._sample_frames)
        n_frames = len(np.unique(frames))
        counts = {k: float((lbl == k).sum()) / n_frames for k in range(3)}
        self.counts_per_frame = counts
        order = sorted(counts, key=counts.get)       # fewest boxes per frame first
        if counts[order[0]] >= self.cfg.other_count_ratio * counts[order[1]]:
            return                                   # "other" not clear-cut: unreliable
        # Teams are numbered by whiteness so labels are stable between runs.
        teams = sorted(order[1:], key=lambda k: -centres[k][0])
        self._cluster_to_label = {teams[0]: 0, teams[1]: 1, order[0]: OTHER}
        self._centres = centres
        self.reliable = True

    # -------------------------------------------------------------- labelling
    def __call__(self, frame: np.ndarray, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Label every box (``UNKNOWN`` until the classifier is fitted and confident)."""
        self.frame += 1
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        labels = np.full(len(boxes), UNKNOWN, dtype=int)
        clean = self._usable(boxes, scores)            # used to FIT: clean crops only
        was_reliable = self.reliable                   # a model fitted THIS frame labels from the next one

        # ---- 1. collect samples and (re)fit ------------------------------------
        if not self._final and clean.any():
            cf = torso_colour_features(frame, boxes[clean])
            for f in cf[~np.isnan(cf).any(axis=1)]:
                self._samples.append(f)
                self._sample_frames.append(self.frame)
            early = self.cfg.early_fit_frames
            if self.frame >= self.cfg.warmup_frames:
                self._fit()                            # final model, all warm-up samples
                self._final = True
            elif early > 0 and not self._early_done and self.frame >= early:
                self._fit()                            # 007a: provisional model
                self._early_done = True
        if not was_reliable or not self.reliable:
            return labels

        # ---- 2. label --------------------------------------------------------------
        if self.cfg.label_occluded:
            usable = np.asarray(scores) >= self.cfg.min_score
            feats = torso_colour_features(frame, boxes[usable], occluders=boxes,
                                          min_visible=self.cfg.min_visible)
        else:
            usable = clean
            feats = torso_colour_features(frame, boxes[usable]) if usable.any() else np.zeros((0, 3))
        if not usable.any():
            return labels
        ok = ~np.isnan(feats).any(axis=1)

        Z = (feats - self._mu) / self._sd
        D = np.sqrt(((Z[:, None, :] - self._centres[None]) ** 2).sum(-1))
        nearest = np.argsort(D, axis=1)
        d1 = D[np.arange(len(D)), nearest[:, 0]]
        d2 = D[np.arange(len(D)), nearest[:, 1]]
        confident = ok & (d1 / np.maximum(d2, 1e-9) < self.cfg.confidence_ratio)
        idx = np.where(usable)[0]
        for j, (c, k) in enumerate(zip(confident, nearest[:, 0], strict=True)):
            if c:
                labels[idx[j]] = self._cluster_to_label[int(k)]
        self.seen += int(usable.sum())
        self.labelled += int((labels >= 0).sum())
        return labels

    def classify(self, frame: np.ndarray, boxes: np.ndarray, others: np.ndarray | None = None) -> np.ndarray:
        """Label ``boxes`` with the current model, without updating any state (007c).

        Used offline to label tracks that ended before the model existed (e.g. a
        referee seen only during warm-up). ``others``: the rest of the boxes in
        the frame, used to mask out occluded torso pixels.
        """
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        labels = np.full(len(boxes), UNKNOWN, dtype=int)
        if not self.reliable or len(boxes) == 0:
            return labels
        occ = boxes if others is None else np.vstack([boxes, np.asarray(others, dtype=float).reshape(-1, 4)])
        feats = torso_colour_features(frame, boxes, occluders=occ, min_visible=self.cfg.min_visible)
        ok = ~np.isnan(feats).any(axis=1)
        if not ok.any():
            return labels
        Z = (feats[ok] - self._mu) / self._sd
        D = np.sqrt(((Z[:, None, :] - self._centres[None]) ** 2).sum(-1))
        nearest = np.argsort(D, axis=1)
        d1 = D[np.arange(len(D)), nearest[:, 0]]
        d2 = D[np.arange(len(D)), nearest[:, 1]]
        conf = d1 / np.maximum(d2, 1e-9) < self.cfg.confidence_ratio
        idx = np.where(ok)[0]
        for j in np.where(conf)[0]:
            labels[idx[j]] = self._cluster_to_label[int(nearest[j, 0])]
        return labels

    def stats(self) -> dict:
        return {
            "teams_reliable": int(self.reliable),
            "teams_labelled_%": 100.0 * self.labelled / max(self.seen, 1),
        }


def build_team_classifier(cfg: dict | None) -> TeamClassifier | None:
    """``teams.enabled: true`` -> classifier; absent / false -> None (behaviour up to 005)."""
    cfg = cfg or {}
    if not cfg.get("enabled", False):
        return None
    return TeamClassifier(TeamClassifierConfig.from_dict(cfg))
