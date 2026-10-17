"""Identity-centric evaluation.

Which metric answers which question?
------------------------------------
* MOTA          — mostly detection quality (FP + FN dominate). Says little
                  about identity.
* IDF1          — fraction of boxes carrying the CORRECT identity, after a
                  global one-to-one gt<->pred id assignment. The headline
                  metric for "did we keep identities?".
* ID switches   — how many times a gt player's predicted id changed. The
                  direct count of "substitutions".
* HOTA / AssA   — (via TrackEval) HOTA = sqrt(DetA * AssA). AssA isolates
                  association quality; it is the SportsMOT leaderboard metric.
* Re-entry recovery (ours) — of all the times a player LEFT the frame and
                  CAME BACK, in what fraction did he get his old id back?
                  Short-term trackers score ~0 % here by construction; it is
                  the metric that shows what the long-term memory buys.

MOTA/IDF1/IDSW are computed with py-motmetrics; HOTA with the official
TrackEval code (see ``trackeval_runner.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import motmetrics as mm
import numpy as np
import pandas as pd

from ..tracking.matching import iou_matrix, linear_assignment

MOT_METRICS = [
    "idf1", "idp", "idr", "mota", "motp", "num_switches", "num_fragmentations",
    "mostly_tracked", "mostly_lost", "num_false_positives", "num_misses", "num_unique_objects",
]

Frames = dict[int, tuple[np.ndarray, np.ndarray]]  # frame -> (ids, TLBR boxes)


def _iou_distance(gt_boxes: np.ndarray, pr_boxes: np.ndarray, thresh: float) -> np.ndarray:
    """motmetrics expects distances with NaN for "cannot match"."""
    iou = iou_matrix(gt_boxes, pr_boxes)
    dist = 1.0 - iou
    dist[iou < thresh] = np.nan
    return dist


def accumulate(gt: Frames, pred: Frames, iou_thresh: float = 0.5) -> mm.MOTAccumulator:
    acc = mm.MOTAccumulator(auto_id=True)
    empty = (np.zeros(0, dtype=int), np.zeros((0, 4)))
    for f in sorted(set(gt) | set(pred)):
        g_ids, g_boxes = gt.get(f, empty)
        p_ids, p_boxes = pred.get(f, empty)
        acc.update(list(g_ids), list(p_ids), _iou_distance(g_boxes, p_boxes, iou_thresh))
    return acc


def clear_and_identity(
    per_seq: dict[str, tuple[Frames, Frames]], iou_thresh: float = 0.5
) -> pd.DataFrame:
    """CLEAR-MOT + IDF1 per sequence plus an OVERALL row."""
    accs, names = [], []
    for name, (gt, pred) in per_seq.items():
        accs.append(accumulate(gt, pred, iou_thresh))
        names.append(name)
    mh = mm.metrics.create()
    return mh.compute_many(accs, metrics=MOT_METRICS, names=names, generate_overall=True)


# ------------------------------------------------------------------ re-entry
@dataclass
class ReentryResult:
    events: int = 0        # gt re-entries where the player was tracked before AND after
    recovered: int = 0     # ... and kept the same predicted id
    new_id: int = 0        # ... got an id never used before (fragmentation)
    stolen: int = 0        # ... got an id that belonged to ANOTHER player (worst case)

    @property
    def rate(self) -> float:
        return self.recovered / self.events if self.events else float("nan")

    def __add__(self, other: ReentryResult) -> ReentryResult:
        return ReentryResult(
            self.events + other.events, self.recovered + other.recovered,
            self.new_id + other.new_id, self.stolen + other.stolen,
        )


def frame_matches(gt: Frames, pred: Frames, iou_thresh: float = 0.5) -> dict[int, dict[int, int]]:
    """Per-frame Hungarian matching: frame -> {gt_id: pred_id}."""
    out: dict[int, dict[int, int]] = {}
    for f, (g_ids, g_boxes) in gt.items():
        if f not in pred:
            out[f] = {}
            continue
        p_ids, p_boxes = pred[f]
        m, _, _ = linear_assignment(1.0 - iou_matrix(g_boxes, p_boxes), 1.0 - iou_thresh)
        out[f] = {int(g_ids[r]): int(p_ids[c]) for r, c in m}
    return out


def reentry_recovery(
    gt: Frames, pred: Frames, min_gap: int = 15, search: int = 10, iou_thresh: float = 0.5
) -> ReentryResult:
    """Measure identity recovery after a player leaves and re-enters the view.

    A *re-entry* is a gap of at least ``min_gap`` frames in a gt track. For
    each one we take the predicted id matched to the player just before the
    gap and just after it (looking up to ``search`` frames on each side, since
    the detector may need a few frames to pick him up again).
    """
    matches = frame_matches(gt, pred, iou_thresh)

    # gt id -> sorted list of frames where it is annotated
    presence: dict[int, list[int]] = {}
    for f, (ids, _) in gt.items():
        for i in ids:
            presence.setdefault(int(i), []).append(f)

    # pred id -> chronological list of (frame, gt id) it was matched to.
    # Used to tell a brand-new id from an id "stolen" from another player.
    history: dict[int, list[tuple[int, int]]] = {}
    for f in sorted(matches):
        for g, p in matches[f].items():
            history.setdefault(p, []).append((f, g))

    res = ReentryResult()
    for gid, frames in presence.items():
        frames = sorted(frames)
        for a, b in zip(frames[:-1], frames[1:], strict=True):
            if b - a < min_gap:
                continue
            before = next(
                (matches[f][gid] for f in range(a, a - search, -1) if gid in matches.get(f, {})),
                None,
            )
            after = next(
                (matches[f][gid] for f in range(b, b + search) if gid in matches.get(f, {})),
                None,
            )
            if before is None or after is None:
                continue
            res.events += 1
            if after == before:
                res.recovered += 1
            elif any(f < b and g != gid for f, g in history.get(after, [])):
                res.stolen += 1      # that id was already someone else's
            else:
                res.new_id += 1
    return res
