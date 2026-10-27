"""Team (0/1) of every ground-truth player, without team labels.

SportsMOT gives player ids but not teams. The median torso colour (Lab) of each
ground-truth player, sampled over the clip, is clustered with k-means (k = 2):
the two kits are the two clusters. Used by evaluation tools only (004 ID-switch
breakdown, 016 team-mate ReID benchmark), never by the tracker.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..data.sportsmot import Sequence


def player_teams(seq: Sequence, stride: int = 25) -> dict[int, int]:
    """Ground-truth id -> team (0/1) from jersey colour clustering."""
    gt = seq.load_gt()
    colours: dict[int, list[np.ndarray]] = {}
    for f in range(1, seq.length + 1, stride):
        if f not in gt:
            continue
        lab = cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2LAB)
        ids, boxes = gt[f]
        for pid, (x1, y1, x2, y2) in zip(ids, boxes.astype(int), strict=True):
            h, w = y2 - y1, x2 - x1
            # torso: 20-50 % of the height, central half of the width
            torso = lab[max(y1 + int(0.2 * h), 0): y1 + int(0.5 * h),
                        max(x1 + int(0.25 * w), 0): x1 + int(0.75 * w)]
            if torso.size:
                colours.setdefault(int(pid), []).append(np.median(torso.reshape(-1, 3), axis=0))
    ids = sorted(colours)
    X = np.array([np.median(colours[i], axis=0) for i in ids], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, _ = cv2.kmeans(X, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return {i: int(lbl) for i, lbl in zip(ids, labels.ravel(), strict=True)}
