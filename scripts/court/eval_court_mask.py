"""Court mask from the homography vs the colour court mask (court chapter, C04).

For every 2nd frame, each detection (score >= 0.5) is labelled *player* if it
matches a ground-truth box (IoU >= 0.5) — SportsMOT only labels the ten players
— and *non-player* otherwise (referees, bench, crowd, photographers). A mask
should keep all players and remove as many non-players as possible.

* colour mask: ``CourtMask`` (experiment 003), the tracker's current filter;
* homography mask: a person is kept if their feet (bottom centre of the box),
  projected onto the court plan, fall inside the court rectangle plus a margin.

Usage:
    python scripts/court/eval_court_mask.py \
        --homographies outputs/court_homographies/v_00HRwkvvjtQ_c001/homographies.npz \
        --sequence v_00HRwkvvjtQ_c001 --out experiments/court/C04_anchor_homographies/mask_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.leagues import SPECS
from hooptrack.detection.filters import CourtMask
from hooptrack.tracking.matching import iou_matrix


def feet_on_court(H: np.ndarray, boxes: np.ndarray, length: float, width: float, margin: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0, bool)
    feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
    c = cv2.perspectiveTransform(feet[None].astype(np.float64), np.linalg.inv(H))[0]
    return (c[:, 0] >= -margin) & (c[:, 0] <= length + margin) & (c[:, 1] >= -margin) & (c[:, 1] <= width + margin)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homographies", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--margins", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    z = np.load(args.homographies)
    spec = SPECS[str(z["league"])]
    Hs = dict(zip(z["frames"].tolist(), z["H"], strict=True))
    clip = Clip(args.sequence)
    gt = clip.seq.load_gt()
    cm = CourtMask()
    cm.reset()
    names = ["colour"] + [f"homography_{m}m" for m in args.margins]
    res = {n: np.zeros(4, int) for n in names}      # players kept, players, non-players removed, non-players
    for f in range(1, clip.length + 1, args.stride):
        img = clip.frame(f)
        b, s = clip.det(img, frame_idx=f)
        b = b[s >= 0.5]
        if not len(b) or f not in Hs:
            continue
        gb = gt.get(f, (None, np.zeros((0, 4))))[1]
        player = (iou_matrix(b, gb).max(1) >= 0.5) if len(gb) else np.zeros(len(b), bool)
        keeps = {"colour": cm.keep_mask(img, b)}
        for m in args.margins:
            keeps[f"homography_{m}m"] = feet_on_court(Hs[f], b, spec.length, spec.width, m)
        for n, k in keeps.items():
            res[n] += [(k & player).sum(), player.sum(), (~k & ~player).sum(), (~player).sum()]
    summary = {n: {"players_kept_%": round(100 * r[0] / max(r[1], 1), 2),
                   "non_players_removed_%": round(100 * r[2] / max(r[3], 1), 1),
                   "players": int(r[1]), "non_players": int(r[3])} for n, r in res.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
