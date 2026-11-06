"""046 ceiling · The links a PERFECT crossing verifier would give (ground truth; evaluation only).

038 cuts every online tracklet at the end of each crossing (``split_at_crossings``) and then decides every piece
by appearance; the online tracker kept the right person through ~96 % of the crossings (032), but 038 only uses that
as a soft bonus. A mask-propagation verifier (SAM2 / EdgeTAM through the crossing) would say "same person" for the
cuts where the tracker was right, and the pieces could be joined before the assignment. This writes, for every cut,
the link (piece before, piece after) when both pieces are mostly the SAME ground-truth player: the input of
``stitch_tracklets.py --link-file`` for the ceiling of that idea. A real verifier will confirm fewer and get some
wrong; this is the most it can give.

Usage (repo root):
    python scripts/crossing_links_oracle.py --run outputs/runs/011b_no_referee_filter/segmentation \\
        --out outputs/runs/046_links_oracle.csv
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import load_mot_txt
from hooptrack.tracking.matching import iou_matrix
from hooptrack.tracking.stitch import split_at_crossings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--run", type=Path, default=Path("outputs/runs/011b_no_referee_filter/segmentation"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=val", "data.only=null"]))}
    rows, stats = [], Counter()
    for clip_dir in sorted(p for p in args.run.iterdir() if p.is_dir()):
        clip = clip_dir.name
        pred = load_mot_txt(clip_dir / "results" / f"{clip}.txt")
        seg = split_at_crossings(pred, 0.2)
        gt = seqs[clip].load_gt()
        votes: dict[int, Counter] = defaultdict(Counter)
        for f, (ids, boxes) in pred.items():
            if f not in gt or not len(gt[f][0]):
                continue
            iou = iou_matrix(np.asarray(boxes, float), np.asarray(gt[f][1], float))
            for i, t in enumerate(ids):
                j = int(iou[i].argmax())
                if iou[i, j] >= 0.5:
                    votes[seg[(f, int(t))]][int(gt[f][0][j])] += 1
        who = {p: v.most_common(1)[0][0] for p, v in votes.items()}
        chain: dict[int, list] = defaultdict(list)
        for (f, t), p in seg.items():
            chain[t].append((f, p))
        for t, lst in chain.items():
            order = [p for _, p in sorted(lst)]
            for a, b in zip(order, order[1:], strict=False):
                if a == b:
                    continue
                stats["cuts"] += 1
                if a in who and b in who and who[a] == who[b]:
                    rows.append({"clip": clip, "a": a, "b": b})
                    stats["same person"] += 1
                elif a in who and b in who:
                    stats["swap"] += 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(dict(stats), f"-> {len(rows)} links in {args.out}")


if __name__ == "__main__":
    main()
