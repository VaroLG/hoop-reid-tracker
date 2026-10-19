"""Calibrate the court-mask margin on the TRAIN split (experiment 008).

Question: how close to the edge of the visible floor do *players* put their
feet? The floor hull (CourtMask, 003) covers all the wood, including the apron
strip outside the lines where referees, bench players and staff stand. If
players never step there, the mask can require feet to be a little INSIDE the
hull and drop those people without touching a single player.

For every ground-truth player box (players only: SportsMOT does not annotate
anybody else) we measure the signed distance from the foot point to the hull
edge, in units of the box height (> 0 inside, < 0 outside), and propose

    margin = -(1st percentile)      -> keeps 99 % of player boxes

Same rule as 005b: thresholds come from train data, never from the clip we
evaluate on.

Usage:
    python scripts/calibrate_court_margin.py --stride 5
    python scripts/calibrate_court_margin.py --stride 5 --out experiments/008_court_border/calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hooptrack.config import load_config
from hooptrack.data.sportsmot import list_sequences
from hooptrack.detection.filters import CourtMask, CourtMaskConfig


def foot_distances(seq, mask: CourtMask, stride: int) -> np.ndarray:
    """Signed foot-to-hull-edge distance / box height for every GT box (sampled frames)."""
    gt = seq.load_gt()
    out = []
    for f in range(1, seq.length + 1, stride):
        if f not in gt:
            continue
        _, boxes = gt[f]
        frame = cv2.imread(str(seq.frame_path(f)))
        hull = mask.estimate_hull(frame, boxes)        # GT boxes stand in for detections
        if hull is None:
            continue                                   # mask not trusted on this frame
        for x1, y1, x2, y2 in boxes:
            d = cv2.pointPolygonTest(hull, (float((x1 + x2) / 2), float(y2)), True)
            out.append(d / max(y2 - y1, 1.0))
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=5, help="use every N-th frame")
    ap.add_argument("--out", type=Path, default=None, help="folder for calibration json + plot")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], args.split, d.get("sport"), d["splits_dir"])
    mask = CourtMask(CourtMaskConfig.from_dict((cfg.get("filters") or {}).get("court_mask_params")))

    per_seq, all_d = {}, []
    for seq in seqs:
        mask.reset()
        dist = foot_distances(seq, mask, args.stride)
        if len(dist):
            per_seq[seq.name] = {"boxes": int(len(dist)), "p1": float(np.percentile(dist, 1)),
                                 "outside_%": float(100 * (dist < 0).mean()),
                                 # players the CURRENT margin would delete in this video
                                 "dropped_at_current_margin_%": float(100 * (dist < -mask.cfg.margin).mean())}
            all_d.append(dist)
        r = per_seq.get(seq.name, {})
        print(f"{seq.name}: {len(dist)} player boxes, p1 = {r.get('p1', float('nan')):+.3f}, "
              f"dropped at margin {mask.cfg.margin}: {r.get('dropped_at_current_margin_%', float('nan')):.1f} %")
    dist = np.concatenate(all_d)
    pct = {f"p{q}": float(np.percentile(dist, q)) for q in (0.5, 1, 5, 50)}
    result = {
        "split": args.split, "sequences": len(per_seq), "player_boxes": int(len(dist)),
        "percentiles": pct,
        "outside_hull_%": float(100 * (dist < 0).mean()),
        "current_margin": mask.cfg.margin,
        "dropped_at_current_margin_%": float(100 * (dist < -mask.cfg.margin).mean()),
        "suggested_margin": round(-pct["p1"], 3),
        "per_sequence": per_seq,
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_sequence"}, indent=2))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "court_margin.json").write_text(json.dumps(result, indent=2))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(np.clip(dist, -0.5, 3), bins=140, color="#4a7bd0")
        ax.axvline(pct["p1"], color="#d04a4a", ls="--", label=f"p1 = {pct['p1']:+.2f}")
        ax.axvline(-0.15, color="#888", ls=":", label="003 margin (feet up to 0.15 h outside)")
        ax.set_xlabel("foot distance to floor edge / box height  (> 0 = inside)")
        ax.set_ylabel("player boxes")
        ax.set_title(f"Players' feet vs floor edge — {args.split} split, {len(dist):,} boxes")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "court_margin.png", dpi=120)
        print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
