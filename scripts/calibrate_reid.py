"""Calibrate the appearance thresholds for a given ReID extractor.

Cosine-distance thresholds are NOT universal: a colour histogram puts two
team-mates at ~0.33 (half the vector — the jersey — is identical), while a
trained ReID network may put them at ~0.6. Copying thresholds from a paper
for a different embedding is a classic source of identity switches.

This script measures, on the TRAIN split ground truth (never on val, to avoid
tuning on the evaluation data):

  * same-identity distances   (one player, two frames >= --min-gap apart)
  * different-identity distances, split into team-mate-like hard negatives
    (the closest other player) and all negatives

and proposes:
  * ``appearance_gate`` = 95th percentile of same-identity distances
    (keeps 95 % of true matches inside the gate)
  * ``memory_thresh``   = 1st percentile of HARD-negative distances
    (reviving the wrong person must be rare — it is the costliest error)

Usage:
    python scripts/calibrate_reid.py --config configs/default.yaml --set reid.type=osnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import list_sequences
from hooptrack.reid import build_reid
from hooptrack.tracking.matching import cosine_distance, coverage_matrix


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=10, help="sample 1 frame every N")
    ap.add_argument("--min-gap", type=int, default=30, help="frames between two views of a player")
    ap.add_argument("--max-seqs", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("outputs/calibration"))
    args = ap.parse_args()

    cfg = apply_overrides(load_config(args.config, args.set), ["data.split=train"])
    reid = build_reid(cfg["reid"])
    if reid is None:
        raise SystemExit("reid.type=none has nothing to calibrate")
    d = cfg["data"]
    seqs = list_sequences(d["root"], "train", d.get("sport"), d["splits_dir"])[: args.max_seqs]

    same, hard_neg, all_neg = [], [], []
    for seq in seqs:
        gt = seq.load_gt()
        feats: dict[int, list[tuple[int, np.ndarray]]] = {}
        for f in range(1, seq.length + 1, args.stride):
            ids, boxes = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
            if len(ids) == 0:
                continue
            clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(1, bool)
            emb = reid(cv2.imread(str(seq.frame_path(f))), boxes[clean])
            for tid, e in zip(ids[clean], emb, strict=True):
                feats.setdefault(int(tid), []).append((f, e))

        tids = list(feats)
        for a in tids:
            fa = np.stack([e for _, e in feats[a]])
            frames_a = np.array([f for f, _ in feats[a]])
            # positives: same player, far apart in time (pose/viewpoint changed)
            dd = cosine_distance(fa, fa)
            far = np.abs(frames_a[:, None] - frames_a[None, :]) >= args.min_gap
            same.extend(dd[np.triu(far, 1)].tolist())
            # negatives
            closest = []
            for b in tids:
                if b == a:
                    continue
                db = cosine_distance(fa, np.stack([e for _, e in feats[b]]))
                all_neg.extend(db.flatten().tolist())
                closest.append(db.min(axis=1))
            if closest:
                hard_neg.extend(np.min(np.stack(closest), axis=0).tolist())
        print(f"{seq.name}: {len(tids)} players")

    same, hard_neg, all_neg = map(np.asarray, (same, hard_neg, all_neg))

    def pct(x: np.ndarray, qs: list[int]) -> dict[str, float]:
        return {f"p{q}": round(float(np.percentile(x, q)), 3) for q in qs}

    report = {
        "extractor": cfg["reid"]["type"],
        "n_pairs": {"same": len(same), "hard_neg": len(hard_neg), "all_neg": len(all_neg)},
        "same_pct": pct(same, [50, 90, 95, 99]),
        "hard_neg_pct": pct(hard_neg, [1, 5, 50]),
        "suggested": {
            "tracker.appearance_gate": round(float(np.percentile(same, 95)), 3),
            "tracker.memory_thresh": round(float(np.percentile(hard_neg, 1)), 3),
        },
    }
    if report["suggested"]["tracker.memory_thresh"] >= report["suggested"]["tracker.appearance_gate"]:
        report["note"] = "Embedding separates identities well: memory_thresh may be relaxed."
    else:
        report["note"] = (
            "Hard negatives overlap with positives: this extractor cannot reliably tell "
            "team-mates apart. Fine-tune it (scripts/export_reid_crops.py)."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{cfg['reid']['type']}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 3.5))
        bins = np.linspace(0, 1, 60)
        ax.hist(same, bins, alpha=0.6, density=True, label="same player")
        ax.hist(hard_neg, bins, alpha=0.6, density=True, label="closest other player")
        for k, v in report["suggested"].items():
            ax.axvline(v, ls="--", c="k", lw=1)
            ax.text(v, ax.get_ylim()[1] * 0.9, k.split(".")[1], rotation=90, ha="right", fontsize=8)
        ax.set_xlabel("cosine distance")
        ax.set_title(f"ReID separability — {cfg['reid']['type']}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / f"{cfg['reid']['type']}.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
