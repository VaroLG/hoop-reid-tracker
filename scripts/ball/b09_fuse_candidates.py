"""B09 · WASB's candidates + our YOLO's, in one tracker: chosen on TRAIN, tested once.

Why
---
B08 (fine-tuned WASB + B05's tracker) finds 87 % of the balls in the air or on the floor, but only 70 % of those
seen over a player and 38 % over two or more (dribbles, hands, crowds), which is where possession is decided. On
41 % of B08's misses the YOLO of B03 had a candidate on the right spot (52 % for a ball over one player): the two
detectors fail in different places. B03 alone is noisy (still look-alikes), WASB alone is precise.

Change (one value)
------------------
Both detectors' candidates go into the same B05 tracker (dynamic programming over one ball, physics-limited moves);
YOLO's confidences are multiplied by ``alpha`` (0 = B08 exactly). A YOLO candidate is then kept only when the path
needs it, i.e. when WASB has nothing nearby that fits the motion.

1. **Choice, on the 3 held-out TRAIN clips** (hand clicks of B02; neither detector was trained on them): alpha in
   0, 0.1, 0.25, 0.5, 0.75, 1; best F1 (criterion fixed before running).
2. **Test, once, on the 4 test clips**, same scoring as B04-B08.

Usage (repo root):
    python scripts/ball/b09_fuse_candidates.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b04_eval_ball import score  # noqa: E402

from hooptrack.ball.labels import ball_diameter, load_labels  # noqa: E402
from hooptrack.ball.track import BallTrackConfig, track_ball  # noqa: E402
from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402

TRAIN_CLIPS = ["v_-6Os86HzwCs_c009", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c013"]
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def candidates(wasb: Path, yolo: Path, alpha: float) -> dict[int, np.ndarray]:
    """{frame: (N, 5) [cx, cy, d, conf, source]}: source 0 = WASB, 1 = YOLO (conf x alpha)."""
    parts = [pd.read_csv(wasb).assign(src=0)]
    if alpha > 0:
        y = pd.read_csv(yolo)
        parts.append(y.assign(conf=y["conf"] * alpha, src=1))
    c = pd.concat(parts)
    return {int(f): g[["cx", "cy", "d", "conf", "src"]].to_numpy(float) for f, g in c.groupby("frame")}


def evaluate(labels: pd.DataFrame, seqs: dict, wasb_dir: Path, yolo_dir: Path, alpha: float,
             cfg: BallTrackConfig) -> pd.DataFrame:
    rows = []
    for clip, lab in labels.groupby("clip"):
        track = track_ball(candidates(wasb_dir / f"{clip}.csv", yolo_dir / f"{clip}.csv", alpha), cfg)
        gt = seqs[clip].load_gt()
        for _, r in lab.iterrows():
            f, vis = int(r["frame"]), bool(r["visible"])
            diam = ball_diameter(r["x"], r["y"], gt.get(f, (None, np.zeros((0, 4))))[1]) if vis else 0.0
            row = {"clip": clip, "frame": f, "visible": vis, "conf": 0.0, "dist": np.inf, "diam": diam}
            if f in track:
                row["conf"] = 1.0
                if vis:
                    row["dist"] = float(np.hypot(track[f][0] - r["x"], track[f][1] - r["y"]))
            rows.append(row)
    return pd.DataFrame(rows)


def f1(s: dict) -> float:
    return 2 * s["recall"] * s["precision"] / max(1e-9, s["recall"] + s["precision"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--train-wasb", type=Path, default=Path("outputs/ball/dets_wasb_ft_train"))
    ap.add_argument("--train-yolo", type=Path, default=Path("outputs/ball/dets_b03_train"))
    ap.add_argument("--test-wasb", type=Path, default=Path("outputs/ball/dets_wasb_ft"))
    ap.add_argument("--test-yolo", type=Path, default=Path("outputs/ball/dets_b03"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/b09"))
    args = ap.parse_args()
    cfg = BallTrackConfig()                       # B05's tracker as is (B08)
    cfg_all = load_config(args.config, [])
    seqs = {s.name: s for split in ("train", "val") for s in
            _sequences(apply_overrides(cfg_all, [f"data.split={split}", "data.only=null"]))}
    args.out.mkdir(parents=True, exist_ok=True)

    train = load_labels(Path("data/ball_labels/train.csv"))
    train = train[train["clip"].isin(TRAIN_CLIPS)]
    grid = []
    for a in ALPHAS:
        s = score(evaluate(train, seqs, args.train_wasb, args.train_yolo, a, cfg), 0.5)
        grid.append({"alpha": a, **s, "f1": round(f1(s), 3)})
    g = pd.DataFrame(grid)
    g.to_csv(args.out / "train_grid.csv", index=False)
    cols = ["alpha", "visible", "found", "wrong_place", "missed", "false_alarm_no_ball", "recall", "precision", "f1"]
    print("TRAIN (3 held-out clips):\n" + g[cols].to_string(index=False))
    best = float(g.loc[g["f1"].idxmax(), "alpha"])
    print(f"\nchosen on train: alpha = {best}")

    test = load_labels(Path("data/ball_labels/val.csv"))
    table = []
    for name, a in [("B08 (alpha 0)", 0.0), (f"B09 (alpha {best})", best)]:
        df = evaluate(test, seqs, args.test_wasb, args.test_yolo, a, cfg)
        for clip, gdf in list(df.groupby("clip")) + [("ALL", df)]:
            s = score(gdf, 0.5)
            table.append({"method": name, "clip": clip, **s, "f1": round(f1(s), 3)})
    t = pd.DataFrame(table)
    t.to_csv(args.out / "per_clip.csv", index=False)
    print("\nTEST (4 clips, once):\n" + t[["method", "clip", "visible", "found", "wrong_place", "missed",
                                            "false_alarm_no_ball", "recall", "precision", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
