"""B05b / B05c · Static look-alikes removed on the court plan, "no ball" for weak segments: chosen on TRAIN, tested
once.

Two changes to B05's tracker, each behind a value:

* **B05b** ``min_frames``: candidates on a spot of the court plan where detections were seen in at least that many
  frames are background (``hooptrack.integrations.ball_static``);
* **B05c** ``min_segment_score``: segments of the path worth less than that are dropped ("no ball").

1. **Choice, on the 3 held-out TRAIN clips** (hand-clicked in B02, not used for the detector's weights; the detector
   must have been run on all their frames, ``b05_detect_all.py --split train``): every pair of values of the grid is
   scored with B04's rule; the pair with the best F1 (balance of recall and precision, fixed as the criterion before
   running) is kept.
2. **Test, once, on the 4 test clips**: detector alone, B05, B05b (static filter only) and B05c (both), same scoring
   as B04/B05.

Usage (repo root):
    python scripts/ball/b05b_tune_ball.py --train-dets outputs/ball/dets_b03_train --test-dets outputs/ball/dets_b03
"""

from __future__ import annotations

import argparse
import itertools
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
from hooptrack.integrations.ball_static import load_court_h, remove_static  # noqa: E402

COURT_DIRS = ["outputs/court_complete/c09_pipeline", "outputs/court_verified/train",
              "outputs/court_verified/auto_final", "outputs/court_verified/c09_pipeline"]
GRID_FRAMES = [None, 25, 50, 100, 200]            # 1, 2, 4, 8 s at 25 fps
GRID_SCORE = [None, 2.0, 4.0, 8.0, 16.0]


def load_clip(split: str, clip: str, dets_dir: Path, seqs: dict, court_dirs: list[str]):
    c = pd.read_csv(dets_dir / f"{clip}.csv")
    dets = {int(f): np.c_[g[["cx", "cy", "d", "conf"]].to_numpy(), np.zeros(len(g))] for f, g in c.groupby("frame")}
    return dets, load_court_h(court_dirs, clip), seqs[clip].load_gt()


def rows_for(lab: pd.DataFrame, gt: dict, track: dict | None, dets: dict | None, clip: str) -> list[dict]:
    """B04's per-frame rows, from a track (answer = tracked position) or from the detector (best candidate)."""
    out = []
    for _, r in lab.iterrows():
        f, vis = int(r["frame"]), bool(r["visible"])
        row = {"clip": clip, "frame": f, "visible": vis, "conf": 0.0, "dist": np.inf,
               "diam": ball_diameter(r["x"], r["y"], gt.get(f, (None, np.zeros((0, 4))))[1]) if vis else 0.0}
        p = None
        if track is not None and f in track:
            p, row["conf"] = track[f][:2], 1.0
        if dets is not None and f in dets and len(dets[f]):
            d = dets[f][np.argmax(dets[f][:, 3])]
            p, row["conf"] = d[:2], float(d[3])
        if p is not None and vis:
            row["dist"] = float(np.hypot(p[0] - r["x"], p[1] - r["y"]))
        out.append(row)
    return out


def run(clips: dict, labels: pd.DataFrame, min_frames, min_score) -> tuple[list[dict], int]:
    rows, removed = [], 0
    for clip, (dets, H, gt) in clips.items():
        kept, n = remove_static(dets, H, min_frames)
        removed += n
        track = track_ball(kept, BallTrackConfig(min_segment_score=min_score))
        rows += rows_for(labels[labels["clip"] == clip], gt, track, None, clip)
    return rows, removed


def f1(s: dict) -> float:
    return 0.0 if not s["found"] else 2 * s["recall"] * s["precision"] / (s["recall"] + s["precision"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--train-labels", type=Path, default=Path("data/ball_labels/train.csv"))
    ap.add_argument("--test-labels", type=Path, default=Path("data/ball_labels/val.csv"))
    ap.add_argument("--train-dets", type=Path, default=Path("outputs/ball/dets_b03_train"))
    ap.add_argument("--test-dets", type=Path, default=Path("outputs/ball/dets_b03"))
    ap.add_argument("--court-dirs", nargs="*", default=COURT_DIRS)
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/b05b"))
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    seq = {sp: {s.name: s for s in _sequences(apply_overrides(cfg, [f"data.split={sp}", "data.only=null"]))}
           for sp in ("train", "val")}
    args.out.mkdir(parents=True, exist_ok=True)

    # 1 · choice on the held-out train clips
    tl = load_labels(args.train_labels)
    train = {p.stem: load_clip("train", p.stem, args.train_dets, seq["train"], args.court_dirs)
             for p in sorted(args.train_dets.glob("*.csv"))}
    tl = tl[tl["clip"].isin(train)]
    print("train clips:", {c: f"{len(H)} frames with a court" for c, (_, H, _) in train.items()})
    grid = []
    for mf, ms in itertools.product(GRID_FRAMES, GRID_SCORE):
        rows, removed = run(train, tl, mf, ms)
        s = score(pd.DataFrame(rows), 0.5)
        grid.append({"min_frames": mf, "min_segment_score": ms, "removed": removed, **s, "f1": round(f1(s), 3)})
    g = pd.DataFrame(grid)
    g.to_csv(args.out / "grid_train.csv", index=False)
    print("\nGrid on the held-out train clips:\n" + g[["min_frames", "min_segment_score", "removed", "recall",
                                                        "precision", "f1"]].to_string(index=False))
    best = g.sort_values("f1", ascending=False, kind="stable").iloc[0]
    mf = None if pd.isna(best["min_frames"]) else int(best["min_frames"])
    ms = None if pd.isna(best["min_segment_score"]) else float(best["min_segment_score"])
    print(f"\nchosen on train: min_frames={mf}, min_segment_score={ms} (F1 {best['f1']})")

    # 2 · test, once
    vl = load_labels(args.test_labels)
    test = {c: load_clip("val", c, args.test_dets, seq["val"], args.court_dirs) for c in sorted(vl["clip"].unique())
            if (args.test_dets / f"{c}.csv").is_file()}
    table = []
    det_rows = [r for c, (d, _, gt) in test.items() for r in rows_for(vl[vl["clip"] == c], gt, None, d, c)]
    variants = {"detector (B04)": (det_rows, 0.25), "B05 tracker": (run(test, vl, None, None)[0], 0.5),
                f"B05b + static filter ({mf})": (run(test, vl, mf, None)[0], 0.5),
                f"B05c + weak segments out ({ms})": (run(test, vl, mf, ms)[0], 0.5)}
    for name, (rows, conf) in variants.items():
        df = pd.DataFrame(rows)
        for clip, gg in list(df.groupby("clip")) + [("ALL", df)]:
            s = score(gg, conf)
            table.append({"method": name, "clip": clip, **s, "f1": round(f1(s), 3)})
    t = pd.DataFrame(table)
    t.to_csv(args.out / "test.csv", index=False)
    print("\nTest clips:\n" + t[["method", "clip", "found", "wrong_place", "missed", "false_alarm_no_ball", "recall",
                                 "precision", "f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
