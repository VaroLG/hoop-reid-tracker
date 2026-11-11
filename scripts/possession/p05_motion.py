"""P02 · Air decided by the ball's own motion; and how much better possession gets with a perfect ball (P05).

1. **Choice on the 3 held-out train clips** (tracked ball B05c, ground-truth player boxes, hand-marked changes):
   ``v_air`` (ball diameters per frame, camera motion removed) x ``min_air`` (frames); the best event F1 wins
   (criterion fixed before running).
2. **Test, once, on the 4 test clips** (player tracks of 038), P01 and P02, each with two balls:

   * the **tracked ball** (B05c: 78 % of the visible balls found, 73 % precision): what the system really does;
   * an **"oracle" ball**: the hand clicks of B02 (one frame in ten), interpolated in between when both ends are
     visible. Not available on a new video; it tells how much of the possession error comes from the ball's position
     and how much from the possession rules themselves.

Usage (repo root):
    python scripts/possession/p05_motion.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p03_eval_possession import changes, label_states, match  # noqa: E402
from p04_tune_possession import load, pred_events  # noqa: E402

from hooptrack.ball.labels import ball_diameter, load_labels  # noqa: E402
from hooptrack.ball.possession import MotionPossessionConfig, PossessionConfig, possession, possession_motion  # noqa: E402
from hooptrack.config import load_config  # noqa: E402

GRID_V = [1.0, 1.5, 2.0, 2.5]
GRID_MIN = [2, 4, 6]


def oracle_ball(clicks: pd.DataFrame, gt: dict) -> dict[int, tuple[float, float, float]]:
    """Hand clicks interpolated between two consecutive VISIBLE clicks; no ball elsewhere."""
    c = clicks.sort_values("frame")
    pts = [(int(r["frame"]), r["x"], r["y"], ball_diameter(r["x"], r["y"], gt.get(int(r["frame"]),
                                                                                  (None, np.zeros((0, 4))))[1]))
           if r["visible"] else (int(r["frame"]), None, None, None) for _, r in c.iterrows()]
    out = {}
    for (f0, x0, y0, d0), (f1, x1, y1, d1) in zip(pts, pts[1:], strict=False):
        if x0 is None:
            continue
        out[f0] = (x0, y0, d0)
        if x1 is None:
            continue
        for f in range(f0 + 1, f1):
            w = (f - f0) / (f1 - f0)
            out[f] = (x0 + w * (x1 - x0), y0 + w * (y1 - y0), d0 + w * (d1 - d0))
    if pts and pts[-1][1] is not None:
        out[pts[-1][0]] = pts[-1][1:]
    return out


def evaluate(clips: dict, labels: pd.DataFrame, fn) -> dict:
    agree, frames, n_gt, n_pred, n_found = 0, 0, 0, 0, 0
    for clip, (ball, players, n) in clips.items():
        h = fn(ball, players, n)
        gt = label_states(labels[labels["clip"] == clip], n)
        agree += int(((h >= 0).astype(int) == gt).sum())
        frames += n
        ge, pe = changes(gt), pred_events(h)
        found, _ = match(ge, pe, 5)
        n_gt, n_pred, n_found = n_gt + len(ge), n_pred + len(pe), n_found + found
    r, p = n_found / max(1, n_gt), n_found / max(1, n_pred)
    return {"frame_agreement": round(agree / frames, 3), "events_labelled": n_gt, "events_pred": n_pred,
            "event_recall": round(r, 3), "event_precision": round(p, 3),
            "event_f1": round(2 * r * p / (r + p), 3) if r + p else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--train-labels", type=Path, default=Path("data/possession_labels/train_events.csv"))
    ap.add_argument("--test-labels", type=Path, default=Path("data/possession_labels/val_events.csv"))
    ap.add_argument("--test-clicks", type=Path, default=Path("data/ball_labels/val.csv"))
    ap.add_argument("--train-tracks", type=Path, default=Path("outputs/ball/tracks_b05c_train"))
    ap.add_argument("--test-tracks", type=Path, default=Path("outputs/ball/tracks_b05c"))
    ap.add_argument("--test-players", default="outputs/runs/038_continuity")
    ap.add_argument("--out", type=Path, default=Path("outputs/possession/p05"))
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    args.out.mkdir(parents=True, exist_ok=True)

    tl = pd.read_csv(args.train_labels)
    train = load("train", args.train_tracks, "gt", tl, cfg)
    grid = []
    for v, m in itertools.product(GRID_V, GRID_MIN):
        c = MotionPossessionConfig(v_air=v, min_air=m)
        run = lambda b, p, n, c=c: possession_motion(b, p, n, c)  # noqa: E731
        grid.append({"v_air": v, "min_air": m, **evaluate(train, tl, run)})
    g = pd.DataFrame(grid)
    g.to_csv(args.out / "grid_train.csv", index=False)
    print("Grid on the held-out train clips:\n" + g.to_string(index=False))
    best = g.sort_values("event_f1", ascending=False, kind="stable").iloc[0]
    chosen = MotionPossessionConfig(v_air=float(best["v_air"]), min_air=int(best["min_air"]))
    print(f"\nchosen on train: v_air={chosen.v_air}, min_air={chosen.min_air}")

    vl = pd.read_csv(args.test_labels)
    tracked = load("val", args.test_tracks, args.test_players, vl, cfg)
    clicks = load_labels(args.test_clicks)
    from hooptrack.cli import _sequences
    from hooptrack.config import apply_overrides

    seqs = {s.name: s for s in _sequences(apply_overrides(cfg, ["data.split=val", "data.only=null"]))}
    oracle = {c: (oracle_ball(clicks[clicks["clip"] == c], seqs[c].load_gt()), pl, n)
              for c, (_, pl, n) in tracked.items()}
    methods = {"P01 rules": lambda b, p, n: possession(b, p, n, PossessionConfig()),
               "P02 ball motion": lambda b, p, n: possession_motion(b, p, n, chosen)}
    rows = []
    for (mname, fn), (bname, data) in itertools.product(methods.items(), [("tracked ball", tracked),
                                                                         ("oracle ball", oracle)]):
        for clip in list(data) + ["ALL"]:
            sub = data if clip == "ALL" else {clip: data[clip]}
            rows.append({"method": mname, "ball": bname, "clip": clip, **evaluate(sub, vl, fn)})
    t = pd.DataFrame(rows)
    t.to_csv(args.out / "test.csv", index=False)
    always = np.mean(np.concatenate([label_states(vl[vl["clip"] == c], n) for c, (_, _, n) in tracked.items()]))
    print(f"\nTest clips (players: 038 tracks); 'always held' would give {always:.3f} of the frames right\n"
          + t.to_string(index=False))


if __name__ == "__main__":
    main()
