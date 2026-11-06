"""P05 · Possession with a learned per-frame score + a two-state temporal model, trained on TRAIN, tested once.

See ``hooptrack.ball.possession_temporal`` for the model. Inputs:

* **train**: the 3 held-out train clips, hand-marked catches/releases (``data/possession_labels/train_events.csv``),
  ball = B08 (fine-tuned WASB + B05's tracker) on those clips, players = ground-truth boxes (no tracker run there);
* **test**: the 4 test clips, ball = B08 (``outputs/ball/tracks_b08``), players = 038 (as P04).

Scored exactly as P03 (frames right, catches/releases within 5 frames). The held frames take the closest player
within reach as holder (P01's rule 3) so the output has the same format as P01/P04; only held/air is measured.

Usage (repo root):
    python scripts/possession/p07_temporal.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_hota import result_file  # noqa: E402
from p03_eval_possession import changes, label_states, match  # noqa: E402

from hooptrack.ball.possession import AIR, PossessionConfig, raw_holder  # noqa: E402
from hooptrack.ball.possession_temporal import FEATURES, frame_features, viterbi  # noqa: E402
from hooptrack.ball.track import BallTrackConfig, track_ball  # noqa: E402
from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt  # noqa: E402


def ball_from_dets(path: Path) -> dict[int, tuple]:
    c = pd.read_csv(path)
    dets = {int(f): np.c_[g[["cx", "cy", "d", "conf"]].to_numpy(), np.zeros(len(g))] for f, g in c.groupby("frame")}
    return track_ball(dets, BallTrackConfig())         # B08's tracker as is


def ball_from_track(path: Path) -> dict[int, tuple]:
    return {int(r.frame): (r.cx, r.cy, r.d, r.source) for r in pd.read_csv(path).itertuples()}


def score_clip(gt: np.ndarray, pred: np.ndarray, tol: int = 5) -> dict:
    ge, pe = changes(gt), changes(pred)
    n, errs = match(ge, pe, tol)
    return {"frames": len(gt), "frame_agreement": float((gt == pred).mean()), "always_held": float(gt.mean()),
            "events_labelled": len(ge), "events_pred": len(pe), "events_found": n,
            "median_timing_frames": float(np.median(errs)) if errs else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--train-dets", type=Path, default=Path("outputs/ball/dets_wasb_ft_train"))
    ap.add_argument("--test-tracks", type=Path, default=Path("outputs/ball/tracks_b08"))
    ap.add_argument("--players", type=Path, default=Path("outputs/runs/038_continuity"))
    ap.add_argument("--out", type=Path, default=Path("outputs/possession/p07"))
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    cfg = load_config(args.config, [])
    seqs = {s.name: s for split in ("train", "val") for s in
            _sequences(apply_overrides(cfg, [f"data.split={split}", "data.only=null"]))}
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- train
    lab = pd.read_csv("data/possession_labels/train_events.csv")
    Xs, ys, n_events, n_frames = [], [], 0, 0
    for clip, marks in lab.groupby("clip"):
        seq = seqs[clip]
        X = frame_features(ball_from_dets(args.train_dets / f"{clip}.csv"), seq.load_gt(), seq.length)
        y = label_states(marks, seq.length)
        Xs.append(X)
        ys.append(y)
        n_events += len(changes(y))
        n_frames += len(y) - 1
    X, y = np.concatenate(Xs), np.concatenate(ys)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y)
    prior, p_switch = float(y.mean()), n_events / n_frames
    coef = dict(zip(FEATURES, np.round(model[-1].coef_[0], 2), strict=True))
    print(f"train: {len(y)} frames, held {prior:.0%}, {n_events} changes -> p_switch {p_switch:.4f} per frame")
    print(f"logistic weights (standardised): {coef}")

    # ---- test, once
    lab = pd.read_csv("data/possession_labels/val_events.csv")
    rows = []
    for clip, marks in lab.groupby("clip"):
        seq = seqs[clip]
        ball = ball_from_track(args.test_tracks / f"{clip}.csv")
        players = load_mot_txt(result_file(args.players, clip))
        X = frame_features(ball, players, seq.length)
        p = model.predict_proba(X)[:, 1]
        gt = label_states(marks, seq.length)
        for name, pred in [("logistic, per frame (p > 0.5)", (p > 0.5).astype(int)),
                           ("logistic + temporal model (P05)", viterbi(p, prior, p_switch))]:
            rows.append({"method": name, "clip": clip, **score_clip(gt, pred)})
        st = viterbi(p, prior, p_switch)               # holder ids for the output files (P01's format)
        pc = PossessionConfig()
        holder = np.full(seq.length, AIR)
        for k in np.flatnonzero(st):
            f = k + 1
            if f in ball:
                ids, boxes = players.get(f, (np.zeros(0, int), np.zeros((0, 4))))
                holder[k] = raw_holder(ball[f][:3], ids, boxes, pc)
        pd.DataFrame({"frame": range(1, seq.length + 1), "held": st, "p_held": p.round(3), "holder": holder}
                     ).to_csv(args.out / f"{clip}_holder.csv", index=False)
    df = pd.DataFrame(rows)
    tot = []
    for name, g in df.groupby("method", sort=False):
        t = {"method": name, "clip": "ALL", "frames": g.frames.sum(),
             "frame_agreement": (g.frame_agreement * g.frames).sum() / g.frames.sum(),
             "always_held": (g.always_held * g.frames).sum() / g.frames.sum(),
             "events_labelled": g.events_labelled.sum(), "events_pred": g.events_pred.sum(),
             "events_found": g.events_found.sum(), "median_timing_frames": np.nan}
        tot.append(t)
    df = pd.concat([df, pd.DataFrame(tot)], ignore_index=True)
    df["event_recall"] = df.events_found / df.events_labelled
    df["event_precision"] = df.events_found / df.events_pred.clip(lower=1)
    df["event_f1"] = 2 * df.event_recall * df.event_precision / (df.event_recall + df.event_precision).clip(lower=1e-9)
    df = df.round(3)
    df.to_csv(args.out / "per_clip.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
