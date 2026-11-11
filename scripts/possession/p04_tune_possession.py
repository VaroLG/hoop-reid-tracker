"""P01b · Two values of the possession rules chosen on the held-out TRAIN clips, then tested once (P04).

P01 says "air" too often (P01 README): short air stretches inside one player's possession (dribble bounces, the ball
hidden for a moment) and between two close players (the holder flipping between an attacker and his defender).
Two values decide how short is "short":

* ``max_drop``: air between two stretches of the SAME player, filled as his (P01: 3 frames);
* ``max_drop_switch``: air between two DIFFERENT players, filled as held, half each (P01: 0 = off).

1. **Choice on the 3 held-out train clips** (catches / releases marked by hand, ``p02_label_events.py --split train``;
   players = the ground-truth boxes, the ball = B05c tracks): every pair of the grid; the best **event F1** (balance
   of finding the marked catches / releases and not inventing others, fixed as the criterion before running) wins.
2. **Test, once**, on the 4 test clips with the player tracks of 038: P01 vs P01b, scored by ``p03_eval_possession``.

Usage (repo root):
    python scripts/possession/p04_tune_possession.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hota import result_file  # noqa: E402
from p03_eval_possession import changes, label_states, match  # noqa: E402

from hooptrack.ball.possession import PossessionConfig, possession  # noqa: E402
from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt  # noqa: E402

GRID_DROP = [3, 6, 10, 15]
GRID_SWITCH = [0, 3, 6, 10]


def pred_events(h: np.ndarray) -> list[tuple[int, str]]:
    out, prev = [], None
    for f, v in enumerate(h):
        if v == -2:
            continue
        if prev is not None and v != prev:
            if prev >= 0:
                out.append((f + 1, "release"))
            if v >= 0:
                out.append((f + 1, "catch"))
        prev = v
    return out


def evaluate(clips: dict, labels: pd.DataFrame, cfg: PossessionConfig, tol: int = 5) -> dict:
    agree, frames, n_gt, n_pred, n_found = 0, 0, 0, 0, 0
    for clip, (ball, players, n) in clips.items():
        h = possession(ball, players, n, cfg)
        gt = label_states(labels[labels["clip"] == clip], n)
        agree += int(((h >= 0).astype(int) == gt).sum())
        frames += n
        ge, pe = changes(gt), pred_events(h)
        found, _ = match(ge, pe, tol)
        n_gt, n_pred, n_found = n_gt + len(ge), n_pred + len(pe), n_found + found
    r, p = n_found / max(1, n_gt), n_found / max(1, n_pred)
    return {"frame_agreement": round(agree / frames, 3), "events_labelled": n_gt, "events_pred": n_pred,
            "event_recall": round(r, 3), "event_precision": round(p, 3),
            "event_f1": round(2 * r * p / (r + p), 3) if r + p else 0.0}


def load(split: str, tracks: Path, players: str, labels: pd.DataFrame, cfg) -> dict:
    seqs = {s.name: s for s in _sequences(apply_overrides(cfg, [f"data.split={split}", "data.only=null"]))}
    out = {}
    for clip in sorted(labels["clip"].unique()):
        t = pd.read_csv(tracks / f"{clip}.csv")
        ball = {int(r.frame): (r.cx, r.cy, r.d) for r in t.itertuples()}
        seq = seqs[clip]
        pl = seq.load_gt() if players == "gt" else load_mot_txt(result_file(Path(players), clip))
        out[clip] = (ball, pl, seq.length)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--train-labels", type=Path, default=Path("data/possession_labels/train_events.csv"))
    ap.add_argument("--test-labels", type=Path, default=Path("data/possession_labels/val_events.csv"))
    ap.add_argument("--train-tracks", type=Path, default=Path("outputs/ball/tracks_b05c_train"))
    ap.add_argument("--test-tracks", type=Path, default=Path("outputs/ball/tracks_b05c"))
    ap.add_argument("--test-players", default="outputs/runs/038_continuity")
    ap.add_argument("--out", type=Path, default=Path("outputs/possession/p04"))
    args = ap.parse_args()
    cfg = load_config(args.config, [])
    args.out.mkdir(parents=True, exist_ok=True)

    tl = pd.read_csv(args.train_labels)
    train = load("train", args.train_tracks, "gt", tl, cfg)
    grid = [{"max_drop": d, "max_drop_switch": s,
             **evaluate(train, tl, PossessionConfig(max_drop=d, max_drop_switch=s))}
            for d, s in itertools.product(GRID_DROP, GRID_SWITCH)]
    g = pd.DataFrame(grid)
    g.to_csv(args.out / "grid_train.csv", index=False)
    print("Grid on the held-out train clips (players: ground truth):\n" + g.to_string(index=False))
    best = g.sort_values("event_f1", ascending=False, kind="stable").iloc[0]
    chosen = PossessionConfig(max_drop=int(best["max_drop"]), max_drop_switch=int(best["max_drop_switch"]))
    print(f"\nchosen on train: max_drop={chosen.max_drop}, max_drop_switch={chosen.max_drop_switch}")

    vl = pd.read_csv(args.test_labels)
    test = load("val", args.test_tracks, args.test_players, vl, cfg)
    rows = []
    for name, c in {"P01": PossessionConfig(), "P01b": chosen}.items():
        for clip in list(test) + ["ALL"]:
            sub = test if clip == "ALL" else {clip: test[clip]}
            rows.append({"method": name, "clip": clip, **evaluate(sub, vl, c)})
    t = pd.DataFrame(rows)
    always = {clip: round(label_states(vl[vl["clip"] == clip], n).mean(), 3) for clip, (_, _, n) in test.items()}
    t["always_held"] = t["clip"].map(always)
    t.to_csv(args.out / "test.csv", index=False)
    print("\nTest clips (players: 038 tracks):\n" + t.to_string(index=False))


if __name__ == "__main__":
    main()
