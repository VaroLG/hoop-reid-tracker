"""P03 · Possession (P01) against the hand-marked catches and releases of the test clips (P02).

The labels give, for every frame, a state: **held** (after a ``catch``, a player has the ball) or **air** (after a
``release``: in the air or loose). Each frame takes the state of the last mark before it (dense marks and marks only
at the changes give the same states); the first mark of a clip only sets the starting state. The **events** are the
changes of state.

Measured (P01's frames: holder >= 0 = held, -1 = air, -2 = no ball):

1. **Frames**: share of frames where P01's state is the labelled one (frames without a ball counted as air, and
   also reported apart); and the trivial reference "always held".
2. **Events**: a labelled catch (release) is found if P01 has a catch (release) within ``--tol`` frames (fixed:
   5 = 0.2 s), one to one; recall, precision, median timing error. A direct player -> player change counts as a
   release and a catch at the same frame.

Who has the ball is not labelled; only held / air is measured here.

Usage (repo root):
    python scripts/possession/p03_eval_possession.py --pred outputs/possession/p01 --name p01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def label_states(marks: pd.DataFrame, n: int) -> np.ndarray:
    """1 = held, 0 = air, per frame (index = frame - 1), from the marks of one clip."""
    s = np.zeros(n, int)
    m = marks.sort_values("frame")
    cur, k = (1 if m.iloc[0]["event"] == "catch" else 0), 0
    rows = list(m.itertuples())
    for f in range(1, n + 1):
        while k < len(rows) and rows[k].frame <= f:
            cur = 1 if rows[k].event == "catch" else 0
            k += 1
        s[f - 1] = cur
    return s


def changes(state: np.ndarray) -> list[tuple[int, str]]:
    return [(int(f) + 1, "catch" if state[f] else "release") for f in np.flatnonzero(np.diff(state)) + 1]


def match(gt: list[tuple[int, str]], pred: list[tuple[int, str]], tol: int) -> tuple[int, list[int]]:
    """Greedy one-to-one matching of same-kind events within ``tol`` frames -> (matches, timing errors)."""
    pairs = sorted((abs(g[0] - p[0]), i, j) for i, g in enumerate(gt) for j, p in enumerate(pred)
                   if g[1] == p[1] and abs(g[0] - p[0]) <= tol)
    used_g, used_p, errs = set(), set(), []
    for d, i, j in pairs:
        if i not in used_g and j not in used_p:
            used_g.add(i)
            used_p.add(j)
            errs.append(d)
    return len(errs), errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, default=Path("data/possession_labels/val_events.csv"))
    ap.add_argument("--pred", type=Path, default=Path("outputs/possession/p01"))
    ap.add_argument("--name", default="p01")
    ap.add_argument("--tol", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("outputs/possession/p03"))
    args = ap.parse_args()
    labels = pd.read_csv(args.labels)
    rows = []
    for clip, marks in labels.groupby("clip"):
        src = args.pred / f"{clip}_holder.csv"
        if not src.is_file():
            continue
        h = pd.read_csv(src)["holder"].to_numpy()
        gt = label_states(marks, len(h))
        pred = (h >= 0).astype(int)
        ball = h != -2
        ge, pe = changes(gt), []
        prev = None
        for f, v in enumerate(h):                       # P01's events, with player -> player as release + catch
            if v == -2:
                continue
            if prev is not None and v != prev:
                if prev >= 0:
                    pe.append((f + 1, "release"))
                if v >= 0:
                    pe.append((f + 1, "catch"))
            prev = v
        n, errs = match(ge, pe, args.tol)
        rows.append({"clip": clip, "frames": len(h), "labelled_held": round(gt.mean(), 3),
                     "frame_agreement": round((pred == gt).mean(), 3),
                     "agreement_with_ball": round((pred[ball] == gt[ball]).mean(), 3),
                     "always_held": round(gt.mean(), 3),
                     "events_labelled": len(ge), "events_pred": len(pe), "events_found": n,
                     "event_recall": round(n / max(1, len(ge)), 3), "event_precision": round(n / max(1, len(pe)), 3),
                     "median_timing_frames": float(np.median(errs)) if errs else np.nan})
    df = pd.DataFrame(rows)
    tot = {"clip": "ALL", "frames": df.frames.sum(),
           "labelled_held": round((df.labelled_held * df.frames).sum() / df.frames.sum(), 3),
           "frame_agreement": round((df.frame_agreement * df.frames).sum() / df.frames.sum(), 3),
           "agreement_with_ball": np.nan, "always_held": round((df.always_held * df.frames).sum() / df.frames.sum(), 3),
           "events_labelled": df.events_labelled.sum(), "events_pred": df.events_pred.sum(),
           "events_found": df.events_found.sum()}
    tot["event_recall"] = round(tot["events_found"] / tot["events_labelled"], 3)
    tot["event_precision"] = round(tot["events_found"] / tot["events_pred"], 3)
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_clip.csv", index=False)
    print(f"{args.name} (tolerance {args.tol} frames)\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
