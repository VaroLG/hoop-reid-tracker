"""048 · An id that leaves a body and comes back: show (and output) the body's id all the time.

Why
---
In the demo video an id often leaves a player for a second in a crowd and comes back. 047 looked for that pattern in
the online tracklets and found almost nothing: the pattern lives in the FINAL output, on boxes that follow one body.
This post-process works only on a finished run (no ground truth):

1. **Bodies**: the boxes of consecutive frames are linked one to one when they overlap a lot (Hungarian on IoU,
   pairs below ``--min-iou`` refused). A body is a chain of such links, whatever ids the tracker gave its boxes.
   In a tight crowd the overlap gets ambiguous and the chain simply ends there.
2. **Excursions**: along a body, a run of frames with id Q between frames with the same id P on both sides, at most
   ``--max-seconds`` long, is an excursion P -> Q -> P.
3. It goes back to P if no other box shows P in those frames; if another box shows P only because of the mirror
   excursion (a body going Q -> P -> Q at the same time), both are swapped back. Otherwise it is left alone (two
   boxes with one id in a frame are never created).

The result is a normal run folder: it is evaluated like any other, so the metrics describe exactly what the video
shows. ``--max-seconds`` = 2 is fixed before running (most excursions measured in 047 last < 2 s; that measurement was
made on val, said here).

Usage (repo root):
    python scripts/smooth_body_labels.py --run outputs/runs/038_continuity --out outputs/runs/048_body_smooth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from hooptrack.data.sportsmot import load_mot_txt, write_mot_txt
from hooptrack.tracking.matching import iou_matrix


def bodies(pred: dict, min_iou: float, max_gap: int = 5) -> dict[tuple[int, int], int]:
    """{(frame, box index): body id}: one-to-one links on IoU with the last box of every body seen in the last
    ``max_gap`` frames (a box missed for a frame or two does not end its body)."""
    body: dict[tuple[int, int], int] = {}
    last: dict[int, tuple[int, np.ndarray]] = {}          # body -> (frame, box)
    nxt = 0
    for f in sorted(pred):
        _, boxes = pred[f]
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        alive = [b for b, (g, _) in last.items() if 0 < f - g <= max_gap]
        assigned = {}
        if alive and len(boxes):
            iou = iou_matrix(np.array([last[b][1] for b in alive]), boxes)
            r, c = linear_sum_assignment(-iou)
            for a, k in zip(r, c, strict=True):
                if iou[a, k] >= min_iou:
                    assigned[k] = alive[a]
        for k in range(len(boxes)):
            if k not in assigned:
                assigned[k], nxt = nxt, nxt + 1
            body[(f, k)] = assigned[k]
            last[assigned[k]] = (f, boxes[k])
    return body


def smooth(pred: dict, min_iou: float, max_len: int) -> tuple[dict, int, int]:
    """-> (relabelled pred, excursions sent back alone, excursions swapped back in pairs)."""
    body = bodies(pred, min_iou)
    label = {(f, k): int(pred[f][0][k]) for f in pred for k in range(len(pred[f][0]))}
    seq: dict[int, list[tuple[int, int]]] = {}
    for (f, k), b in body.items():
        seq.setdefault(b, []).append((f, k))
    exc = []                                       # (keys of the run, Q now, P around, frames)
    for keys in seq.values():
        keys.sort()
        labs = [label[x] for x in keys]
        runs, s = [], 0
        for i in range(1, len(labs) + 1):
            if i == len(labs) or labs[i] != labs[s]:
                runs.append((s, i))
                s = i
        for j in range(1, len(runs) - 1):
            (a, b), (pa, _) = runs[j], runs[j - 1]
            if labs[pa] == labs[runs[j + 1][0]] != labs[a] and b - a <= max_len:
                run = keys[a:b]
                exc.append((run, labs[a], labs[pa], {f for f, _ in run}))
    by_frame_label: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for x, l_ in label.items():
        by_frame_label.setdefault((x[0], l_), []).append(x)
    alone = pairs = 0
    done: set[int] = set()
    new = dict(label)
    for i, (run, q, p, fr) in enumerate(exc):
        if i in done:
            continue
        mine = set(run)
        busy = {x for f in fr for x in by_frame_label.get((f, p), []) if x not in mine}
        if not busy:
            for x in run:
                new[x] = p
            done.add(i)
            alone += 1
            continue
        for j, (run2, q2, p2, fr2) in enumerate(exc):
            if j in done or j == i or (q2, p2) != (p, q) or not busy <= set(run2):
                continue
            busy2 = {x for f in fr2 for x in by_frame_label.get((f, q), []) if x not in set(run2) and x not in mine}
            if busy2:
                continue
            for x in run:
                new[x] = p
            for x in run2:
                new[x] = q
            done |= {i, j}
            pairs += 1
            break
    out = {f: (np.array([new[(f, k)] for k in range(len(ids))]), boxes) for f, (ids, boxes) in pred.items()}
    return out, alone, pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-iou", type=float, default=0.5)
    ap.add_argument("--max-seconds", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    for d in sorted(p for p in args.run.iterdir() if (p / "results").is_dir()):
        clip = d.name
        pred = load_mot_txt(d / "results" / f"{clip}.txt")
        out, alone, pairs = smooth(pred, args.min_iou, int(round(args.max_seconds * args.fps)))
        rows = [(f, int(i), b, 1.0) for f, (ids, bs) in sorted(out.items()) for i, b in zip(ids, bs, strict=True)]
        write_mot_txt(args.out / clip / "results" / f"{clip}.txt", rows)
        print(f"{clip}: {alone} excursions sent back, {pairs} swapped pairs")


if __name__ == "__main__":
    main()
