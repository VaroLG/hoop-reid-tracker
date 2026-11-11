"""042 diagnosis · Where do the wrong identities of a closed-set run come from?

Every predicted box matched to a ground-truth player (IoU >= 0.5, Hungarian per frame) is **right** when its final
id's owner (the ground-truth player holding most of that id's boxes) is that player, **wrong** otherwise. Each wrong
box is traced back to its piece (the online tracklet cut at crossings, ``split_at_crossings`` as in 038):

* ``impure piece``            the piece itself contains two players (a swap the crossing cut did not separate) and
                               the box belongs to its minority player: an ONLINE tracker error;
* ``wrong label: team-mate``  the piece is pure but was given to a team-mate: an ASSIGNMENT error inside a kit;
* ``wrong label: opponent``   the piece is pure but was given to a player of the other kit.

Kits of the ground-truth players: ``hooptrack.eval.gt_teams`` (torso colour), as in ``switch_report.py``.
Evaluation only (uses ground truth); it changes nothing in the tracker.

Usage (repo root):
    python scripts/identity_error_breakdown.py --runs 038=outputs/runs/038_continuity \\
        --online outputs/runs/011b_no_referee_filter/segmentation --out experiments/042_soft_team_cost/errors.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.eval.gt_teams import player_teams
from hooptrack.tracking.stitch import split_at_crossings


def _load(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, header=None, usecols=range(6), names=["f", "id", "x", "y", "w", "h"])


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax2, ay2, bx2, by2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2) - np.maximum(a[:, None, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2) - np.maximum(a[:, None, 1], b[:, 1]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:].prod(1)[:, None] + b[:, 2:].prod(1) - inter)


def _results(run: Path, clip: str) -> Path:
    p = run / clip / "results" / f"{clip}.txt"
    return p if p.is_file() else run / "results" / f"{clip}.txt"


def breakdown(seq, online: Path, final: Path) -> pd.DataFrame:
    on, fin, gt = _load(online), _load(final), _load(seq.root / "gt" / "gt.txt")
    pred = {int(f): (g.id.to_numpy(), np.c_[g.x, g.y, g.x + g.w, g.y + g.h]) for f, g in on.groupby("f")}
    seg = split_at_crossings(pred, 0.2)
    on["piece"] = [seg[(int(f), int(t))] for f, t in zip(on.f, on.id, strict=True)]
    on["g"] = -1
    for f, g in on.groupby("f"):
        G = gt[gt.f == f]
        if not len(G):
            continue
        M = _iou(g[["x", "y", "w", "h"]].to_numpy(), G[["x", "y", "w", "h"]].to_numpy())
        r, c = linear_sum_assignment(-M)
        ok = M[r, c] >= 0.5
        on.loc[g.index[r[ok]], "g"] = G.id.to_numpy()[c[ok]]
    key = lambda d: list(zip(d.f, d.x.round(1), d.y.round(1), strict=True))  # noqa: E731
    final_id = dict(zip(key(fin), fin.id, strict=True))
    on["final"] = [final_id.get(k, -1) for k in key(on)]
    on["piece_len"] = on.piece.map(on.groupby("piece").size())
    m = on[on.g >= 0].copy()
    major = lambda s: s.value_counts().index[0]  # noqa: E731
    m["owner"] = m.final.map(m.groupby("final").g.agg(major))
    m["piece_major"] = m.piece.map(m.groupby("piece").g.agg(major))
    kit = player_teams(seq)
    m["kind"] = np.where(m.g == m.owner, "right", np.where(
        m.g != m.piece_major, "impure piece",
        np.where([kit.get(a, -1) == kit.get(b, -2) for a, b in zip(m.g, m.owner, strict=True)],
                 "wrong label: team-mate", "wrong label: opponent")))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=DIR of stitched runs")
    ap.add_argument("--online", type=Path, required=True, help="the online run the pieces come from")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=val", "data.only=null"]))}
    rows = []
    for spec in args.runs:
        name, run = spec.split("=", 1)
        run = Path(run)
        clips = args.clips or sorted(p.name for p in run.iterdir() if p.is_dir() and p.name in seqs)
        for clip in clips:
            m = breakdown(seqs[clip], _results(args.online, clip), _results(run, clip))
            m["len"] = pd.cut(m.piece_len, [0, 25, 75, 250, 10**6], labels=["<1s", "1-3s", "3-10s", ">10s"])
            for (kind, ln), g in m.groupby(["kind", "len"], observed=True):
                rows.append({"run": name, "clip": clip, "kind": kind, "piece_length": ln, "boxes": len(g)})
            print(f"{name} {clip}: {(m.kind != 'right').mean():.1%} of the matched boxes with a wrong identity")
    t = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(args.out, index=False)
    tot = t.groupby(["run", "kind"]).boxes.sum().unstack(0)
    print("\nboxes per kind:\n" + tot.to_string())
    print("\nwrong boxes by piece length:\n" + t[t.kind != "right"].pivot_table(
        index=["run", "kind"], columns="piece_length", values="boxes", aggfunc="sum", observed=True).to_string())


if __name__ == "__main__":
    main()
