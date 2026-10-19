"""Classify every ID switch of a run: cross-team, same-team, or lost track.

Knowing *which kind* of identity error dominates tells you which fix to try:

* cross-team  -> a player took the id of an OPPONENT. A team constraint
                 (two jersey colours) can forbid this outright.
* same-team   -> a player took the id of a TEAM-MATE. Needs fine-grained
                 appearance (ReID, jersey number).
* lost track  -> the player's track was cut and he came back with a new id
                 (or one of his own older ids). Needs better continuity:
                 motion, camera compensation, gates that do not over-reject.

Teams are recovered without labels: the median torso colour (Lab) of each
ground-truth player, sampled over the clip, is clustered with k-means (k=2).
The script prints the two groups so you can check they make sense.

Usage:
    python scripts/idsw_breakdown.py outputs/ablation/004a_court/2_plus_gmc \
        outputs/ablation/004a_court/3_plus_reid --set "data.only=[v_00HRwkvvjtQ_c001]"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hooptrack.config import load_config
from hooptrack.data.sportsmot import Sequence, list_sequences, load_mot_txt
from hooptrack.eval.metrics import accumulate


def player_teams(seq: Sequence, stride: int = 25) -> dict[int, int]:
    """Ground-truth id -> team (0/1) from jersey colour clustering."""
    gt = seq.load_gt()
    colours: dict[int, list[np.ndarray]] = {}
    for f in range(1, seq.length + 1, stride):
        if f not in gt:
            continue
        lab = cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2LAB)
        ids, boxes = gt[f]
        for pid, (x1, y1, x2, y2) in zip(ids, boxes.astype(int), strict=True):
            h, w = y2 - y1, x2 - x1
            # torso: 20-50 % of the height, central half of the width
            torso = lab[max(y1 + int(0.2 * h), 0): y1 + int(0.5 * h),
                        max(x1 + int(0.25 * w), 0): x1 + int(0.75 * w)]
            if torso.size:
                colours.setdefault(int(pid), []).append(np.median(torso.reshape(-1, 3), axis=0))
    ids = sorted(colours)
    X = np.array([np.median(colours[i], axis=0) for i in ids], dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, labels, _ = cv2.kmeans(X, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return {i: int(lbl) for i, lbl in zip(ids, labels.ravel(), strict=True)}


def breakdown(gt: dict, pred: dict, team: dict[int, int]) -> dict[str, int]:
    """Walk motmetrics events in time order and label each SWITCH."""
    events = accumulate(gt, pred).mot_events
    last_owner: dict[int, int] = {}          # predicted id -> last gt id it covered
    out = {"cross_team": 0, "same_team": 0, "lost_track": 0}
    for _, row in events.iterrows():
        if row.Type not in ("MATCH", "SWITCH"):
            continue
        gid, hid = int(row.OId), int(row.HId)
        if row.Type == "SWITCH":
            prev = last_owner.get(hid)
            if prev is None or prev == gid:
                out["lost_track"] += 1       # brand-new id, or one of his own old ids
            elif team.get(prev) != team.get(gid):
                out["cross_team"] += 1
            else:
                out["same_team"] += 1
        last_owner[hid] = gid
    out["total"] = sum(out.values())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run directories (with results/<seq>.txt)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], d["split"], d.get("sport"), d["splits_dir"])
    if d.get("only"):
        seqs = [s for s in seqs if s.name in set(d["only"])]

    for seq in seqs:
        team = player_teams(seq)
        groups = {t: [i for i, g in team.items() if g == t] for t in (0, 1)}
        print(f"## {seq.name}  teams: {groups}\n")
        print("| run | cross-team | same team | lost track | total |\n|---|---|---|---|---|")
        gt = seq.load_gt()
        for run in args.runs:
            res = run / "results" / f"{seq.name}.txt"
            if not res.is_file():
                continue
            b = breakdown(gt, load_mot_txt(res), team)
            print(f"| {run.name} | {b['cross_team']} | {b['same_team']} | {b['lost_track']} | {b['total']} |")
        print()


if __name__ == "__main__":
    main()
