"""The gate of 033: do enough pieces get a reliable jersey number, and would it block the wrong joins?

Reads a stitching run made with ``--jersey-rule --jersey-reader classifier`` (it saves ``<run>/<clip>/pieces.txt``,
the pieces before joining, and the reads of every piece in ``summary.json``) and, with the ground truth:

1. **coverage**: share of pieces (and of boxes) that get a number (023, Florence-2 zero-shot: 22 % of the pieces);
2. **consistency**: pairs of numbered pieces of the SAME player: same number? Pairs of numbered pieces of two
   different TEAM-MATES: same number? (that is an error: numbers are unique inside a team);
3. **use**: the joins of a reference run without the rule (``--reference``, 029 by default, same pieces): of the
   wrong joins, how many would the rule block (both sides numbered, different numbers), and of the right joins, how
   many would it block by mistake.

Each piece is the ground-truth player it covers most (box IoU >= 0.5); teams from jersey colour (``gt_teams``).

Usage (repo root):
    python scripts/jersey_coverage.py --config configs/player.yaml --run outputs/runs/033_jersey `
        --reference outputs/runs/029_siglip_ft
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import load_mot_txt
from hooptrack.eval.gt_teams import player_teams
from hooptrack.tracking.matching import iou_matrix


def piece_players(gt: dict, pieces: dict) -> tuple[dict[int, int], Counter]:
    """{piece: ground-truth player it covers most}, and the number of boxes of each piece."""
    votes: dict[int, Counter] = defaultdict(Counter)
    size = Counter()
    for f, (pids, pboxes) in pieces.items():
        for p in pids:
            size[int(p)] += 1
        if f not in gt or not len(gt[f][0]):
            continue
        gids, gboxes = gt[f]
        iou = iou_matrix(np.asarray(pboxes, float), np.asarray(gboxes, float))
        for i, p in enumerate(pids):
            j = int(iou[i].argmax())
            if iou[i, j] >= 0.5:
                votes[int(p)][int(gids[j])] += 1
    return {p: v.most_common(1)[0][0] for p, v in votes.items()}, size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run", type=Path, required=True, help="stitching run with --jersey-reader classifier")
    ap.add_argument("--reference", type=Path, default=Path("outputs/runs/029_siglip_ft"),
                    help="run without the rule, same pieces: its joins are checked against the numbers")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seqs = {s.name: s for s in _sequences(apply_overrides(cfg, ["data.only=null"]))}
    jersey = json.loads((args.run / "summary.json").read_text())["jersey"]
    ref_log = json.loads((args.reference / "summary.json").read_text())["merges"]
    rows, joins = [], Counter()
    same_player, mates = Counter(), Counter()
    for clip, js in jersey.items():
        seq = seqs[clip]
        gt, teams = seq.load_gt(), player_teams(seq)
        pieces = load_mot_txt(args.run / clip / "pieces.txt")
        who, size = piece_players(gt, pieces)
        nums = {int(k): v for k, v in js["numbers"].items()}
        rows.append({"clip": clip, "pieces": len(size), "numbered": len(nums),
                     "boxes": sum(size.values()), "boxes_numbered": sum(size[p] for p in nums)})
        # consistency between numbered pieces
        for a, b in itertools.combinations(sorted(nums), 2):
            if a not in who or b not in who:
                continue
            if who[a] == who[b]:
                same_player["same number" if nums[a] == nums[b] else "different numbers"] += 1
            elif teams.get(who[a], -1) == teams.get(who[b], -2):
                mates["same number (error)" if nums[a] == nums[b] else "different numbers"] += 1
        # the reference run's joins, judged by the ground truth and by the numbers
        for e in ref_log.get(clip, []):
            if not e.get("merged"):
                continue
            ga = Counter(who[m] for m in e["a"] if m in who).most_common(1)
            gb = Counter(who[m] for m in e["b"] if m in who).most_common(1)
            if not ga or not gb:
                continue
            right = ga[0][0] == gb[0][0]
            na, nb = {nums[m] for m in e["a"] if m in nums}, {nums[m] for m in e["b"] if m in nums}
            if na and nb:
                outcome = "both numbered, numbers agree" if na & nb else "both numbered, numbers differ -> blocked"
            else:
                outcome = "not both numbered"
            joins[("right join" if right else "WRONG join", outcome)] += 1

    df = pd.DataFrame(rows)
    tot = df[["pieces", "numbered", "boxes", "boxes_numbered"]].sum()
    print(f"\n{len(df)} clips | pieces with a number: {tot['numbered']} of {tot['pieces']} "
          f"({tot['numbered'] / tot['pieces']:.1%}; 023: 22 %) | boxes in numbered pieces: "
          f"{tot['boxes_numbered'] / tot['boxes']:.1%}")
    print("\nPer clip:\n" + df.assign(share=(df.numbered / df.pieces).round(2)).to_string(index=False))
    print("\nNumbered pieces of the SAME player:", dict(same_player))
    print("Numbered pieces of two different TEAM-MATES:", dict(mates))
    print(f"\nJoins of the reference run ({args.reference.name}):")
    for k in sorted(joins):
        print(f"  {k[0]:11s} | {k[1]:42s} {joins[k]}")


if __name__ == "__main__":
    main()
