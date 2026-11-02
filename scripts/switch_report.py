"""Identity switches of finished runs, by kind and per minute, on every clip (029).

The README tables give totals (IDF1, HOTA, ID switches); a side-by-side video shows one clip. Neither says WHAT
kind of identity errors are left, nor whether a clip is representative. This report counts, for every clip and
run, the identity switches against the ground truth by kind (CLEAR MOT matching, ``render_compare.clear_matches``:
the total per run is the ``IDSW`` that ``scripts/eval_hota.py`` reports, which is the check that it counts right):

* ``new``      the player gets an id never used before: a fragment;
* ``mate``     he gets an id last carried by a TEAM-MATE: a swap inside one kit;
* ``opponent`` he gets an id last carried by a player of the OTHER kit: a cross-team swap;
* ``unknown``  the previous owner's kit is unknown (ground-truth kits come from jersey colour, ``gt_teams``).

Counts are also given **per minute** (clips last 20-60 s, so raw counts favour short clips).

``at_crossing`` (030): was the player overlapping another ground-truth player (one box covering >= 20 % of the
other, the tracker's "occluded crop" rule) at some point in the ``--crossing-window`` seconds before the switch?
That separates the switches an identity check at crossing exits could fix from the others (fragments after
leaving the frame or a long occlusion, detector misses, ...).

``--online NAME``: for runs built on that online run (stitching), every cross-team switch is traced: was the same
switch (same frame, same ground-truth player) already in the online run (a swap the team cut did not remove), or is
it new (created by a merge of the stitching)?

Usage (repo root):
    python scripts/switch_report.py --config configs/player.yaml --online 011b --runs `
        011b=outputs/runs/011b_no_referee_filter/segmentation 020=outputs/runs/020_siglip_k2_15 `
        029=outputs/runs/029_siglip_ft --out outputs/runs/029_switches.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hota import result_file  # noqa: E402
from render_compare import KINDS, load_mot, switch_records  # noqa: E402

from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.tracking.matching import coverage_matrix  # noqa: E402


def overlapping_frames(gt: dict, min_coverage: float = 0.2) -> dict[int, set[int]]:
    """{ground-truth id: frames where his box covers / is covered by another player's box >= min_coverage}."""
    out: dict[int, set[int]] = {}
    for f, (ids, boxes) in gt.items():
        if len(ids) < 2:
            continue
        cov = coverage_matrix(np.asarray(boxes, float))
        hit = (cov >= min_coverage) | (min_coverage <= cov.T)
        for k in np.flatnonzero(hit.any(axis=1)):
            out.setdefault(int(ids[k]), set()).add(int(f))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--runs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--online", default=None, help="name of the online run the others are built on")
    ap.add_argument("--out", type=Path, default=None, help="optional CSV, one row per switch")
    ap.add_argument("--crossing-window", type=float, default=1.0,
                    help="seconds before a switch in which an overlap makes it an at-crossing switch")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seqs = _sequences(apply_overrides(cfg, ["data.only=null"]))
    runs = dict(r.split("=", 1) for r in args.runs)
    rows, per_clip = [], []
    for s in seqs:
        files = {n: result_file(Path(p), s.name) for n, p in runs.items()}
        if any(f is None for f in files.values()):
            continue
        gt, teams, minutes = s.load_gt(), player_teams(s), s.length / s.fps / 60
        overlaps, window = overlapping_frames(gt), int(round(args.crossing_window * s.fps))
        online_keys = None
        for name, f in files.items():
            recs = switch_records(load_mot(f), gt, s.length, teams)
            if name == args.online:
                online_keys = {(r["frame"], r["gt_id"]) for r in recs}
            for r in recs:
                ov = overlaps.get(r["gt_id"], set())
                r["at_crossing"] = any(f in ov for f in range(r["frame"] - window, r["frame"] + 1))
                rows.append({"clip": s.name, "run": name, **r})
            counts = {k: sum(r["kind"] == k for r in recs) for k in KINDS}
            per_clip.append({"clip": s.name, "run": name, "minutes": round(minutes, 2), **counts,
                             "total": len(recs), "per_min": round(len(recs) / minutes, 1),
                             "opponent_per_min": round(counts["opponent"] / minutes, 1)})
        if online_keys is not None:                    # trace cross-team switches back to the online run
            for r in rows:
                if r["clip"] == s.name and r["run"] != args.online and r["kind"] == "opponent":
                    r["origin"] = "online" if (r["frame"], r["gt_id"]) in online_keys else "stitching"
    if not per_clip:
        raise SystemExit("no clip present in every run")
    pc = pd.DataFrame(per_clip)
    tot = pc.groupby("run", sort=False)[["minutes", *KINDS, "total"]].sum()
    for col in [*KINDS, "total"]:
        tot[f"{col}/min"] = (tot[col] / tot["minutes"]).round(1)
    print(f"\n{pc['clip'].nunique()} clips, {tot['minutes'].iloc[0]:.1f} minutes in total\n")
    print("All clips (counts, then per minute):\n" + tot.round(1).to_string())
    print("\nPer clip, switches per minute:\n"
          + pc.pivot(index="clip", columns="run", values="per_min")[list(runs)].to_string())
    print("\nPer clip, cross-team switches (count):\n"
          + pc.pivot(index="clip", columns="run", values="opponent")[list(runs)].to_string())
    df = pd.DataFrame(rows)
    print("\nSwitches at a crossing (overlap in the last "
          f"{args.crossing_window:g} s) or elsewhere, by kind:\n"
          + df.groupby(["run", "at_crossing", "kind"]).size().unstack(fill_value=0).to_string())
    if args.online and "origin" in df:
        o = df[df["kind"] == "opponent"].dropna(subset=["origin"])
        print("\nCross-team switches of the stitched runs: already in the online run, or from the stitching:\n"
              + o.groupby(["run", "origin"]).size().unstack(fill_value=0).to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        pc.to_csv(args.out.with_name(args.out.stem + "_per_clip.csv"), index=False)
        print(f"\nSaved {args.out} and {args.out.with_name(args.out.stem + '_per_clip.csv')}")


if __name__ == "__main__":
    main()
