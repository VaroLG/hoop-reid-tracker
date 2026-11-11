"""Would the court position tell two crossing players apart? Diagnosis before any tracker change (034).

Why
---
The swaps that are left (032) happen when two players overlap in the IMAGE. Overlapping in the image does not mean
being in the same place: the camera looks at the court from the side and above, so a player 2 m further back can be
hidden behind another. On the court plan (metres, C08) the two would still be 2 m apart, and a tracker that
associates in metres could keep them apart. But that only helps if (a) the players are really apart on the court
when the swaps happen, and (b) their position can be measured right there, while one hides the other (the
position is the bottom centre of the box, the "feet": if the feet of the player behind are hidden, his box is cut
and its bottom is not where he stands).

This script measures both on the ground truth, with no change to the tracker:

1. **Crossing episodes.** Two ground-truth players whose boxes overlap (one covers >= 20 % of the other, the
   tracker's occlusion rule) in consecutive frames (gaps <= 5 frames) form one episode. For each episode:
   * ``min_dist_m``: the smallest distance between their feet on the court during the episode (ground-truth boxes,
     projected with the clip's verified homographies): how close they really were;
   * ``feet_hidden``: share of the episode's frames where the feet of the player behind (the box whose bottom is
     higher in the image, i.e. further from the camera) fall inside the box of the player in front;
   * ``swap``: did the run swap these two players (a switch where one gets an id last carried by the other) during
     the episode or within ``--window`` seconds after it?
   The swap rate by distance says whether metres could separate the swaps: if swaps only happen below ~0.5 m, the
   court cannot tell the players apart better than the image.
2. **Position error of the run's boxes.** For run boxes matched to a ground-truth box (IoU >= 0.5), the distance
   between the two projected feet, split by whether the player overlaps another one in that frame. This is what a
   tracker in metres would see: the error of its own boxes, worst exactly at crossings.
3. **Coverage**: frames with a verified court (frames without one get no position: the tracker would fall back
   to pixels there).

Outputs (``--out``): ``episodes.csv`` (one row per episode), ``position_error.csv`` (per clip) and a printed
summary.

Usage (repo root, after the court of every clip exists: ``scripts/court/c09_pipeline.ps1 -CourtOnly``):
    python scripts/court_crossings.py --config configs/player.yaml --run outputs/runs/029_siglip_ft `
        --court-dir outputs/court_complete/c09_pipeline --out outputs/runs/034_court_crossings
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_hota import result_file  # noqa: E402  (both run layouts: online and stitched)
from render_compare import load_mot, switch_records  # noqa: E402

from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.integrations.topdown import TopDownConfig, load_homographies, project_tracks  # noqa: E402
from hooptrack.tracking.matching import coverage_matrix, iou_matrix, linear_assignment  # noqa: E402

DIST_BINS = [0, 0.5, 1.0, 2.0, 3.0, np.inf]           # metres, for the swap-rate table


def positions(tracks: dict, H: dict, spec, tdc: TopDownConfig) -> dict[tuple[int, int], tuple[float, float]]:
    """{(frame, id): (x, y) in metres}, with the map's own projection (C08), so the numbers are the map's."""
    df = project_tracks(tracks, H, spec, tdc)
    return {(int(f), int(i)): (x, y) for f, i, x, y in df[["frame", "id", "x", "y"]].itertuples(index=False)}


def overlap_pairs(gt: dict, min_coverage: float = 0.2) -> dict[tuple[int, int], list[int]]:
    """{(id a, id b) with a < b: frames where one box covers >= min_coverage of the other}."""
    out: dict[tuple[int, int], list[int]] = {}
    for f, (ids, boxes) in sorted(gt.items()):
        if len(ids) < 2:
            continue
        cov = coverage_matrix(np.asarray(boxes, float))
        hit = (cov >= min_coverage) | (cov.T >= min_coverage)  # noqa: SIM300  (symmetric: either covers the other)
        for i, j in zip(*np.nonzero(np.triu(hit, 1)), strict=True):
            a, b = sorted((int(ids[i]), int(ids[j])))
            out.setdefault((a, b), []).append(int(f))
    return out


def episodes(frames: list[int], max_gap: int = 5) -> list[tuple[int, int]]:
    """Sorted frames -> (first, last) of each run of frames with gaps <= max_gap."""
    runs, start, prev = [], frames[0], frames[0]
    for f in frames[1:]:
        if f - prev > max_gap + 1:
            runs.append((start, prev))
            start = f
        prev = f
    runs.append((start, prev))
    return runs


def feet_hidden(box_a: np.ndarray, box_b: np.ndarray) -> bool:
    """Are the feet of the player BEHIND inside the box of the player in front?

    Behind = the box whose bottom is higher in the image (further from the camera, since the camera looks down on
    the court). His feet are the bottom centre of his box; if that point falls inside the other box, the detector
    cannot see them and its box will usually be cut above them.
    """
    back, front = (box_a, box_b) if box_a[3] < box_b[3] else (box_b, box_a)
    cx, cy = (back[0] + back[2]) / 2, back[3]
    return bool(front[0] <= cx <= front[2] and front[1] <= cy <= front[3])


def position_errors(run: dict, gt: dict, run_pos: dict, gt_pos: dict, overlapping: dict[int, set[int]]) -> list:
    """(error in metres, overlapping?) for every run box matched to a ground-truth box (IoU >= 0.5)."""
    out = []
    for f, (gids, gboxes) in gt.items():
        if f not in run:
            continue
        rids, rboxes = run[f]
        iou = iou_matrix(np.asarray(gboxes, float), np.asarray(rboxes, float))
        matches, _, _ = linear_assignment(1.0 - iou, 0.5)
        for g, r in matches:
            a, b = gt_pos.get((f, int(gids[g]))), run_pos.get((f, int(rids[r])))
            if a is not None and b is not None:
                out.append((float(np.hypot(a[0] - b[0], a[1] - b[1])), int(gids[g]) in overlapping.get(f, set())))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--run", type=Path, required=True, help="run whose swaps are diagnosed (029)")
    ap.add_argument("--court-dir", default="outputs/court_complete/c09_pipeline",
                    help="<dir>/<clip>/homographies.npz (C09 + C07, completed)")
    ap.add_argument("--window", type=float, default=1.0, help="seconds after an episode in which a swap counts")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    tdc = replace(TopDownConfig.from_dict(cfg.get("topdown")), dir=args.court_dir)
    tdc_gt = replace(tdc, smooth=0)                    # ground-truth boxes are exact: no smoothing
    ep_rows, err_rows = [], []
    for s in _sequences(apply_overrides(cfg, ["data.only=null"])):
        res, court = result_file(args.run, s.name), load_homographies(tdc, s.name)
        if res is None or court is None:
            print(f"{s.name}: {'no run result' if res is None else 'no verified court'} - skipped")
            continue
        spec, H = court
        gt, run = s.load_gt(), load_mot(res)
        gt_pos, run_pos = positions(gt, H, spec, tdc_gt), positions(run, H, spec, tdc)
        window = int(round(args.window * s.fps))

        # swaps of the run, as unordered pairs of ground-truth players, with their frame
        swaps = [(r["frame"], tuple(sorted((r["gt_id"], r["previous_owner"]))))
                 for r in switch_records(run, gt, s.length, player_teams(s)) if r["kind"] in ("mate", "opponent")]
        used = set()

        pairs = overlap_pairs(gt)
        overlapping: dict[int, set[int]] = {}         # frame -> players overlapping someone (for section 2)
        for (a, b), frames in pairs.items():
            for f in frames:
                overlapping.setdefault(f, set()).update((a, b))
        gt_box = {(f, int(i)): bx for f, (ids, boxes) in gt.items() for i, bx in zip(ids, boxes, strict=True)}

        for (a, b), frames in pairs.items():
            for first, last in episodes(frames):
                fr = [f for f in frames if first <= f <= last]
                d = [np.hypot(gt_pos[(f, a)][0] - gt_pos[(f, b)][0], gt_pos[(f, a)][1] - gt_pos[(f, b)][1])
                     for f in fr if (f, a) in gt_pos and (f, b) in gt_pos]
                hidden = np.mean([feet_hidden(gt_box[(f, a)], gt_box[(f, b)]) for f in fr])
                hit = [k for k, (f, p) in enumerate(swaps) if p == (a, b) and first <= f <= last + window]
                used.update(hit)
                ep_rows.append({"clip": s.name, "a": a, "b": b, "first": first, "last": last, "frames": len(fr),
                                "court_frames": len(d), "min_dist_m": round(float(min(d)), 3) if d else np.nan,
                                "feet_hidden": round(float(hidden), 3), "swap": bool(hit)})
        n_out = len(swaps) - len(used)               # swaps of two players that never overlapped (other causes)

        err = position_errors(run, gt, run_pos, gt_pos, overlapping)
        e = pd.DataFrame(err, columns=["err", "overlap"])
        err_rows.append({"clip": s.name, "court_coverage": round(len(H) / s.length, 3), "swaps": len(swaps),
                         "swaps_without_overlap": n_out,
                         "err_median_free": round(e[~e.overlap].err.median(), 3),
                         "err_p90_free": round(e[~e.overlap].err.quantile(0.9), 3),
                         "err_median_overlap": round(e[e.overlap].err.median(), 3),
                         "err_p90_overlap": round(e[e.overlap].err.quantile(0.9), 3)})
        print(f"{s.name}: court on {len(H)}/{s.length} frames, {len(swaps)} swaps")

    ep, er = pd.DataFrame(ep_rows), pd.DataFrame(err_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    ep.to_csv(args.out / "episodes.csv", index=False)
    er.to_csv(args.out / "position_error.csv", index=False)

    print(f"\n{len(er)} clips, {len(ep)} crossing episodes, {int(ep.swap.sum())} with a swap "
          f"(+ {int(er.swaps_without_overlap.sum())} swaps between players that never overlapped)")
    known = ep.dropna(subset=["min_dist_m"])
    known = known.assign(dist=pd.cut(known.min_dist_m, DIST_BINS, right=False))
    tab = known.groupby("dist", observed=False).agg(episodes=("swap", "size"), swaps=("swap", "sum"),
                                                    feet_hidden=("feet_hidden", "mean"))
    tab["swap_rate"] = (tab.swaps / tab.episodes).round(3)
    tab["share_of_swaps"] = (tab.swaps / tab.swaps.sum()).round(3)
    print("\nEpisodes by the smallest real distance between the two players (ground truth, metres):\n"
          + tab.round(3).to_string())
    print("\nPosition error of the run's boxes on the court (m), free vs overlapping another player:\n"
          + er.to_string(index=False))


if __name__ == "__main__":
    main()
