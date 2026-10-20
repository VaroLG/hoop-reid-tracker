"""Homography for every frame of a clip, with no annotation (court chapter, C05).

    candidates (straight lines, one frame at a time)  ->  confirmation (two chains must agree)
    ->  merge (best line agreement per frame)

See ``hooptrack.court.autoinit`` for the method. If ``--reference`` is given
(e.g. the hand-anchored homographies of C04), every automatic homography is
compared with it — in metres on the visible court — to MEASURE the method; the
reference is never used to produce the result.

Usage:
    python scripts/court/auto_homographies.py --sequence v_00HRwkvvjtQ_c001 --league NCAA \
        --reference outputs/court_homographies/v_00HRwkvvjtQ_c001/homographies.npz
    python scripts/court/auto_homographies.py --sequence v_2j7kLB-vEEk_c002 --league auto
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from _clip import Clip

from hooptrack.court.autoinit import confirm, evidence, find_candidates
from hooptrack.court.leagues import SPECS
from hooptrack.court.propagate import disagreement
from hooptrack.court.search import search_lines
from hooptrack.detection.filters import CourtMask


def pick_league(clip: Clip, hull, ignore_below, frames, evidence_kw=None) -> tuple[str, dict]:
    """League whose template explains the straight lines best, summed over a few frames."""
    totals = {}
    for name, spec in SPECS.items():
        t = 0.0
        for f in frames:
            img = clip.frame(f)
            ev = evidence(img, _people(clip, f), hull(f, img), ignore_below, evidence_kw)
            c = search_lines(ev, spec, img.shape[1::-1])
            t += max(c[0].score, 0) if c else 0
        totals[name] = round(t, 1)
    return max(totals, key=totals.get), totals


def _people(clip: Clip, f: int):
    return clip.people(f)        # None when the clip has no cached detections: players are not masked


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--league", default="auto", help="NBA, FIBA, NCAA or auto")
    ap.add_argument("--stride", type=int, default=25, help="look for a candidate every N frames")
    ap.add_argument("--min-score", type=float, default=100.0)
    ap.add_argument("--agree-m", type=float, default=0.5)
    ap.add_argument("--ignore-below", type=int, default=575, help="score bar: rows below are ignored")
    ap.add_argument("--reference", type=Path, help="homographies to compare with (measurement only)")
    ap.add_argument("--out", type=Path, default=Path("outputs/court_auto"))
    ap.add_argument("--edges", action="store_true", help="009: also use borders of painted areas as lines")
    ap.add_argument("--white", action="store_true", help="009: also look for white strokes in min(R,G,B)")
    ap.add_argument("--budget", type=float, default=0, help="stop (resumable) after this many seconds; 0 = no limit")
    args = ap.parse_args()

    t0 = time.time()
    clip = Clip(args.sequence)
    evidence_kw = {"edges": args.edges, "white": args.white}
    tag = "".join(k[0] for k, v in evidence_kw.items() if v)        # cache files differ per evidence setting
    out = args.out / args.sequence
    out.mkdir(parents=True, exist_ok=True)
    cm = CourtMask()

    def hull(f, img):
        return cm.estimate_hull(img, _people(clip, f) if _people(clip, f) is not None else np.zeros((0, 4)))

    people = (lambda f: _people(clip, f)) if _people(clip, 1) is not None else None
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    league_file = out / f"league{tag}.json"
    if args.league != "auto":
        league, totals = args.league, {}
    elif league_file.exists():                                    # decided in an earlier (resumed) run
        league, totals = json.loads(league_file.read_text()).values()
    else:
        league, totals = pick_league(clip, hull, args.ignore_below,
                                     range(1, clip.length + 1, max(clip.length // 6, 1)), evidence_kw)
        league_file.write_text(json.dumps({"league": league, "scores": totals}))
    log(f"league: {league} {totals}")
    spec = SPECS[league]

    # resumable: everything computed so far is kept in a cache file and reused by the next run
    cache_path = out / f"cache_{league}_s{args.stride}{tag}.pkl"
    cache = pickle.loads(cache_path.read_bytes()) if cache_path.exists() else {"cands": {}, "links": {}}

    def checkpoint():
        cache_path.write_bytes(pickle.dumps(cache))
        if args.budget and time.time() - t0 > args.budget:
            print("time budget reached: run the same command again to continue", flush=True)
            sys.exit(3)

    cands = find_candidates(clip.frame, clip.length, spec, people, hull, args.ignore_below, args.stride,
                            args.min_score, log, cache["cands"], checkpoint, evidence_kw)
    log(f"{len(cands)} candidates over {len(range(1, clip.length + 1, args.stride))} searched frames "
        f"({time.time() - t0:.0f} s)")
    res = confirm(clip.frame, clip.length, cands, spec, people, args.ignore_below, args.agree_m, log=log,
                  cache=cache["links"], on_step=checkpoint, evidence_kw=evidence_kw)
    log(f"confirmed anchors: {res.confirmed} ({time.time() - t0:.0f} s)")

    frames = np.array(sorted(res.homographies))
    report = {
        "sequence": args.sequence, "league": league, "league_scores": totals, "stride": args.stride,
        "evidence": evidence_kw,
        "candidates": {int(f): round(c.score, 1) for f, c in cands.items()},
        "confirmed": res.confirmed,
        "links": {f"{a}-{b}": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}
                  for (a, b), s in res.links.items()},
        "frames_with_H": int(len(frames)), "clip_length": clip.length,
        "frames_dropped_conflict": len(res.conflict),
        "seconds": round(time.time() - t0),
    }
    if len(frames):
        agree = np.array([res.homographies[f].agreement for f in frames])
        report["agreement_median"] = round(float(np.median(agree)), 3)
        np.savez(out / "homographies.npz", frames=frames, H=np.array([res.homographies[f].H for f in frames]),
                 agreement=agree, league=league)
    if args.reference and len(frames):
        z = np.load(args.reference)
        ref = dict(zip(z["frames"].tolist(), z["H"], strict=True))
        err = np.array([disagreement(res.homographies[f].H, ref[f], spec, (1280, args.ignore_below))[1]
                        for f in frames if f in ref])
        report["vs_reference"] = {
            "frames": int(len(err)), "median_m": round(float(np.nanmedian(err)), 3),
            "p90_m": round(float(np.nanpercentile(err, 90)), 3),
            "share_below_0.5m": round(float(np.mean(err < 0.5)), 3),
            "share_below_1m": round(float(np.mean(err < 1.0)), 3),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    (out / "log.txt").write_text("\n".join(log_lines))
    print(json.dumps({k: v for k, v in report.items() if k not in ("candidates", "links")}, indent=2))


if __name__ == "__main__":
    main()
