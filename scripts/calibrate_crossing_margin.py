"""Margin of the crossing check (030), calibrated on the TRAIN split.

The crossing check swaps two identities when ``keep - swap > margin`` (``hooptrack.tracking.crossing_check``).
Two errors pull the margin in opposite directions:

* a real swap made by the tracker is fixed only if the model sees it clearly enough (low margin);
* a crossing where the tracker was right must not be "fixed" (high margin). This matters more, because most
  crossings end with no swap: 015 measured 21 swaps in 493 crossings (~4 %), so a model wrong 1 % of the time on
  the other 96 % makes about as many false swaps as there are real ones (the 016 base-rate argument).

Data: ground-truth crossings of the train clips with the 016 benchmark machinery
(``crossing_events`` / ``crossing_decisions``), with the SAME windows as the online check: up to 1.6 s of clean
crops before the crossing, 0.6 s after it, one crop every 5 frames (the benchmark's stride). With ground-truth ids
every crossing is a "keep": ``D = swap - keep`` should be positive.

* a crossing where the tracker was right is swapped by mistake if ``keep - swap > m``, i.e. ``D < -m``;
* a crossing where the tracker DID swap the two players shows the model the same crops with the names exchanged,
  so its keep and swap are exchanged too: the check fixes it if ``D > m``.

Rule (fixed before running): with ``p`` = share of crossings that end in a swap (0.04, from 015), choose

    m* = argmax_m  p * P(D > m)  -  (1 - p) * P(D < -m)        (expected swaps fixed minus swaps created, per crossing)

over a grid. Reported per kind of crossing (team-mates / opponents) as well.

Usage (repo root; the train embeddings of 029's calibration are reused from ``outputs/reid_bench/<name>/``):
    python scripts/calibrate_crossing_margin.py --config configs/player.yaml `
        --model siglip_ft:siglip:weights/siglip_reid_sportsmot
Result: ``outputs/calibration/crossing_margin.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reid_teammate_benchmark as bench  # noqa: E402  (same crops and embedding cache as the benchmark)

from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import list_sequences  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.eval.reid_bench import crossing_decisions, crossing_events  # noqa: E402
from hooptrack.reid import build_reid  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", default="siglip_ft:siglip:weights/siglip_reid_sportsmot", help="name:type[:weights]")
    ap.add_argument("--split", default="train")
    ap.add_argument("--swap-rate", type=float, default=0.04, help="share of crossings that end in a swap (015)")
    ap.add_argument("--before", type=float, default=1.6, help="seconds of crops before it (online: 8 x 5 frames)")
    ap.add_argument("--after", type=float, default=0.6, help="seconds of crops after it (online: 3 x 5 frames)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-height", type=float, default=50)
    ap.add_argument("--cache", type=Path, default=Path("outputs/reid_bench"))
    ap.add_argument("--out", type=Path, default=Path("outputs/calibration/crossing_margin.json"))
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], args.split, d.get("sport"), d["splits_dir"])
    name, rtype, *w = args.model.split(":", 2)
    rcfg = apply_overrides(cfg, [f"reid.type={rtype}"] + ([f"reid.weights={w[0]}"] if w else []))["reid"]
    reid, decisions = None, []
    for s in seqs:
        cr = bench.sample_crops(s, args.stride, args.min_height)
        cache = args.cache / name / f"{s.name}.npz"
        if cache.is_file() and len(np.load(cache)["pid"]) == len(cr):
            feats = np.load(cache)["feats"]
        else:
            reid = reid or build_reid(rcfg)
            feats = bench.embed(s, cr, reid)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, feats=feats, pid=np.array([p for _, p, _ in cr]))
        teams = player_teams(s)
        pid = np.array([p for _, p, _ in cr])
        dec = crossing_decisions(feats, pid, np.array([teams.get(int(p), -1) for p in pid]),
                                 np.array([f for f, _, _ in cr]), crossing_events(s.load_gt()),
                                 window=int(round(args.before * s.fps)), min_crops=2,
                                 window_after=int(round(args.after * s.fps)))
        decisions += dec
        print(f"{s.name}: {len(dec)} crossings", flush=True)

    D = np.array([x["swap"] - x["keep"] for x in decisions])
    kind = np.array([x["kind"] for x in decisions])
    p = args.swap_rate
    grid = np.round(np.arange(-0.2, 0.6001, 0.01), 3)
    net = [p * np.mean(m < D) - (1 - p) * np.mean(-m > D) for m in grid]
    m_best = float(grid[int(np.argmax(net))])
    rows = []
    for m in grid:
        row = {"margin": float(m), "fix_rate": float(np.mean(m < D)), "false_swap_rate": float(np.mean(-m > D))}
        for k in ("mate", "opp"):
            dk = D[kind == k]
            row[f"fix_rate_{k}"] = float(np.mean(dk > m)) if len(dk) else None
            row[f"false_swap_rate_{k}"] = float(np.mean(dk < -m)) if len(dk) else None
        rows.append(row)
    best = next(r for r in rows if r["margin"] == m_best)
    report = {"split": args.split, "model": args.model, "crossings": len(D),
              "by_kind": {k: int(np.sum(kind == k)) for k in ("mate", "opp")},
              "windows_s": {"before": args.before, "after": args.after}, "swap_rate": p,
              "margin": m_best, "at_margin": best, "curve": rows}
    print(f"\n{len(D)} crossings ({report['by_kind']}), swap rate assumed {p}")
    print(f"margin {m_best:+.2f}: fixes {best['fix_rate']:.3f} of real swaps, swaps by mistake "
          f"{best['false_swap_rate']:.4f} of right crossings "
          f"(team-mates {best['false_swap_rate_mate']}, opponents {best['false_swap_rate_opp']})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
