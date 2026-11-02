"""Stitching threshold for a new ReID model, calibrated on the TRAIN split (experiment 029).

Why: ``stitch_gta`` joins two pieces when the mean cosine distance of their crops is below ``thresh`` (0.4, the
value GTA published for SportsMOT, used by 013b-020 with OSNet). Each model has its own distance scale: the
fine-tuned SigLIP of 028 puts the same player at ~0.4 and other players at ~0.7, OSNet at ~0.04 and ~0.16. Using
0.4 with SigLIP would be a different operating point, and choosing a threshold by looking at the val clips would
tune on the clips we report. So the threshold is transferred on the train clips.

Rule (fixed before running):

* **primary**: the new threshold makes the same share of WRONG joins between team-mate pieces as the reference
  model at its threshold (021: wrong joins between team-mates are the main error left);
* **fallback**, only if the reference makes no team-mate error at all on train (then the primary rule has no
  operating point to copy): the same share of RIGHT joins between pieces of the same player.

Pieces and distances exactly as the team-mate benchmark (016) measures them, i.e. as ``stitch_gta`` compares
pieces: 2-second pieces of ground-truth crops (clean, >= 50 px tall), mean cosine distance over all crop pairs,
pairs of pieces at least 2 s apart. Teams from jersey colour on the ground truth (``hooptrack.eval.gt_teams``).

Caveat, written down: both OSNet 012 and the 028 models were TRAINED on these clips, so both look better here than
on val. The rule copies an error RATE from one model to the other on the same data, which is less sensitive to
that than an absolute threshold would be, but it is not unbiased.

Usage (repo root):
    python scripts/calibrate_stitch_threshold.py --config configs/player.yaml `
        --reference osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth --reference-thresh 0.4 `
        --model siglip_ft:siglip:weights/siglip_reid_sportsmot clip_ft:clip:weights/clip_reid_sportsmot
Embeddings are cached in ``outputs/reid_bench/<name>/`` (the benchmark's cache: train clips have their own files).
Result: ``outputs/calibration/stitch_threshold.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reid_teammate_benchmark as bench  # noqa: E402  (same crops and embeddings as the benchmark)

from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import list_sequences  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.eval.reid_bench import piece_level, pieces  # noqa: E402
from hooptrack.reid import build_reid  # noqa: E402


def piece_distances(spec: str, cfg: dict, seqs, crops, teams, cache_root: Path) -> dict[str, np.ndarray]:
    """{"same", "mate", "opp"}: piece-pair distances over all clips, for the model ``name:type[:weights]``."""
    name, rtype, *w = spec.split(":", 2)
    rcfg = apply_overrides(cfg, [f"reid.type={rtype}"] + ([f"reid.weights={w[0]}"] if w else []))["reid"]
    reid, out = None, {"same": [], "mate": [], "opp": []}
    for s in seqs:
        cr = crops[s.name]
        cache = cache_root / name / f"{s.name}.npz"
        if cache.is_file() and len(np.load(cache)["pid"]) == len(cr):
            feats = np.load(cache)["feats"]
        else:
            reid = reid or build_reid(rcfg)
            feats = bench.embed(s, cr, reid)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, feats=feats, pid=np.array([p for _, p, _ in cr]))
        pid = np.array([p for _, p, _ in cr])
        team = np.array([teams[s.name].get(int(p), -1) for p in pid])
        frame = np.array([f for f, _, _ in cr])
        gap = int(round(2 * s.fps))
        d, _ = piece_level(feats, pid, team, frame, pieces(pid, frame, gap, 4), gap)
        for k in out:
            out[k].append(d[k])
        print(f"{name} | {s.name}: {len(cr)} crops", flush=True)
    return {k: np.concatenate(v) for k, v in out.items()}


def threshold_for_rate(dist: np.ndarray, rate: float) -> float:
    """Threshold t such that the share of ``dist`` below t is ``rate`` (the empirical quantile)."""
    return float(np.quantile(dist, rate))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--reference", default="osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth")
    ap.add_argument("--reference-thresh", type=float, default=0.4)
    ap.add_argument("--model", nargs="+", required=True, help="name:type[:weights]")
    ap.add_argument("--split", default="train", help="train: the split the threshold is transferred on")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-height", type=float, default=50)
    ap.add_argument("--cache", type=Path, default=Path("outputs/reid_bench"))
    ap.add_argument("--out", type=Path, default=Path("outputs/calibration/stitch_threshold.json"))
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], args.split, d.get("sport"), d["splits_dir"])
    if not seqs:
        raise SystemExit(f"no {args.split} clips found")
    crops = {s.name: bench.sample_crops(s, args.stride, args.min_height) for s in seqs}
    teams = {s.name: player_teams(s) for s in seqs}

    ref = piece_distances(args.reference, cfg, seqs, crops, teams, args.cache)
    t_ref = args.reference_thresh
    wrong_rate = float(np.mean(ref["mate"] < t_ref))          # team-mate pieces that would be joined
    right_rate = float(np.mean(ref["same"] < t_ref))          # same-player pieces that would be joined
    report = {"split": args.split, "clips": len(seqs),
              "reference": {"model": args.reference, "thresh": t_ref, "pairs": {k: len(v) for k, v in ref.items()},
                            "teammate_wrong_join_rate": wrong_rate, "same_player_join_rate": right_rate},
              "models": {}}
    rule = "primary: same team-mate wrong-join rate" if wrong_rate > 0 else \
        "fallback: same same-player join rate (the reference makes no team-mate error on this split)"
    print(f"\nreference {args.reference.split(':')[0]} at {t_ref}: team-mate wrong joins {wrong_rate:.4f}, "
          f"same-player joins {right_rate:.4f} -> rule {rule}")
    for spec in args.model:
        dm = piece_distances(spec, cfg, seqs, crops, teams, args.cache)
        t = threshold_for_rate(dm["mate"], wrong_rate) if wrong_rate > 0 else threshold_for_rate(dm["same"], right_rate)
        report["models"][spec.split(":")[0]] = {
            "model": spec, "thresh": round(t, 4), "rule": rule,
            "teammate_wrong_join_rate": float(np.mean(dm["mate"] < t)),
            "same_player_join_rate": float(np.mean(dm["same"] < t)),
            "opponent_join_rate": float(np.mean(dm["opp"] < t)),
            "median": {k: float(np.median(v)) for k, v in dm.items()}}
        r = report["models"][spec.split(":")[0]]
        print(f"{spec.split(':')[0]}: thresh {t:.4f} | team-mate wrong joins {r['teammate_wrong_join_rate']:.4f} | "
              f"same-player joins {r['same_player_join_rate']:.4f} (reference {right_rate:.4f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
