"""Team-mate ReID benchmark (experiment 016): same player vs team-mate vs opponent.

Measures how well each appearance model separates a player from his TEAM-MATES,
the case that causes the identity swaps in crossings (015). The metrics are
explained in ``hooptrack.eval.reid_bench``. In short: ``AUC mate`` = probability
that two crops of the same player are closer than two crops of team-mates
(0.5 = coin, 1.0 = perfect).

Crops (from the ground truth, so the benchmark measures the ReID model only, not
the detector or the tracker):

* one frame in ``--stride``; boxes at least ``--min-height`` px tall;
* only CLEAN boxes (no other player covering 20 % or more of it): a crop that
  mixes two players says nothing about either;
* random pairs: at most ``--per-player`` crops per player, spread over the clip, pairs at
  least 2 s apart (``min_gap``), so adjacent near-identical frames do not count;
* crossings (ground truth: two players overlapping >= 20 %): every clean crop in the 2 s before
  and the 2 s after. Is each player closer to himself than to the other one? This is the
  situation of the swaps found in 015, and it is harder than random pairs (same moment, light, pose).

Teams come from jersey-colour clustering of the ground-truth players
(``hooptrack.eval.gt_teams``), the same tool as the 004 ID-switch breakdown.

Default split = **val**, the 4 unseen games where tracking is measured. This is a
test: models are compared here, nothing is tuned here. ``--split train`` gives the
same numbers on the games the fine-tuned OSNet was trained on (expected optimistic).

Embeddings are cached per model and clip (``<out>/<model>/<clip>.npz``): adding
a model later only computes that model.

Usage (repo root), one entry per model as ``name:type[:weights]``:
    python scripts/reid_teammate_benchmark.py --config configs/player.yaml `
        --model resnet50_imagenet:resnet osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import list_sequences
from hooptrack.eval.gt_teams import player_teams
from hooptrack.eval.reid_bench import (
    auc,
    crossing_decisions,
    crossing_events,
    pair_distances,
    piece_level,
    pieces,
    rank1_among_teammates,
)
from hooptrack.reid import build_reid
from hooptrack.tracking.matching import coverage_matrix


def sample_crops(seq, stride: int, min_height: float):
    """[(frame, gt id, box)] of EVERY clean ground-truth box, one frame in ``stride``."""
    gt = seq.load_gt()
    by_pid: dict[int, list[tuple[int, np.ndarray]]] = {}
    for f in range(1, seq.length + 1, stride):
        if f not in gt:
            continue
        ids, boxes = gt[f]
        boxes = np.asarray(boxes, float)
        cov = coverage_matrix(boxes).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
        for pid, box, c in zip(ids, boxes, cov, strict=True):
            if c < 0.2 and box[3] - box[1] >= min_height:
                by_pid.setdefault(int(pid), []).append((f, box))
    out = [(f, pid, box) for pid, lst in by_pid.items() for f, box in lst]
    return sorted(out, key=lambda x: x[0])


def spread_subset(pid: np.ndarray, frame: np.ndarray, per_player: int) -> np.ndarray:
    """Indices of at most ``per_player`` crops per player, evenly spread in time (for the pair metrics:
    every player weighs the same, whatever his minutes on screen)."""
    keep = []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        idx = idx[np.argsort(frame[idx])]
        sel = np.unique(np.linspace(0, len(idx) - 1, min(per_player, len(idx))).round().astype(int))
        keep += idx[sel].tolist()
    return np.array(sorted(keep))


def embed(seq, crops, reid) -> np.ndarray:
    """L2-normalised embedding of every crop; one image read per frame."""
    rows: list[np.ndarray | None] = [None] * len(crops)
    by_frame: dict[int, list[int]] = {}
    for k, (f, _, _) in enumerate(crops):
        by_frame.setdefault(f, []).append(k)
    for f, ks in by_frame.items():
        img = cv2.imread(str(seq.frame_path(f)))
        e = reid(img, np.array([crops[k][2] for k in ks]))
        for k, v in zip(ks, e, strict=True):
            rows[k] = v / (np.linalg.norm(v) + 1e-12)
    return np.stack(rows).astype(np.float32) if rows else np.zeros((0, 0), np.float32)


def summarise(per_clip: list[dict], min_gap: int, piece_len: int, piece_min: int, per_player: int) -> dict:
    """Pool the pairs of all clips (pairs never cross clips) and compute the metrics."""
    crop = {"same": [], "mate": [], "opp": []}
    piece = {"same": [], "mate": [], "opp": []}
    r1c, r1p, rows, decisions = [], [], [], []
    for c in per_clip:
        # crossings: every clean crop around each crossing (window = min_gap = 2 s)
        dec = crossing_decisions(c["feats"], c["pid"], c["team"], c["frame"], c["events"], window=min_gap)
        decisions += [{**x, "clip": c["clip"]} for x in dec]
        s = spread_subset(c["pid"], c["frame"], per_player)
        args = (c["feats"][s], c["pid"][s], c["team"][s], c["frame"][s])
        dc = pair_distances(*args, min_gap)
        full = (c["feats"], c["pid"], c["team"], c["frame"])      # pieces use every crop, like a tracklet
        dp, r1_piece = piece_level(*full, pieces(c["pid"], c["frame"], piece_len, piece_min), min_gap)
        r1_crop = rank1_among_teammates(*args, min_gap)
        for k in crop:
            crop[k].append(dc[k])
            piece[k].append(dp[k])
        r1c.append((r1_crop, len(s)))
        r1p.append(r1_piece)
        rows.append({"clip": c["clip"], "crops": int(len(s)), "players": int(len(np.unique(c["pid"]))),
                     "auc_mate_crop": round(auc(dc["same"], dc["mate"]), 3),
                     "auc_mate_piece": round(auc(dp["same"], dp["mate"]), 3),
                     "rank1_mate_crop": round(r1_crop, 3)})
    cat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in crop.items()}
    pcat = {k: np.concatenate(v) if v else np.zeros(0) for k, v in piece.items()}
    w = np.array([n for _, n in r1c], float)
    return {
        "crop": {"auc_mate": auc(cat["same"], cat["mate"]), "auc_opp": auc(cat["same"], cat["opp"]),
                 "rank1_mate": float(np.average([r for r, _ in r1c], weights=w)),
                 "median_same": float(np.median(cat["same"])), "median_mate": float(np.median(cat["mate"])),
                 "median_opp": float(np.median(cat["opp"])),
                 "pairs": {k: int(len(v)) for k, v in cat.items()}},
        "piece": {"auc_mate": auc(pcat["same"], pcat["mate"]), "auc_opp": auc(pcat["same"], pcat["opp"]),
                  "rank1_mate": float(np.nanmean(r1p)), "pairs": {k: int(len(v)) for k, v in pcat.items()}},
        "crossing": crossing_summary(decisions),
        "per_clip": rows,
        "crossings": decisions,
    }


def crossing_summary(decisions: list[dict]) -> dict:
    """AUC (same player before/after vs the other player) and how often keep < swap, per kind."""
    out = {}
    for kind in ("mate", "opp"):
        ds = [x for x in decisions if x["kind"] == kind]
        same = [v for x in ds for v in x["same"]]
        other = [v for x in ds for v in x["other"]]
        out[kind] = {"crossings": len(ds), "auc": auc(same, other),
                     "keep_wins": float(np.mean([x["keep"] < x["swap"] for x in ds])) if ds else float("nan")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", nargs="+", required=True, help="name:type[:weights], e.g. osnet_ft:osnet:weights/x.pth")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every basketball clip of the split")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-height", type=float, default=50)
    ap.add_argument("--per-player", type=int, default=60)
    ap.add_argument("--out", type=Path, default=Path("outputs/reid_bench"))
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], args.split, d.get("sport"), d["splits_dir"])
    if args.clips:
        seqs = [s for s in seqs if s.name in set(args.clips)]
    if not seqs:
        raise SystemExit("no clips found")

    # crops and teams do not depend on the model: computed once
    crops = {s.name: sample_crops(s, args.stride, args.min_height) for s in seqs}
    events = {s.name: crossing_events(s.load_gt()) for s in seqs}
    teams = {s.name: player_teams(s) for s in seqs}
    min_gap = {s.name: int(round(2 * s.fps)) for s in seqs}

    results = {}
    for spec in args.model:
        name, rtype, *w = spec.split(":", 2)
        rcfg = apply_overrides(cfg, [f"reid.type={rtype}"] + ([f"reid.weights={w[0]}"] if w else []))["reid"]
        reid = None
        per_clip = []
        for s in seqs:
            cache = args.out / name / f"{s.name}.npz"
            cr = crops[s.name]
            if cache.is_file() and len(np.load(cache)["pid"]) == len(cr):
                feats = np.load(cache)["feats"]
            else:
                reid = reid or build_reid(rcfg)
                feats = embed(s, cr, reid)
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.savez(cache, feats=feats, pid=np.array([p for _, p, _ in cr]))
            pid = np.array([p for _, p, _ in cr])
            per_clip.append({"clip": s.name, "feats": feats, "pid": pid,
                             "team": np.array([teams[s.name].get(int(p), -1) for p in pid]),
                             "frame": np.array([f for f, _, _ in cr]), "events": events[s.name]})
            print(f"{name} | {s.name}: {len(cr)} crops, {len(np.unique(pid))} players")
        gap = min_gap[seqs[0].name]
        results[name] = summarise(per_clip, gap, piece_len=gap, piece_min=4, per_player=args.per_player)

    print(f"\n## Team-mate ReID benchmark · split {args.split} · {len(seqs)} clips\n")
    print("Random pairs >= 2 s apart (whole clip):\n")
    print("| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates (crop) "
          "| AUC mate (2 s piece) | rank-1 among team-mates (piece) |")
    print("|---|---|---|---|---|---|")
    for name, r in results.items():
        c, p = r["crop"], r["piece"]
        print(f"| {name} | {c['auc_mate']:.3f} | {c['auc_opp']:.3f} | {c['rank1_mate']:.3f} "
              f"| {p['auc_mate']:.3f} | {p['rank1_mate']:.3f} |")
    print("\nAround crossings (2 s before vs 2 s after, the decision a swap check has to take):\n")
    print("| model | team-mate crossings | AUC | keep < swap | opponent crossings | AUC | keep < swap |")
    print("|---|---|---|---|---|---|---|")
    for name, r in results.items():
        m, o = r["crossing"]["mate"], r["crossing"]["opp"]
        print(f"| {name} | {m['crossings']} | {m['auc']:.3f} | {m['keep_wins']:.3f} "
              f"| {o['crossings']} | {o['auc']:.3f} | {o['keep_wins']:.3f} |")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"summary_{args.split}.json"
    old = json.loads(out.read_text()) if out.is_file() else {}
    old.update(results)
    out.write_text(json.dumps(old, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
