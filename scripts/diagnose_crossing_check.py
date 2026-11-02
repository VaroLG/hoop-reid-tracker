"""Why does the crossing check fail in the tracker when the same model is right in the benchmark? (032)

The 016 benchmark gives the fine-tuned SigLIP of 028 98.8 % right decisions at team-mate crossings and 100 % at
opponent crossings. In the tracker (031) only ~40 % of its corrections were right. Between the two, several things
change at once. This script takes the ground-truth crossings of a split and makes the SAME decision
(keep = d(A_before, A_after) + d(B_before, B_after) vs swap = d(A_before, B_after) + d(B_before, A_after); right
when keep < swap, since ground-truth identities never swap) with inputs that come closer to the tracker's, one
change at a time:

* **L1 benchmark, tracker windows**: ground-truth boxes and identities, clean crops (no other ground-truth box
  covers 20 % of it, >= 50 px), one every 5 frames, 1.6 s before the crossing and 0.6 s after it (the online
  check's windows; the 016 benchmark uses 2 s and 2 s).
* **L2 detector boxes**: the same frames and players, but the crop is the cached DETECTION matched to the
  ground-truth box (IoU >= 0.5), as the tracker sees it. Frames with no matching detection are skipped.
* **L3 the tracker's look before**: the look before of a player is what the online run (011b) had under the id he
  carried just before the crossing: that track's boxes in the 1.6 s before, clean among the run's boxes. After an
  earlier undetected swap that id was on someone else: those crops are counted as **contaminated** (their best
  ground-truth match, IoU >= 0.5, is another player). Look after as L2.

A crossing is decided at a level when both players have >= 2 crops before and after (as the benchmark). The level
where accuracy falls is where the tracker loses what the model knows. Nothing here changes the tracker.

Usage (repo root; ground-truth crops of L1 reuse the benchmark's cache ``outputs/reid_bench/<name>/``):
    python scripts/diagnose_crossing_check.py --config configs/player.yaml `
        --model siglip_ft:siglip:weights/siglip_reid_sportsmot `
        --online outputs/runs/011b_no_referee_filter/segmentation --out outputs/runs/032_diagnosis.csv
"""

# The helper functions defined inside the per-clip loop read that clip's data (gt, run, dets) and are only
# called within the same iteration, so binding the loop variables late is intended here.
# ruff: noqa: B023
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reid_teammate_benchmark as bench  # noqa: E402
from eval_hota import result_file  # noqa: E402
from render_compare import load_mot  # noqa: E402

from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import list_sequences  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.eval.reid_bench import crossing_events  # noqa: E402
from hooptrack.reid import build_reid  # noqa: E402
from hooptrack.tracking.matching import coverage_matrix, iou_matrix  # noqa: E402


def md(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(1.0 - x @ y.T))


def clean_mask(boxes: np.ndarray, min_cov: float = 0.2) -> np.ndarray:
    return coverage_matrix(boxes).max(axis=1) < min_cov if len(boxes) > 1 else np.ones(len(boxes), bool)


class Embedder:
    """Embeddings of (frame, box) requests, computed once, one image read per frame."""

    def __init__(self, seq, reid):
        self.seq, self.reid, self.memo = seq, reid, {}

    def __call__(self, reqs: list[tuple[int, tuple]]) -> np.ndarray:
        todo: dict[int, list[tuple]] = {}
        for f, b in reqs:
            if (f, b) not in self.memo:
                todo.setdefault(f, []).append(b)
        for f, bs in todo.items():
            img = cv2.imread(str(self.seq.frame_path(f)))
            e = self.reid(img, np.array(bs, float))
            e = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)
            for b, v in zip(bs, e, strict=True):
                self.memo[(f, b)] = v
        return np.stack([self.memo[r] for r in reqs]) if reqs else np.zeros((0, 1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--model", default="siglip_ft:siglip:weights/siglip_reid_sportsmot")
    ap.add_argument("--online", type=Path, required=True, help="online run whose ids give the look before (L3)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--before", type=float, default=1.6)
    ap.add_argument("--after", type=float, default=0.6)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--min-height", type=float, default=50)
    ap.add_argument("--cache", type=Path, default=Path("outputs/reid_bench"))
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/032_diagnosis.csv"))
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], args.split, d.get("sport"), d["splits_dir"])
    det_dir = Path(cfg["detector"]["cache_dir"])
    name, rtype, *w = args.model.split(":", 2)
    reid = build_reid(apply_overrides(cfg, [f"reid.type={rtype}"] + ([f"reid.weights={w[0]}"] if w else []))["reid"])
    rows = []
    for s in seqs:
        rf = result_file(args.online, s.name)
        if rf is None:
            continue
        gt, teams, run = s.load_gt(), player_teams(s), load_mot(rf)
        dets = load_mot(det_dir / f"{s.name}.txt")                    # frame -> (-1 ids, TLBR boxes)
        emb = Embedder(s, reid)
        wb, wa = int(round(args.before * s.fps)), int(round(args.after * s.fps))
        # L1 crops = the benchmark's clean ground-truth crops (and its embedding cache, same model)
        crops = bench.sample_crops(s, args.stride, args.min_height)
        cache = args.cache / name / f"{s.name}.npz"
        feats = np.load(cache)["feats"] if cache.is_file() and len(np.load(cache)["pid"]) == len(crops) \
            else bench.embed(s, crops, reid)
        gt_crop = {(f, int(p)): (k, tuple(np.round(b, 2))) for k, (f, p, b) in enumerate(crops)}

        def gt_box(f, p):
            ids, boxes = gt.get(f, (np.zeros(0), np.zeros((0, 4))))
            hit = np.flatnonzero(np.asarray(ids) == p)
            return np.asarray(boxes, float)[hit[0]] if len(hit) else None

        def det_box(f, p):
            g = gt_box(f, p)
            if g is None or f not in dets or not len(dets[f][1]):
                return None
            iou = iou_matrix(g[None], dets[f][1])[0]
            return tuple(np.round(dets[f][1][iou.argmax()], 2)) if iou.max() >= 0.5 else None

        def owner(f, box):                                         # ground-truth player under a box (or None)
            ids, boxes = gt.get(f, (np.zeros(0), np.zeros((0, 4))))
            if not len(ids):
                return None
            iou = iou_matrix(np.asarray(box, float)[None], np.asarray(boxes, float))[0]
            return int(ids[iou.argmax()]) if iou.max() >= 0.5 else None

        def track_at(f, p):                                        # run id on player p at frame f (or None)
            g = gt_box(f, p)
            if g is None or f not in run:
                return None
            iou = iou_matrix(g[None], run[f][1])[0]
            return int(run[f][0][iou.argmax()]) if iou.max() >= 0.5 else None

        for a, b, start, end in crossing_events(gt):
            frames_b = [f for f in range(start - wb, start) if f % args.stride == 1 % args.stride]
            frames_a = [f for f in range(end + 1, end + wa + 1) if f % args.stride == 1 % args.stride]
            row = {"clip": s.name, "a": a, "b": b, "start": start, "end": end,
                   "kind": "mate" if teams.get(a, -1) == teams.get(b, -2) else "opp"}
            looks = {}
            for lvl in ("L1", "L2", "L3"):
                ok = True
                for p in (a, b):
                    # look after: L1 ground-truth crops, L2/L3 detector boxes (both on the right player)
                    aft = [(f, gt_crop[(f, p)]) for f in frames_a if (f, p) in gt_crop]
                    if lvl == "L1":
                        bef = [(f, gt_crop[(f, p)]) for f in frames_b if (f, p) in gt_crop]
                        fb = feats[[k for _, (k, _) in bef]] if bef else np.zeros((0, 1))
                        fa = feats[[k for _, (k, _) in aft]] if aft else np.zeros((0, 1))
                    else:
                        clean_after = [f for f, _ in aft]                       # same clean frames as L1
                        req_a = [(f, det_box(f, p)) for f in clean_after]
                        req_a = [r for r in req_a if r[1] is not None]
                        fa = emb(req_a) if req_a else np.zeros((0, 1))
                        if lvl == "L2":
                            req_b = [(f, det_box(f, p)) for f, _ in
                                     [(f, gt_crop[(f, p)]) for f in frames_b if (f, p) in gt_crop]]
                            req_b = [r for r in req_b if r[1] is not None]
                        else:                                               # L3: the tracker's id before
                            tid = next((track_at(f, p) for f in range(start - 1, start - 6, -1)
                                        if track_at(f, p) is not None), None)
                            req_b, cont = [], 0
                            if tid is not None:
                                for f in frames_b:
                                    if f not in run:
                                        continue
                                    ids, boxes = run[f]
                                    hit = np.flatnonzero(ids == tid)
                                    if not len(hit) or not clean_mask(boxes)[hit[0]] \
                                            or boxes[hit[0]][3] - boxes[hit[0]][1] < args.min_height:
                                        continue
                                    bx = tuple(np.round(boxes[hit[0]], 2))
                                    req_b.append((f, bx))
                                    cont += owner(f, bx) not in (p,)
                            row[f"L3_contaminated_{'a' if p == a else 'b'}"] = (cont, len(req_b))
                        fb = emb(req_b) if req_b else np.zeros((0, 1))
                    if len(fb) < 2 or len(fa) < 2:
                        ok = False
                        break
                    looks[(lvl, p)] = (fb, fa)
                if not ok:
                    row[f"{lvl}_decided"] = False
                    continue
                (ba, aa), (bb, ab_) = looks[(lvl, a)], looks[(lvl, b)]
                keep, swap = md(ba, aa) + md(bb, ab_), md(ba, ab_) + md(bb, aa)
                row.update({f"{lvl}_decided": True, f"{lvl}_right": keep < swap, f"{lvl}_D": swap - keep})
            rows.append(row)
        print(f"{s.name}: {sum(r['clip'] == s.name for r in rows)} crossings", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} ground-truth crossings ({(df['kind'] == 'mate').sum()} team-mates, "
          f"{(df['kind'] == 'opp').sum()} opponents)\n")
    print("| level | decided | right (all) | right, team-mates | right, opponents |\n|---|---|---|---|---|")
    for lvl, label in (("L1", "L1 ground truth, tracker windows"), ("L2", "L2 detector boxes"),
                       ("L3", "L3 tracker's look before")):
        dd = df[df[f"{lvl}_decided"].fillna(False).astype(bool)]
        r = dd[f"{lvl}_right"].astype(bool)
        print(f"| {label} | {len(dd)} | {r.mean():.3f} | {r[dd['kind'] == 'mate'].mean():.3f} "
              f"| {r[dd['kind'] == 'opp'].mean():.3f} |")
    if "L3_contaminated_a" in df:
        d3 = df[df["L3_decided"].fillna(False).astype(bool)].copy()

        def share(t):
            return t[0] / t[1] if isinstance(t, tuple) and t[1] else np.nan

        d3["contaminated"] = d3[["L3_contaminated_a", "L3_contaminated_b"]].map(share).max(axis=1) > 0
        print(f"\nL3: crossings where a look before contains another player's crops: "
              f"{d3['contaminated'].mean():.3f}")
        for c, g in d3.groupby("contaminated"):
            print(f"  contaminated={c}: {len(g)} crossings, right {g['L3_right'].astype(bool).mean():.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
