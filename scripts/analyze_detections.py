"""Diagnose a run: is the bottleneck the DETECTOR or the TRACKER?

Tracking metrics mix two error sources. This script separates them using the
cached detections of a run and the ground truth:

1. Detector quality (per-box, IoU >= 0.5 with Hungarian matching)
   - recall at the lowest score kept, precision at several score thresholds
   - size of the false positives vs the size of real players

2. Tracker quality under three inputs, with the SAME motion-only tracker
   (no ReID, no camera compensation, no memory — so the numbers only depend
   on the boxes that come in):
     A. the detector's boxes as they are
     B. only the detector's boxes that match a player  -> what a perfect
        "players only" filter / fine-tuned detector would give
     C. the ground-truth boxes                          -> upper bound

   A -> B is what fixing false positives is worth.
   B -> C, and anything ReID/GMC/memory add on top, is the tracker's job.

Usage:
    python scripts/analyze_detections.py \
        --set detector.cache_dir=outputs/detections/yolo26m "data.only=[v_00HRwkvvjtQ_c001]" \
        --out experiments/001_baseline_yolo26m_coco/detection_analysis.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hooptrack.config import load_config
from hooptrack.data.sportsmot import list_sequences
from hooptrack.eval.metrics import clear_and_identity, reentry_recovery
from hooptrack.tracking import ReIDTracker, TrackerConfig
from hooptrack.tracking.matching import iou_matrix, linear_assignment

EMPTY = (np.zeros((0, 4)), np.zeros(0))


def load_cached(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """MOT-det file -> frame -> (TLBR boxes, scores)."""
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    out = {}
    for f in np.unique(data[:, 0]).astype(int):
        r = data[data[:, 0] == f]
        out[int(f)] = (np.c_[r[:, 2:4], r[:, 2:4] + r[:, 4:6]], r[:, 6])
    return out


def match(gt_boxes: np.ndarray, det_boxes: np.ndarray, iou: float = 0.5) -> set[int]:
    """Indices of detections matched one-to-one to a ground-truth box."""
    m, _, _ = linear_assignment(1.0 - iou_matrix(gt_boxes, det_boxes), 1.0 - iou)
    return {c for _, c in m}


def track(frames: dict, n_frames: int, fps: float, gt: dict) -> dict:
    """Run the motion-only tracker on frame -> (boxes, scores) and score it."""
    trk = ReIDTracker(TrackerConfig(gmc="none", memory_enabled=False, fps=fps))
    pred = {}
    for f in range(1, n_frames + 1):
        boxes, scores = frames.get(f, EMPTY)
        out = trk.update(boxes, scores, None)
        pred[f] = (np.array([o.track_id for o in out], dtype=int),
                   np.array([o.tlbr for o in out]).reshape(-1, 4))
    m = clear_and_identity({"s": (gt, pred)}).loc["OVERALL"]
    r = reentry_recovery(gt, pred)
    return {"IDF1": 100 * m["idf1"], "MOTA": 100 * m["mota"], "IDSW": int(m["num_switches"]),
            "ids": trk._next_id, "reentry": f"{r.recovered}/{r.events}"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.6])
    ap.add_argument("--out", type=Path, default=None, help="write the report as Markdown")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    d = cfg["data"]
    seqs = list_sequences(d["root"], d["split"], d.get("sport"), d["splits_dir"])
    if d.get("only"):
        seqs = [s for s in seqs if s.name in set(d["only"])]
    cache = Path(cfg["detector"]["cache_dir"])

    lines = ["# Detection vs tracking diagnosis", "",
             f"Detections: `{cache}` · split: `{d['split']}` · sequences: {len(seqs)}", ""]
    for seq in seqs:
        gt = seq.load_gt()
        dets = load_cached(cache / f"{seq.name}.txt")
        n_gt = sum(len(v[0]) for v in gt.values())

        # ---------------------------------------------------- detector quality
        tp_scores, fp_scores, fp_h = [], [], []
        for f in range(1, seq.length + 1):
            boxes, scores = dets.get(f, EMPTY)
            ok = match(gt.get(f, (None, np.zeros((0, 4))))[1], boxes)
            for j in range(len(boxes)):
                if j in ok:
                    tp_scores.append(scores[j])
                else:
                    fp_scores.append(scores[j])
                    fp_h.append((scores[j], boxes[j, 3] - boxes[j, 1]))
        tp_s, fp_s = np.array(tp_scores), np.array(fp_scores)
        gt_h = np.concatenate([v[1][:, 3] - v[1][:, 1] for v in gt.values()])
        fp_h = np.array(fp_h).reshape(-1, 2)

        lines += [f"## {seq.name}", "",
                  f"- Ground truth: {n_gt / seq.length:.1f} players/frame, "
                  f"{len(set(np.concatenate([v[0] for v in gt.values()])))} identities, {seq.length} frames",
                  f"- Recall at the lowest kept score: **{100 * len(tp_s) / max(n_gt, 1):.1f} %**",
                  f"- Median box height — players: {np.median(gt_h):.0f} px · "
                  f"false positives (score ≥ 0.5): {np.median(fp_h[fp_h[:, 0] >= 0.5, 1]):.0f} px", "",
                  "| score ≥ | detections/frame | true positives | false positives | precision |",
                  "|---|---|---|---|---|"]
        for th in args.thresholds:
            t, fp = int((tp_s >= th).sum()), int((fp_s >= th).sum())
            lines.append(f"| {th:.1f} | {(t + fp) / seq.length:.1f} | {t} | {fp} | {100 * t / max(t + fp, 1):.1f} % |")

        # ----------------------------------------------------- tracker quality
        only_players = {}
        for f, (boxes, scores) in dets.items():
            keep = sorted(match(gt.get(f, (None, np.zeros((0, 4))))[1], boxes))
            only_players[f] = (boxes[keep], scores[keep])
        gt_frames = {f: (b, np.full(len(b), 0.9)) for f, (_, b) in gt.items()}
        rows = [("A · detector as is", dets),
                ("B · only boxes that match a player", only_players),
                ("C · ground-truth boxes", gt_frames)]
        lines += ["", "Motion-only tracker (no ReID, no GMC, no memory):", "",
                  "| input boxes | IDF1 | MOTA | IDSW | ids created | re-entries |", "|---|---|---|---|---|---|"]
        for name, frames in rows:
            r = track(frames, seq.length, seq.fps, gt)
            lines.append(f"| {name} | {r['IDF1']:.1f} | {r['MOTA']:.1f} | {r['IDSW']} | {r['ids']} | {r['reentry']} |")
        lines.append("")

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
