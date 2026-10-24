"""Evaluate the C06 court segmentation against the colour mask (003) and the automatic homography (C05/009).

Three questions, three measurements:

1. **Is the mask the court?**  IoU with the true court area, on frames whose
   true court is known and that were NEVER used in training:
   * test arenas of the training sources (DeepSport test arenas, v22 test games);
   * the four SportsMOT validation clips: NCAA (C04 hand anchors) and the three
     FIBA arenas of 009 (hand anchors + short propagation, or the court polygon
     drawn by hand), as PNG masks written by ``c06_annotate_eval.py``.
2. **Does it keep the players?**  Share of ground-truth player boxes whose feet
   are inside the mask (SportsMOT labels only the ten players), every
   ``--stride`` frames of the four validation clips.
3. **Does it remove the others?**  Share of detections (score >= 0.5) that match
   no player (bench, press, crowd, referees) and fall outside the mask.

Usage:
    python scripts/court/c06_eval.py --weights outputs/court_runs/c06_yolo26n_seg/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.leagues import SPECS
from hooptrack.court.masks import court_mask, mask_iou
from hooptrack.court.segment import CourtSegmenter, feet_inside
from hooptrack.detection.filters import CourtMask
from hooptrack.tracking.matching import iou_matrix

VAL_CLIPS = ["v_00HRwkvvjtQ_c001", "v_5ekaksddqrc_c001", "v_BgwzTUxJaeU_c008", "v_4r8QL_wglzQ_c001"]


def load_h(path) -> tuple[dict, object]:
    p = Path(path)
    if not p.is_file():
        return {}, None
    z = np.load(p)
    league = str(z["league"]) if "league" in z.files else "NCAA"
    return dict(zip(z["frames"].tolist(), z["H"], strict=True)), SPECS[league]


def colour_mask(cm: CourtMask, img, boxes) -> np.ndarray:
    """Experiment 003's court polygon as a mask (no polygon = no filtering = whole frame)."""
    hull = cm.estimate_hull(img, boxes)
    m = np.zeros(img.shape[:2], np.uint8)
    if hull is None:
        m[:] = 1
    else:
        cv2.fillPoly(m, [hull], 1)
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dataset", type=Path, default=Path("data/court_seg"))
    ap.add_argument("--gt-dir", type=Path, default=Path("outputs/court_eval"))
    ap.add_argument("--auto-dir", type=Path, default=Path("outputs/court_auto_edges"))
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--margin", type=float, default=0.05, help="feet may be outside by this share of box height")
    ap.add_argument("--out", type=Path, default=Path("experiments/court/C06_court_segmentation"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seg = CourtSegmenter(args.weights, args.imgsz, args.conf)
    report = {"weights": args.weights}

    # 1a. test splits of the training sources
    for src in ("deepsport", "v22"):
        d = args.dataset / src / "images" / "test"
        if not d.is_dir():
            continue
        ious = []
        for p in sorted(d.glob("*.jpg")):
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            gt = np.zeros((h, w), np.uint8)
            txt = (args.dataset / src / "labels" / "test" / f"{p.stem}.txt").read_text().split()
            if len(txt) > 6:
                poly = (np.array(txt[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(gt, [poly], 1)
            ious.append(mask_iou(seg(img), gt))
        report[f"test_{src}"] = {"images": len(ious), "iou_mean": round(float(np.mean(ious)), 3),
                                 "iou_p10": round(float(np.percentile(ious, 10)), 3)}
        print(src, report[f"test_{src}"])

    # 1b + 2 + 3. SportsMOT validation clips
    sheet = []
    for seq in VAL_CLIPS:
        clip = Clip(seq)
        gt_files = sorted((args.gt_dir / seq).glob("*.png"))
        auto_h, auto_spec = load_h(args.auto_dir / seq / "homographies.npz")
        cm, cm2 = CourtMask(), CourtMask()
        iou = {"colour": [], "auto_homography": [], "segmentation": []}
        for gf in gt_files:
            f = int(gf.stem)
            img = clip.frame(f)
            boxes = clip.people(f)
            boxes = np.zeros((0, 4)) if boxes is None else boxes
            gt = cv2.imread(str(gf), cv2.IMREAD_GRAYSCALE)
            col = colour_mask(cm, img, boxes)
            auto = court_mask(auto_h[f], auto_spec, img.shape[1::-1]) if f in auto_h else col   # 009 fallback
            pred = seg(img)
            iou["colour"].append(mask_iou(col, gt))
            iou["auto_homography"].append(mask_iou(auto, gt))
            iou["segmentation"].append(mask_iou(pred, gt))
            if len(sheet) < 16 and (len(iou["segmentation"]) - 1) % 4 == 0:
                v = img.copy()
                for m, c in ((gt, (0, 255, 0)), (col, (0, 255, 255)), (pred, (0, 0, 255))):
                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(v, cnts, -1, c, 3)
                cv2.putText(v, f"{seq} {f}  IoU seg {iou['segmentation'][-1]:.2f} colour {iou['colour'][-1]:.2f}",
                            (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                sheet.append(cv2.resize(v, (640, 360)))
        # players kept / others removed, every --stride frames
        gtb = clip.seq.load_gt()
        keep = {k: np.zeros(4, int) for k in ("colour", "segmentation")}
        for f in range(1, clip.length + 1, args.stride):
            img = clip.frame(f)
            b, s = clip.det(img, frame_idx=f)
            b = b[s >= 0.5]
            g = np.asarray(gtb.get(f, (None, np.zeros((0, 4))))[1], float).reshape(-1, 4)
            pred = seg(img)
            masks = {"colour": cm2.keep_mask(img, b) if len(b) else np.zeros(0, bool),
                     "segmentation": feet_inside(pred, b, args.margin)}
            player = (iou_matrix(b, g).max(1) >= 0.5) if len(g) and len(b) else np.zeros(len(b), bool)
            for k, kp in masks.items():
                keep[k] += [(kp & player).sum(), player.sum(), (~kp & ~player).sum(), (~player).sum()]
        report[seq] = {
            "gt_frames": len(iou["segmentation"]),
            "iou_mean": {k: round(float(np.mean(v)), 3) if v else None for k, v in iou.items()},
            "players_kept_%": {k: round(100 * r[0] / max(r[1], 1), 2) for k, r in keep.items()},
            "non_players_removed_%": {k: round(100 * r[2] / max(r[3], 1), 1) for k, r in keep.items()},
        }
        print(seq, json.dumps(report[seq]))
    (args.out / "eval.json").write_text(json.dumps(report, indent=2))
    if sheet:
        while len(sheet) % 2:
            sheet.append(np.zeros_like(sheet[0]))
        grid = np.vstack([np.hstack(sheet[i:i + 2]) for i in range(0, len(sheet), 2)])
        cv2.imwrite(str(args.out / "eval_sheet.jpg"), grid)
    print(f"saved {args.out / 'eval.json'} (green = true court, yellow = colour mask, red = segmentation)")


if __name__ == "__main__":
    main()
