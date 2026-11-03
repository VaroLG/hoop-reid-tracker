"""B01 · Does a COCO detector already see the basketball in broadcast frames? (ball chapter, first gate)

Why
---
SportsMOT has no ball labels, so a ball detector needs labels from somewhere. COCO has a ``sports ball`` class
(id 32) and our first baseline (001) already used a COCO YOLO26m. If it finds the ball in a good share of frames
with few false alarms, it can PSEUDO-LABEL the SportsMOT training frames (filtered in time, checked by a person),
which is much cheaper than labelling from scratch. If it does not, the labels must come from a person.

What it does
------------
Runs the COCO model, ``sports ball`` class only, on every ``--stride``-th frame of the chosen clips, at two input
sizes (``--imgsz``, default 640 and 1280: the ball is ~10-20 px wide in a 1280 x 720 frame, so at 640 px it shrinks
to 5-10 px, near the limit of what a detector sees). For each size it reports, per clip:

* ``frames_with_ball``: share of sampled frames with at least one detection above ``--conf``;
* ``dets_per_frame``: detections per frame (more than 1 = false alarms: there is one ball);
* ``conf_median``: median confidence of the best detection of each frame.

and saves, for a person to judge precision, ``sheet_<imgsz>.jpg``: a grid of the best detection of randomly chosen
frames, each a 96 x 96 crop zoomed around the box with its confidence. Recall cannot be measured without labels;
the sheet and the share of frames say whether this is worth pursuing.

Usage (repo root):
    python scripts/ball/b01_probe_coco_ball.py --out outputs/ball/b01_probe
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config

SPORTS_BALL = 32                       # COCO class id of "sports ball"
DEFAULT_CLIPS = ["v_00HRwkvvjtQ_c001", "v_4r8QL_wglzQ_c001", "v_5ekaksddqrc_c003", "v_BgwzTUxJaeU_c012"]  # 1 per arena


def crop_tile(img: np.ndarray, box: np.ndarray, conf: float, size: int = 96, context: float = 3.0) -> np.ndarray:
    """Square crop centred on the box, ``context`` times its size (at least 24 px), resized to ``size``."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(12.0, context * max(box[2] - box[0], box[3] - box[1]) / 2)
    H, W = img.shape[:2]
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(W, cx + half)), int(min(H, cy + half))
    tile = cv2.resize(img[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, f"{conf:.2f}", (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--weights", default="yolo26m.pt", help="COCO-pretrained YOLO (001 baseline)")
    ap.add_argument("--clips", nargs="*", default=DEFAULT_CLIPS)
    ap.add_argument("--stride", type=int, default=10, help="every n-th frame")
    ap.add_argument("--imgsz", nargs="*", type=int, default=[640, 1280])
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--sheet", type=int, default=64, help="detections shown on each sheet")
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/b01_probe"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    cfg = load_config(args.config, [])
    seqs = {s.name: s for s in _sequences(apply_overrides(cfg, ["data.only=null"]))}
    args.out.mkdir(parents=True, exist_ok=True)
    rows, rng = [], random.Random(0)
    for imgsz in args.imgsz:
        tiles = []
        for clip in args.clips:
            seq = seqs[clip]
            n_frames, n_hit, n_dets, best_conf = 0, 0, 0, []
            for f in range(1, seq.length + 1, args.stride):
                img = cv2.imread(str(seq.frame_path(f)))
                r = model.predict(img, imgsz=imgsz, conf=args.conf, classes=[SPORTS_BALL], verbose=False)[0]
                boxes, confs = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
                n_frames += 1
                n_dets += len(boxes)
                if len(boxes):
                    n_hit += 1
                    k = int(confs.argmax())
                    best_conf.append(float(confs[k]))
                    tiles.append(crop_tile(img, boxes[k], float(confs[k])))
            rows.append({"imgsz": imgsz, "clip": clip, "frames": n_frames,
                         "frames_with_ball": round(n_hit / n_frames, 3),
                         "dets_per_frame": round(n_dets / n_frames, 2),
                         "conf_median": round(float(np.median(best_conf)), 3) if best_conf else np.nan})
            print(rows[-1])
        if tiles:                                           # random sample of best detections, 8 per row
            pick = rng.sample(tiles, min(args.sheet, len(tiles)))
            pick += [np.zeros_like(pick[0])] * (-len(pick) % 8)
            sheet = np.vstack([np.hstack(pick[i:i + 8]) for i in range(0, len(pick), 8)])
            cv2.imwrite(str(args.out / f"sheet_{imgsz}.jpg"), sheet)
    df = pd.DataFrame(rows)
    df.to_csv(args.out / "probe.csv", index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved {args.out / 'probe.csv'} and the sheets sheet_<imgsz>.jpg")


if __name__ == "__main__":
    main()
