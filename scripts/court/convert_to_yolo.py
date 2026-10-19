"""Convert the court-keypoint COCO export to Ultralytics YOLO-pose format (court chapter, C02).

Applies the three fixes decided in C01, in this order:

1. **Split by game** (not Roboflow's random split): no arena / camera appears in
   both train and validation, so validation measures generalisation honestly.
2. **Drop clamped labels**: floor labels that disagree with the image's own
   homography by > 15 px (points really off-screen, placed on the border) get
   visibility 0. The rest of the image is kept.
3. **Restore the 16:9 aspect**: the export stretched 1024 x 576 frames to 576 x 576.
   Images are resized back to 1024 x 576 and x coordinates scaled by 1024 / 576,
   so the model learns from frames shaped like SportsMOT's.

YOLO-pose label format (one line per object, all values normalised to [0, 1]):

    class  cx cy w h   x1 y1 v1   x2 y2 v2 ...   (33 keypoints)

v = 0 not labelled (x = y = 0), 1 labelled but occluded, 2 visible. The dataset
only uses 0 and 2.

Usage:
    python scripts/court/convert_to_yolo.py
    python scripts/court/convert_to_yolo.py --src data/court_keypoints/v22_coco --dst data/court_keypoints/yolo_v22
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from hooptrack.court import nba_template
from hooptrack.court.dataset import clamped_labels, flip_index, load_coco, split_by_game

YOLO_SPLIT = {"train": "train", "valid": "val", "test": "test"}   # Ultralytics folder names


def yolo_line(bbox: np.ndarray, kp: np.ndarray, w: int, h: int) -> str:
    """One label line; ``bbox`` = COCO [x, y, bw, bh] and ``kp`` (K, 3) in output pixels."""
    x, y, bw, bh = bbox
    x0, y0 = max(x, 0.0), max(y, 0.0)                         # clip the box to the image
    x1, y1 = min(x + bw, w), min(y + bh, h)
    vals = [0, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    for kx, ky, v in kp:
        vals += [kx / w, ky / h, int(v)] if v > 0 else [0.0, 0.0, 0]
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/court_keypoints/v22_coco"))
    ap.add_argument("--dst", type=Path, default=Path("data/court_keypoints/yolo_v22"))
    ap.add_argument("--width", type=int, default=1024, help="output width (height stays 576 -> 16:9)")
    ap.add_argument("--outlier-px", type=float, default=15.0)
    args = ap.parse_args()

    tpl = nba_template()
    names, recs, flip_pairs = load_coco(args.src)
    flip = flip_index(names, tpl)
    # Cross-check our geometric flip with the one shipped by the dataset (1-based pairs).
    shipped = {i: i for i in range(len(names))}
    for a, b in flip_pairs:
        shipped[a - 1], shipped[b - 1] = b - 1, a - 1
    assert flip == [shipped[i] for i in range(len(names))], "flip index disagrees with the dataset"

    split = split_by_game([r.game for r in recs])
    split_of = {g: s for s, games in split.items() for g in games}
    stats = Counter()
    for r in recs:
        out_split = YOLO_SPLIT[split_of[r.game]]
        img = cv2.imread(str(r.path(args.src)))
        h, w0 = img.shape[:2]
        sx = args.width / w0                                       # 1024 / 576
        img = cv2.resize(img, (args.width, h), interpolation=cv2.INTER_CUBIC)

        kp = r.kp.copy()
        for n in clamped_labels(r, names, tpl, args.outlier_px):   # fix 2 (in the original pixels)
            kp[names.index(n)] = 0
            stats["clamped_labels_dropped"] += 1
        kp[:, 0] *= sx                                             # fix 3
        bbox = r.bbox * np.array([sx, 1, sx, 1])

        stem = Path(r.file).stem
        (args.dst / "images" / out_split).mkdir(parents=True, exist_ok=True)
        (args.dst / "labels" / out_split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.dst / "images" / out_split / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (args.dst / "labels" / out_split / f"{stem}.txt").write_text(yolo_line(bbox, kp, args.width, h) + "\n")
        stats[f"images_{out_split}"] += 1

    data = {
        "path": ".",                      # resolved to an absolute path by train_keypoints.py
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": {0: "court"},
        "kpt_shape": [len(names), 3],
        "flip_idx": flip,
        "kpt_names": {0: names},          # documentation only; order = label order
    }
    (args.dst / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    (args.dst / "conversion.json").write_text(json.dumps({"split_by_game": split, **stats}, indent=2))
    print(dict(stats))
    print(f"Dataset ready: {args.dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
