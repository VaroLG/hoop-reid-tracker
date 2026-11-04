"""B03 · YOLO dataset of the ball from the hand clicks of the TRAIN clips (``b02_label_ball.py --split train``).

Every labelled frame becomes one training image (the full 1280 x 720 frame) with one box (``visible``: centre = the
click, size = ``hooptrack.ball.labels.ball_diameter`` x ``--margin``) or none (ball not visible: a negative image,
which teaches the detector not to invent a ball). One class, ``ball``.

The last clip of each train game is held out as the YOLO ``val`` split (early stopping and the best epoch); the
SportsMOT validation clips are never used here (they are B04's test).

Output: ``--out`` (a zip to upload to ``MyDrive/hooptrack/``) with ``images/{train,val}``, ``labels/{train,val}`` and
``data.yaml``.

Usage (repo root):
    python scripts/ball/b03_build_ball_dataset.py --labels data/ball_labels/train.csv --out data/ball_det.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np

from hooptrack.ball.labels import ball_diameter, load_labels
from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config

HELD_OUT = ["v_-6Os86HzwCs_c009", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c013"]   # last clip of each train game


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--labels", type=Path, default=Path("data/ball_labels/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/ball_det.zip"))
    ap.add_argument("--margin", type=float, default=1.15, help="box side = estimated diameter x margin")
    ap.add_argument("--held-out", nargs="*", default=HELD_OUT)
    args = ap.parse_args()

    labels = load_labels(args.labels)
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=train", "data.only=null"]))}
    counts = {"train": [0, 0], "val": [0, 0]}                 # [images with a ball, images without]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:  # JPEGs do not compress further
        for clip, rows in labels.groupby("clip"):
            seq = seqs[clip]
            gt = seq.load_gt()
            split = "val" if clip in args.held_out else "train"
            for _, r in rows.iterrows():
                f = int(r["frame"])
                img = cv2.imread(str(seq.frame_path(f)))
                H, W = img.shape[:2]
                name = f"{clip}_{f:06d}"
                z.writestr(f"images/{split}/{name}.jpg", cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                line = ""
                if r["visible"]:
                    d = args.margin * ball_diameter(r["x"], r["y"], gt.get(f, (None, np.zeros((0, 4))))[1])
                    line = f"0 {r['x'] / W:.6f} {r['y'] / H:.6f} {d / W:.6f} {d / H:.6f}\n"
                z.writestr(f"labels/{split}/{name}.txt", line)
                counts[split][0 if line else 1] += 1
        z.writestr("data.yaml", "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n")
    for s, (a, b) in counts.items():
        print(f"{s}: {a + b} images ({a} with the ball, {b} without)")
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
