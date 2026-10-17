"""Convert SportsMOT basketball (MOT format) into a YOLO detection dataset.

Why fine-tune the detector?
  A COCO "person" model detects EVERYONE: referees, coaches, the crowd. In
  SportsMOT only players are annotated, so all those boxes become false
  positives and, worse, extra identities the tracker has to manage. A
  single-class "player" detector trained on the train split learns the
  annotation policy (on-court players only).

Consecutive frames are almost identical, so we keep 1 frame every ``--stride``
to avoid an over-sized, redundant training set. Images are symlinked, not
copied.

Usage:
    python scripts/prepare_yolo_dataset.py --stride 5
    yolo detect train data=data/yolo_basketball/data.yaml model=yolo11m.pt imgsz=1280 epochs=50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hooptrack.data.sportsmot import list_sequences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/SportsMOT/dataset")
    ap.add_argument("--splits-dir", default="data/SportsMOT/splits_txt")
    ap.add_argument("--out", type=Path, default=Path("data/yolo_basketball"))
    ap.add_argument("--stride", type=int, default=5)
    args = ap.parse_args()

    for split in ("train", "val"):
        img_dir, lbl_dir = args.out / "images" / split, args.out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for seq in list_sequences(args.root, split, "basketball", args.splits_dir):
            gt = seq.load_gt()
            for f in range(1, seq.length + 1, args.stride):
                stem = f"{seq.name}_{f:06d}"
                link = img_dir / f"{stem}{seq.im_ext}"
                if not link.exists():
                    os.symlink(seq.frame_path(f).resolve(), link)
                lines = []
                for x1, y1, x2, y2 in gt.get(f, (None, []))[1]:
                    # YOLO label: class cx cy w h, normalised to [0, 1]
                    cx, cy = (x1 + x2) / 2 / seq.width, (y1 + y2) / 2 / seq.height
                    w, h = (x2 - x1) / seq.width, (y2 - y1) / seq.height
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
                n += 1
        print(f"{split}: {n} images")

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: player\n"
    )
    print("Dataset config:", args.out / "data.yaml")


if __name__ == "__main__":
    main()
