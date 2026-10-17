"""Export player crops in Market-1501 layout to fine-tune a ReID model.

Off-the-shelf ReID models are trained on pedestrians in street scenes. In
basketball, team-mates wear IDENTICAL jerseys, so the model must learn finer
cues (jersey number, shoes, skin tone, hair, body shape). Fine-tuning on
crops from the train split teaches exactly that.

Identity label = (sequence, gt id): ids are only unique inside one clip.

Output (compatible with torchreid's Market1501-style custom datasets)::

    data/reid_basketball/
      bounding_box_train/<pid>_c1s1_<frame>.jpg
      query/ ...           # val split: 1 crop per (pid, camera-chunk)
      bounding_box_test/ ...

Usage:
    python scripts/export_reid_crops.py --stride 3 --min-height 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hooptrack.data.sportsmot import list_sequences


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/SportsMOT/dataset")
    ap.add_argument("--splits-dir", default="data/SportsMOT/splits_txt")
    ap.add_argument("--out", type=Path, default=Path("data/reid_basketball"))
    ap.add_argument("--stride", type=int, default=3, help="sample 1 frame every N")
    ap.add_argument("--min-height", type=int, default=60, help="skip tiny, blurry crops")
    args = ap.parse_args()

    pid_map: dict[tuple[str, int], int] = {}
    rng = np.random.default_rng(0)

    for split, folders in (("train", ["bounding_box_train"]), ("val", ["query", "bounding_box_test"])):
        for folder in folders:
            (args.out / folder).mkdir(parents=True, exist_ok=True)
        for seq in list_sequences(args.root, split, "basketball", args.splits_dir):
            gt = seq.load_gt()
            for f in range(1, seq.length + 1, args.stride):
                ids, boxes = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
                if len(ids) == 0:
                    continue
                img = cv2.imread(str(seq.frame_path(f)))
                for tid, (x1, y1, x2, y2) in zip(ids, boxes, strict=True):
                    if y2 - y1 < args.min_height:
                        continue
                    pid = pid_map.setdefault((seq.name, int(tid)), len(pid_map))
                    crop = img[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
                    if crop.size == 0:
                        continue
                    if split == "train":
                        folder = "bounding_box_train"
                    else:  # ~10 % of val crops become queries, rest is the gallery
                        folder = "query" if rng.random() < 0.1 else "bounding_box_test"
                    # Market naming: <pid>_c<cam>s<seq>_<frame>.jpg ; camera 1 for gallery, 2 for query
                    cam = 2 if folder == "query" else 1
                    cv2.imwrite(str(args.out / folder / f"{pid:05d}_c{cam}s1_{f:06d}.jpg"), crop)
        print(f"{split}: {len(pid_map)} identities so far")


if __name__ == "__main__":
    main()
