"""Turn the hand-labelled train players (041) into training stubs for the jersey reader, in E-BARD's format.

``scripts/label_jersey_numbers.py`` gives one number per ground-truth player of the train clips. Here every
labelled player becomes up to ``--stubs`` stubs of ``--crops`` crops (the tallest boxes no other player overlaps,
in consecutive stretches of the clip), like the 16-crop stubs of E-BARD, so the Colab notebook can mix both
datasets with the same code. A player labelled ``none`` (no number readable) gives stubs labelled ``nan``.

One clip per game is held out (``--val-clips``): its stubs set the confidence threshold in Colab, so the threshold
is not chosen on stubs the model was trained on. The validation clips of SportsMOT are never used here.

Output: ``--out`` (a zip to upload to Google Drive, ``MyDrive/hooptrack/``) with ``stubs/<clip>_<id>_<k>/*.jpg``
and ``ours_train.txt`` / ``ours_val.txt`` (one line per stub: ``<folder> <n crops> <label>``, as E-BARD).

Usage (repo root):
    python scripts/export_jersey_stubs.py --labels data/jersey_labels/train_numbers.csv `
        --out data/jersey_labels/sportsmot_train_stubs.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.tracking.matching import coverage_matrix

VAL_CLIPS = ["v_-6Os86HzwCs_c009", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c013"]   # the last clip of each game


def clean_tall_boxes(gt: dict) -> dict[int, list[tuple[int, np.ndarray]]]:
    """{gt id: [(frame, box)]} for boxes no other player covers >= 10 % and at least as tall as his median box."""
    out: dict[int, list] = {}
    for f, (ids, boxes) in sorted(gt.items()):
        boxes = np.asarray(boxes, float)
        cov = coverage_matrix(boxes).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
        for i, b, c in zip(ids, boxes, cov, strict=True):
            if c < 0.1:
                out.setdefault(int(i), []).append((int(f), b))
    for i, lst in out.items():
        med = np.median([b[3] - b[1] for _, b in lst])
        out[i] = [(f, b) for f, b in lst if b[3] - b[1] >= med]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--labels", type=Path, default=Path("data/jersey_labels/train_numbers.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/jersey_labels/sportsmot_train_stubs.zip"))
    ap.add_argument("--stubs", type=int, default=4, help="stubs per player")
    ap.add_argument("--crops", type=int, default=16, help="crops per stub (E-BARD: 16)")
    ap.add_argument("--step", type=int, default=3, help="frames between two crops of a stub")
    ap.add_argument("--val-clips", nargs="*", default=VAL_CLIPS)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels, dtype=str)
    labels["gt_id"] = labels.gt_id.astype(int)
    cfg = load_config(args.config, [])
    seqs = {s.name: s for s in _sequences(apply_overrides(cfg, ["data.split=train", "data.only=null"]))}
    lines = {"ours_train.txt": [], "ours_val.txt": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for clip, rows in labels.groupby("clip"):
            seq, split = seqs[clip], "ours_val.txt" if clip in args.val_clips else "ours_train.txt"
            boxes = clean_tall_boxes(seq.load_gt())
            images: dict[int, np.ndarray] = {}
            for _, r in rows.iterrows():
                lst = boxes.get(r.gt_id, [])[::args.step]
                if len(lst) < 4:
                    continue
                label = "nan" if r.number == "none" else r.number
                n = min(args.stubs, max(1, len(lst) // args.crops))
                for k, chunk in enumerate(np.array_split(np.arange(len(lst)), n)):
                    chunk = chunk[:args.crops]
                    folder = f"stubs/{clip}_{r.gt_id}_{k}"
                    for j in chunk:
                        f, (x1, y1, x2, y2) = lst[j]
                        if f not in images:
                            images = {f: cv2.imread(str(seq.frame_path(f)))}      # one frame in memory
                        img = images[f]
                        H, W = img.shape[:2]
                        crop = img[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))]
                        ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        z.writestr(f"{folder}/{folder.split('/')[-1]}_{f}.jpg", buf.tobytes())
                    lines[split].append(f"{folder} {len(chunk)} {label}")
        for name, ls in lines.items():
            z.writestr(name, "\n".join(ls) + "\n")
    for name, ls in lines.items():
        print(f"{name}: {len(ls)} stubs ({sum(s.endswith(' nan') for s in ls)} 'nan')")
    print(f"saved {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
