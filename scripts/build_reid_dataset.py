"""Build the player re-identification dataset (experiment 012) and pack it as ONE zip for Google Colab.

What ReID has to learn here
---------------------------
The tracker compares an *appearance vector* of every detection with the vector
of every track (cosine distance). Today that vector comes from a ResNet-50
trained on ImageNet: it knows "a person in a white shirt", so two team-mates
are almost the same point. That is why crossings swap identities. A ReID model
trained on players learns the cues that DO tell team-mates apart: jersey
number, skin, hair, shoes, body shape, sleeves, knee pads.

Identities
----------
An identity is (clip, ground-truth id): SportsMOT ids are unique only inside a
clip. The same real player in two clips gets two identities, which is harmless
for training: what is learnt is "these crops are the same person, those are not".

Splits (no leakage, no model selection on the evaluation data)
--------------------------------------------------------------
    train   all 15 SportsMOT train clips (3 games), 1 frame in 3
    test    all 15 SportsMOT val clips (4 other games), 1 frame in 5 — the arenas where
            tracking is measured; used ONLY to report numbers after training

There is no validation split: the notebook trains a fixed recipe (number of
epochs, learning rate) taken from the literature (Deep-EIoU: OSNet, 60 epochs,
Adam 3e-4) instead of picking the best epoch, so nothing is tuned on test.

Crops are saved already resized to 128 x 256 (width x height, the standard ReID
input), so the zip is small and training reads them fast. File name:
``<pid>__<clip>__<frame>.jpg`` — the notebook gets the identity, the clip and
the time of every crop from the name, without an index file.

Usage (repo root):
    python scripts/build_reid_dataset.py            # ~40k train crops, ~200–300 MB zip
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

from hooptrack.data.sportsmot import list_sequences

W, H = 128, 256          # ReID input size (width, height): tall and narrow like a person


def export(seqs, out_dir: Path, stride: int, min_height: int, pid_start: int) -> tuple[int, int]:
    """Crop every ground-truth player every ``stride`` frames. Returns (crops, identities)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pids: dict[tuple[str, int], int] = {}
    n = 0
    for seq in seqs:
        gt = seq.load_gt()                                       # frame -> (ids, boxes TLBR)
        for f in range(1, seq.length + 1, stride):
            ids, boxes = gt.get(f, (np.zeros(0, int), np.zeros((0, 4))))
            if len(ids) == 0:
                continue
            img = cv2.imread(str(seq.frame_path(f)))
            h, w = img.shape[:2]
            for tid, (x1, y1, x2, y2) in zip(ids, boxes, strict=True):
                x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
                if y2 - y1 < min_height or x2 - x1 < 8:          # tiny, blurry crops teach noise
                    continue
                pid = pids.setdefault((seq.name, int(tid)), pid_start + len(pids))
                crop = cv2.resize(img[y1:y2, x1:x2], (W, H), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(out_dir / f"{pid:05d}__{seq.name}__{f:06d}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                n += 1
    return n, len(pids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/SportsMOT/dataset")
    ap.add_argument("--splits-dir", default="data/SportsMOT/splits_txt")
    ap.add_argument("--out", type=Path, default=Path("data/reid_players"))
    ap.add_argument("--stride", type=int, default=3, help="1 frame in N (train)")
    ap.add_argument("--test-stride", type=int, default=5, help="1 frame in N (test)")
    ap.add_argument("--min-height", type=int, default=50, help="skip players smaller than this (px)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    if args.out.exists():
        shutil.rmtree(args.out)                                  # identities are renumbered: start clean
    train = list_sequences(args.root, "train", "basketball", args.splits_dir)
    test = list_sequences(args.root, "val", "basketball", args.splits_dir)
    if not train or not test:
        raise SystemExit("SportsMOT basketball train/val not found under --root")

    n, p = export(train, args.out / "train", args.stride, args.min_height, pid_start=0)
    print(f"train: {n:6d} crops, {p:4d} identities, {len(train)} clips")
    n, p = export(test, args.out / "test", args.test_stride, args.min_height, pid_start=10000)
    print(f"test : {n:6d} crops, {p:4d} identities, {len(test)} clips")

    if not args.no_zip:
        zpath = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
            for f in sorted(args.out.rglob("*.jpg")):
                z.write(f, f.relative_to(args.out.parent))          # zip root = "reid_players/"
        print(f"Packed {zpath} ({zpath.stat().st_size / 1e6:.0f} MB): upload it to MyDrive/hooptrack/")


if __name__ == "__main__":
    main()
