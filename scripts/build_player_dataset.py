"""Build the "player" detection dataset (experiment 011) and pack it as ONE zip for Google Colab.

Why retrain the detector?
-------------------------
Our detector is YOLO26m trained on COCO: its class 0 is "person", so it fires on
EVERYONE — players, referees, coaches, bench, crowd. SportsMOT annotates only the
ten players on court, so every other person is a false positive, and we have been
removing them AFTER detection with hand-made rules (court mask 003/C06, referee
classifier 006). The state of the art (Deep-EIoU, MixSort, DiffMOT…) does the
opposite: it fine-tunes the detector on SportsMOT, so the detector LEARNS the
annotation policy ("a player is a person in a team jersey, on court, in the
game") from the whole image — jersey, position, pose, context — not from a
colour rule. Same class index 0, so nothing else in the pipeline changes.

Splits — no leakage
-------------------
SportsMOT basketball: 15 train clips (3 games), 15 val clips (4 games), test
labels not public. Our TRACKING results are measured on the val clips, so val
frames must never influence training, not even the choice of the best epoch.

    train  12 train clips                         -> the model learns from these
    val    3 train clips, one per game (held out) -> picks the best epoch (same arenas: it
                                                     measures fitting, not generalisation)
    test   the 15 val clips, 1 frame in 10        -> FINAL check on 4 arenas never seen;
                                                     only looked at after training

Only 3 games (arenas) in train is little variety: the "test" numbers tell us
whether that is enough. If not, the next step is adding other basketball data.

Why a stride?
-------------
At 25 fps consecutive frames are nearly identical: training on all of them costs
time and adds almost no information. 1 frame in 3 keeps ~4,100 training images
(motion varies enough in 3 frames); for the test 1 in 10 is plenty.

Output: data/player_det/{images,labels}/{train,val,test} + data.yaml, and
data/player_det.zip (images are already JPEG: the zip only stores them, it is fast).

Usage (repo root):
    python scripts/build_player_dataset.py                 # ~600–900 MB zip, a few minutes
    python scripts/build_player_dataset.py --stride 5      # smaller upload
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from hooptrack.data.sportsmot import list_sequences

# One held-out clip per training game (video id = game): the val split sees all three arenas.
HELD_OUT = ("v_-6Os86HzwCs_c003", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c010")


def yolo_lines(boxes, width: int, height: int) -> list[str]:
    """TLBR pixel boxes -> YOLO lines 'class cx cy w h' (normalised to [0, 1], clipped to the image)."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(width), x2)       # a player half out of frame: keep the visible part
        y1, y2 = max(0.0, y1), min(float(height), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:                      # fully outside: no box
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def export(seqs, split: str, out: Path, stride: int) -> int:
    img_dir, lbl_dir = out / "images" / split, out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for seq in seqs:
        gt = seq.load_gt()                                   # frame -> (ids, boxes)
        for f in range(1, seq.length + 1, stride):
            stem = f"{seq.name}_{f:06d}"                     # clip name in the file name: traceable
            dst = img_dir / f"{stem}{seq.im_ext}"
            if not dst.exists():
                shutil.copy2(seq.frame_path(f), dst)          # a real copy: it has to travel to Colab
            boxes = gt.get(f, (None, []))[1]
            # an EMPTY label file is meaningful: "no player here" (a close-up, the crowd) -> negatives
            (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines(boxes, seq.width, seq.height)))
            n += 1
    print(f"{split:5s}: {n:5d} images from {len(seqs)} clips")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/SportsMOT/dataset")
    ap.add_argument("--splits-dir", default="data/SportsMOT/splits_txt")
    ap.add_argument("--out", type=Path, default=Path("data/player_det"))
    ap.add_argument("--stride", type=int, default=3, help="1 frame in N for train/val")
    ap.add_argument("--test-stride", type=int, default=10, help="1 frame in N for the test split (SportsMOT val)")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    train_all = list_sequences(args.root, "train", "basketball", args.splits_dir)
    test = list_sequences(args.root, "val", "basketball", args.splits_dir)
    train = [s for s in train_all if s.name not in HELD_OUT]
    val = [s for s in train_all if s.name in HELD_OUT]
    if not train or not val or not test:
        raise SystemExit("SportsMOT basketball train/val not found under --root")

    export(train, "train", args.out, args.stride)
    export(val, "val", args.out, args.stride)
    export(test, "test", args.out, args.test_stride)

    # 'path' is rewritten by the notebook to wherever the zip is unpacked
    (args.out / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: player\n")

    if not args.no_zip:
        zpath = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:
            for p in sorted(args.out.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(args.out.parent))      # zip root = "player_det/"
        print(f"Packed {zpath} ({zpath.stat().st_size / 1e6:.0f} MB): upload it to Google Drive")


if __name__ == "__main__":
    main()
