"""B08 · Training data for fine-tuning WASB on our hand clicks: every clicked train frame with its neighbours.

WASB looks at 3 consecutive frames and outputs one heatmap per frame. We clicked one frame in ten (B02), so each
clicked frame ``f`` gives three training windows, (f-2, f-1, f), (f-1, f, f+1) and (f, f+1, f+2), with the click as
the target of the window's frame ``f`` only (the other two outputs are not labelled and are left out of the loss).
Frames where the ball is not visible are kept: an empty target teaches the network not to answer.

Saved at WASB's input size (512 x 288, a plain resize of 1280 x 720): ``frames/<clip>/<frame>.jpg`` for every frame
f-2 .. f+2 of every clicked frame, and ``labels.csv`` (clip, frame, x, y, visible, split) with the click in 512 x 288
pixels. The last clip of each train game is the ``val`` split (best epoch); the SportsMOT validation clips are never
used (B04's test).

Usage (repo root):
    python scripts/ball/b08_export_wasb_data.py --out data/wasb_ft.zip
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import cv2

from hooptrack.ball.labels import load_labels
from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config

HELD_OUT = ["v_-6Os86HzwCs_c009", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c013"]
W, H = 512, 288


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--labels", type=Path, default=Path("data/ball_labels/train.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/wasb_ft.zip"))
    args = ap.parse_args()
    labels = load_labels(args.labels)
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=train", "data.only=null"]))}
    rows = ["clip,frame,x,y,visible,split"]
    n_img = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for clip, lab in labels.groupby("clip"):
            seq = seqs[clip]
            img0 = cv2.imread(str(seq.frame_path(1)))
            sx, sy = W / img0.shape[1], H / img0.shape[0]
            need = sorted({g for f in lab["frame"] for g in range(int(f) - 2, int(f) + 3) if 1 <= g <= seq.length})
            for g in need:
                img = cv2.resize(cv2.imread(str(seq.frame_path(g))), (W, H), interpolation=cv2.INTER_AREA)
                jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 93])[1].tobytes()
                z.writestr(f"frames/{clip}/{g:06d}.jpg", jpg)
                n_img += 1
            split = "val" if clip in HELD_OUT else "train"
            for _, r in lab.iterrows():
                if r["visible"]:
                    rows.append(f"{clip},{int(r['frame'])},{r['x'] * sx:.2f},{r['y'] * sy:.2f},1,{split}")
                else:
                    rows.append(f"{clip},{int(r['frame'])},,,0,{split}")
        z.writestr("labels.csv", "\n".join(rows) + "\n")
    print(f"{len(rows) - 1} clicked frames, {n_img} images; saved {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
