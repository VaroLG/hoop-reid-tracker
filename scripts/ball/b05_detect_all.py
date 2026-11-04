"""B05 · Run the ball detector on EVERY frame of the clips and keep all candidates (the tracker's input).

B04 ran the detector only on the labelled frames (one in ten). A tracker needs every frame. Every box with
confidence >= ``--min-conf`` is saved, not only the best one: the tracker decides which candidate is the ball.

Output: ``<out>/<clip>.csv`` with frame, cx, cy, d (box size, pixels), conf. A clip already done is skipped.

Usage (repo root; CPU, ~20-30 min for the 4 test clips):
    python -u scripts/ball/b05_detect_all.py --weights weights/ball_yolo26s_1280.pt --out outputs/ball/dets_b03
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config

TEST_CLIPS = ["v_00HRwkvvjtQ_c001", "v_4r8QL_wglzQ_c001", "v_5ekaksddqrc_c003", "v_BgwzTUxJaeU_c012"]   # B01/B04


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--weights", default="weights/ball_yolo26s_1280.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-conf", type=float, default=0.05)
    ap.add_argument("--split", default="val")
    ap.add_argument("--clips", nargs="*", default=TEST_CLIPS)
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/dets_b03"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            [f"data.split={args.split}", "data.only=null"]))}
    args.out.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        dst = args.out / f"{clip}.csv"
        if dst.is_file():
            print(f"{clip}: already done")
            continue
        seq, lines = seqs[clip], ["frame,cx,cy,d,conf"]
        for f in range(1, seq.length + 1):
            r = model.predict(cv2.imread(str(seq.frame_path(f))), imgsz=args.imgsz, conf=args.min_conf,
                              verbose=False)[0]
            for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), strict=True):
                lines.append(f"{f},{(x1 + x2) / 2:.1f},{(y1 + y2) / 2:.1f},{((x2 - x1) + (y2 - y1)) / 2:.1f},{c:.4f}")
            if f % 200 == 0:
                print(f"{clip}: {f}/{seq.length}")
        tmp = dst.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + "\n")
        tmp.replace(dst)
        print(f"{clip}: {len(lines) - 1} candidates in {seq.length} frames")


if __name__ == "__main__":
    main()
