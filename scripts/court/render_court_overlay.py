"""Draw the court of every frame on the video, plus a top-down inset (court chapter, C04).

Translucent court floor + painted lines projected with each frame's homography,
and the line agreement in the corner — a visual check that a number cannot give.
With ``--minimap``, every detected person's feet are drawn on a top-down court
plan (a preview of C06: positions in metres).

Usage:
    python scripts/court/render_court_overlay.py \
        --homographies outputs/court_homographies/v_00HRwkvvjtQ_c001/homographies.npz \
        --sequence v_00HRwkvvjtQ_c001 --out outputs/court_homographies/v_00HRwkvvjtQ_c001/court_overlay.mp4 --minimap
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.leagues import SPECS, lines


def plan_view(spec, scale: float = 12.0, pad: int = 10) -> tuple[np.ndarray, callable]:
    w, h = int(spec.length * scale) + 2 * pad, int(spec.width * scale) + 2 * pad
    img = np.full((h, w, 3), (60, 110, 160), np.uint8)

    def to_px(p):
        return (np.asarray(p) * scale + pad).astype(np.int32)

    for pl in lines(spec):
        cv2.polylines(img, [to_px(pl)], False, (255, 255, 255), 1, cv2.LINE_AA)
    return img, to_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--homographies", type=Path, required=True)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--minimap", action="store_true")
    args = ap.parse_args()

    z = np.load(args.homographies)
    spec = SPECS[str(z["league"])]
    clip = Clip(args.sequence, min_score=0.5)
    court = np.array([[0, 0], [spec.length, 0], [spec.length, spec.width], [0, spec.width]], float)
    plan, to_px = plan_view(spec)
    writer = None
    for f, H, a in zip(z["frames"], z["H"], z["agreement"], strict=True):
        img = clip.frame(int(f))
        layer = img.copy()
        poly = cv2.perspectiveTransform(court[None], H)[0].astype(np.int32)
        cv2.fillPoly(layer, [poly], (0, 200, 255))
        img = cv2.addWeighted(layer, 0.25, img, 0.75, 0)
        for pl in lines(spec):
            p = cv2.perspectiveTransform(pl[None].astype(np.float64), H)[0].astype(np.int32)
            cv2.polylines(img, [p], False, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"frame {f}  line agreement {a:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 3, cv2.LINE_AA)
        if args.minimap:
            m = plan.copy()
            b = clip.people(int(f))
            if b is not None and len(b):
                feet = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
                c = cv2.perspectiveTransform(feet[None].astype(np.float64), np.linalg.inv(H))[0]
                for (x, y), (px, py) in zip(c, to_px(c), strict=True):
                    inside = -0.5 <= x <= spec.length + 0.5 and -0.5 <= y <= spec.width + 0.5
                    if inside:
                        cv2.circle(m, (int(px), int(py)), 5, (0, 0, 255), -1, cv2.LINE_AA)
            hm, wm = m.shape[:2]
            y0, x0 = 60, img.shape[1] - wm - 20
            img[y0:y0 + hm, x0:x0 + wm] = m
        img = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])))
        if writer is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), 25, img.shape[1::-1])
        writer.write(img)
    writer.release()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
