"""Smoke test for CourtSegmenter using the trained C06 weights.

Usage:
    python scripts/court/seg_smoke_test.py [frame.png]

Validates end-to-end:
  1. load trained segmentation weights (outputs/court_runs/c06_yolo26n_seg/weights/best.pt)
  2. run inference on a real basketball frame -> court mask
  3. run feet_inside / SegmentationCourtFilter on dummy boxes
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hooptrack.court.segment import CourtSegmenter, feet_inside  # noqa: E402

WEIGHTS = REPO / "outputs" / "court_runs" / "c06_yolo26n_seg" / "weights" / "best.pt"


def find_sample() -> Path:
    for root in (REPO / "data" / "court_seg", REPO / "data" / "SportsMOT" / "dataset"):
        if root.is_dir():
            for pat in ("*.jpg", "*.jpeg", "*.png"):
                hits = sorted(root.rglob(pat))
                if hits:
                    return hits[0]
    raise FileNotFoundError("no sample image found under data/")


def main() -> None:
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_sample()
    print(f"sample frame: {img_path}")
    frame = cv2.imread(str(img_path))
    assert frame is not None, img_path
    print(f"frame shape: {frame.shape} dtype={frame.dtype}")

    seg = CourtSegmenter(
        str(WEIGHTS), imgsz=512, conf=0.25, device="cpu", convex=True, smooth=0.0
    )
    mask = seg(frame)
    print(f"mask: shape={mask.shape} dtype={mask.dtype} min={mask.min()} max={mask.max()}")
    uniq, counts = np.unique(mask, return_counts=True)
    print(f"unique values: {dict(zip(uniq.tolist(), counts.tolist()))}")
    court_px = int((mask > 0).sum())
    print(f"court pixels: {court_px} ({100 * court_px / mask.size:.1f}%)")

    h, w = mask.shape
    boxes = np.array(
        [
            [w * 0.40, h * 0.80, w * 0.50, h * 0.95],  # feet near bottom edge
            [w * 0.10, h * 0.10, w * 0.20, h * 0.20],  # top-left corner
        ],
        dtype=float,
    )
    keep = feet_inside(mask, boxes, margin=0.05)
    print(f"feet_inside keep={keep.tolist()} for boxes={boxes.tolist()}")

    # full integration path used by the tracker (court_source: segmentation)
    from hooptrack.integrations.court_filter import SegmentationCourtFilter  # noqa: E402

    flt = SegmentationCourtFilter()
    scores = np.array([0.9, 0.8])
    boxes_out, scores_out = flt(frame, boxes, scores)
    print(f"filter: kept {len(boxes_out)}/{len(boxes)} boxes, scores={scores_out.tolist()}")
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
