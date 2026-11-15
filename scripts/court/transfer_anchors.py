"""Hand anchors of one clip carried to another clip filmed from the SAME spot (court chapter, C04).

Why it can work: a camera that only pans, tilts and zooms around a fixed point sees the world through one
homography per frame, so two of its frames, even from different clips, are related by an exact image-to-image
homography. It is found by matching the static background (doors, wall bars, scoreboard: SIFT + RANSAC); a
clicked court point of the source frame, sent through it, lands on the same court point in the target frame.

Every transferred frame must pass two checks, or it is not used:

* the image match is solid: at least ``--min-inliers`` background matches agree (a repetitive hall, with
  identical doors and windows, gives few and inconsistent matches);
* the image homography is a plausible zoom (0.25x-6x) and the transferred points are spread over the frame (a
  wrong match can squeeze them all into one spot);
* the court it gives lies on the painted lines, before AND after polishing: line agreement >= ``--min-agreement``
  (hand anchors on the same hall score 0.51-0.66 before, 0.58-0.70 after). Polishing alone is not trusted: on a
  floor painted with several courts it can pull a wrong court onto some other lines.

At most ``--max-anchors`` frames are kept, spread over the clip. No frame passing means no anchors: the clip
gets no court (and no minimap) rather than a guessed one. Strongly zoomed clips, which see little of the hall the
source clip saw, usually end there.

Output: ``<out>/anchors_<target>.json``, the format of ``annotate_anchors.py``, read by
``propagate_homographies.py`` (and by ``run_pipeline.py --anchors``, which calls this script by itself when the
anchors belong to another clip).

Usage (repo root):
    python scripts/court/transfer_anchors.py --anchors outputs/court_anchors/anchors_mamba_clip3.json \\
        --sequence mamba_clip5 --config outputs/pipeline/mamba_clip5/config.yaml --out outputs/court_anchors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.anchors import Anchor, AnchorFile, fit_anchor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=Path, required=True, help="the source clip's hand anchors")
    ap.add_argument("--sequence", required=True, help="the target clip")
    ap.add_argument("--config", default="configs/default.yaml", help="where the target clip's frames are")
    ap.add_argument("--source-config", default=None, help="where the source clip's frames are (default: --config)")
    ap.add_argument("--every", type=int, default=25, help="try one target frame every N")
    ap.add_argument("--min-inliers", type=int, default=60)
    ap.add_argument("--min-agreement", type=float, default=0.45)
    ap.add_argument("--max-anchors", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    src_af = AnchorFile.load(args.anchors)
    spec = src_af.spec
    src = Clip(src_af.sequence, args.source_config or args.config)
    dst = Clip(args.sequence, args.config)
    sift = cv2.SIFT_create(4000)

    def features(img):
        return sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)

    sources = [(a, *features(src.frame(a.frame))) for a in src_af.anchors]
    matcher = cv2.BFMatcher()
    found = []                                        # (inliers, target frame, Anchor, agreement, note)
    for f in range(1, dst.length + 1, args.every):
        img = dst.frame(f)
        kb, db = features(img)
        if db is None or len(kb) < 10:
            continue
        best = None
        for a, ka, da in sources:                    # the source anchor that matches this frame best
            pairs = [m for m, n in matcher.knnMatch(da, db, k=2) if m.distance < 0.75 * n.distance]
            if len(pairs) < args.min_inliers:
                continue
            A = np.float32([ka[m.queryIdx].pt for m in pairs])
            B = np.float32([kb[m.trainIdx].pt for m in pairs])
            Hi, inl = cv2.findHomography(A, B, cv2.RANSAC, 3.0)
            if Hi is not None and int(inl.sum()) >= args.min_inliers and (best is None or inl.sum() > best[0]):
                best = (int(inl.sum()), a, Hi)
        if best is None:
            continue
        n_in, a, Hi = best
        h, w = img.shape[:2]
        pts = {}
        for k, (x, y) in a.points.items():          # clicked points sent through the image homography
            q = cv2.perspectiveTransform(np.array([[[x, y]]], np.float64), Hi)[0, 0]
            if 0 <= q[0] < w and 0 <= q[1] < h:
                pts[k] = (float(q[0]), float(q[1]))
        zoom = float(np.sqrt(abs(np.linalg.det(Hi[:2, :2] / Hi[2, 2]))))
        spread = np.ptp(np.array(list(pts.values())), axis=0) if pts else np.zeros(2)
        if len(pts) < 4 or not 0.25 <= zoom <= 6.0 or spread[0] < 0.2 * w or spread[1] < 0.1 * h:
            continue
        anchor = Anchor(f, pts, note=f"from {src_af.sequence} frame {a.frame}, {n_in} background inliers")
        _, before, after = fit_anchor(img, anchor, spec, dst.people(f))
        agreement = min(before, after)
        print(f"frame {f}: {n_in} inliers with source frame {a.frame}, zoom {zoom:.2f}, {len(pts)} points, "
              f"line agreement {before:.2f} -> {after:.2f}"
              + ("" if agreement >= args.min_agreement else "  -> rejected"))
        if agreement >= args.min_agreement:
            found.append((n_in, f, anchor, agreement))

    # keep the best frame of each part of the clip, so the propagation starts from several places
    found.sort(key=lambda r: r[1])
    keep = []
    if found:
        bins = np.array_split(np.arange(len(found)), min(args.max_anchors, len(found)))
        keep = [max((found[i] for i in b), key=lambda r: r[0] * r[3]) for b in bins if len(b)]
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"anchors_{args.sequence}.json"
    if not keep:
        print(f"no frame of {args.sequence} passed the checks: no anchors written (the clip gets no court)")
        path.unlink(missing_ok=True)
        return
    path.write_text(json.dumps({
        "sequence": args.sequence, "league": src_af.league,
        "anchors": [{"frame": a.frame, "points": {k: list(v) for k, v in a.points.items()}, "note": a.note}
                    for _, _, a, _ in keep]}, indent=1))
    print(f"{len(keep)} anchors transferred -> {path}: frames {[r[1] for r in keep]}")


if __name__ == "__main__":
    main()
