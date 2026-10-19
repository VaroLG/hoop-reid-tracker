"""Audit the public court-keypoint dataset before training on it (court chapter, C01).

A downloaded dataset is a hypothesis, not ground truth. Before spending GPU
hours we check four things, all with numbers:

1. **Content** — images per split, image size, how often each keypoint is labelled.
2. **Geometry of the labels** — does every keypoint name mean the same court
   point in every image? We fit a homography per image between the labels and
   the court template (``hooptrack.court.template``) with RANSAC and measure the
   reprojection error of each label. Small errors = consistent labels AND a
   correct template. Large, systematic errors = wrong template position, or a
   point that is not on the floor (the rims). Rare huge errors = labelling mistakes.
3. **Leakage** — Roboflow splits frames at random. Frames from the same game
   (same arena, same camera) in train AND valid make validation optimistic.
   We parse the game from the file name and count shared games.
4. **A leak-free split** — a deterministic split BY GAME, saved as JSON, used by
   the training step.

Usage:
    python scripts/court/audit_dataset.py --data data/court_keypoints/v22_coco \
        --out experiments/court/C01_dataset_audit
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import cv2
import numpy as np

from hooptrack.court import nba_template
from hooptrack.court.dataset import SPLITS, fit_homography, label_errors, load_coco, split_by_game


def reprojection_errors(rec, names, template, ransac_px):
    """{name: error_px} for every labelled point (rims included), or None if < 4 floor labels."""
    H = fit_homography(rec, names, template, ransac_px)
    return None if H is None else label_errors(rec, names, template, H)


def draw_template(template, path: Path, scale: int = 18) -> None:
    """Top-down court plan with every keypoint name: the reference for the README."""
    pad = 30
    W, H = int(template.length * scale) + 2 * pad, int(template.width * scale) + 2 * pad
    img = np.full((H, W, 3), (205, 225, 240), np.uint8)
    to_px = lambda p: (int(pad + p[0] * scale), int(pad + p[1] * scale))   # noqa: E731
    cv2.rectangle(img, to_px((0, 0)), to_px((template.length, template.width)), (60, 60, 60), 2)
    cv2.line(img, to_px((template.length / 2, 0)), to_px((template.length / 2, template.width)), (60, 60, 60), 2)
    for n, p in template.points.items():
        col = (40, 40, 220) if n in template.non_floor else (200, 90, 20)
        cv2.circle(img, to_px(p), 6, col, -1)
        cv2.putText(img, n, (to_px(p)[0] + 7, to_px(p)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.putText(img, "blue: floor keypoints   red: rims (not on the floor)", (pad, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/court_keypoints/v22_coco"))
    ap.add_argument("--out", type=Path, default=Path("experiments/court/C01_dataset_audit"))
    ap.add_argument("--ransac-px", type=float, default=8.0)
    ap.add_argument("--outlier-px", type=float, default=15.0, help="label error that counts as a mistake")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tpl = nba_template()

    names, recs, _ = load_coco(args.data)
    unknown = [n for n in names if n not in tpl.points]
    if unknown:
        raise SystemExit(f"Keypoints without a template position: {unknown}")

    # 1. content -------------------------------------------------------------------
    content = {s: sum(r.split == s for r in recs) for s in SPLITS}
    sizes = collections.Counter(r.size for r in recs)
    vis = np.array([r.kp[:, 2] > 0 for r in recs])
    per_kp_labelled = {n: int(c) for n, c in zip(names, vis.sum(0), strict=True)}

    # 2. geometry ------------------------------------------------------------------
    errs = collections.defaultdict(list)
    outliers, fitted = [], 0
    for r in recs:
        out = reprojection_errors(r, names, tpl, args.ransac_px)
        if out is None:
            continue
        fitted += 1
        for n, e in out.items():
            errs[n].append(float(e))
            if e > args.outlier_px and n not in tpl.non_floor:
                outliers.append({"file": r.file, "split": r.split, "keypoint": n, "error_px": round(float(e), 1)})
    per_kp_err = {n: {"n": len(v), "median_px": round(float(np.median(v)), 2),
                      "p90_px": round(float(np.percentile(v, 90)), 2)} for n, v in errs.items()}

    # 3. leakage -------------------------------------------------------------------
    games = {s: {r.game for r in recs if r.split == s} for s in SPLITS}
    leakage = {
        "games_per_split": {s: len(g) for s, g in games.items()},
        "valid_games_also_in_train": len(games["valid"] & games["train"]),
        "test_games_also_in_train": len(games["test"] & games["train"]),
    }

    # 4. game-level split ------------------------------------------------------------
    gsplit = split_by_game([r.game for r in recs])
    counts = {s: sum(r.game in set(g) for r in recs) for s, g in gsplit.items()}

    report = {
        "dataset": str(args.data), "images": content, "image_sizes": {str(k): v for k, v in sizes.items()},
        "keypoints": names, "labelled_per_keypoint": per_kp_labelled,
        "homography_fitted_images": fitted, "reprojection_error": per_kp_err,
        "label_outliers": {"threshold_px": args.outlier_px, "count": len(outliers), "items": outliers},
        "leakage": leakage,
        "game_split": {"games": gsplit, "images": counts},
    }
    (args.out / "audit.json").write_text(json.dumps(report, indent=2))
    draw_template(tpl, args.out / "court_template.png")

    print(f"images: {content}   sizes: {dict(sizes)}")
    print(f"homography fitted on {fitted}/{len(recs)} images")
    print("keypoint   labelled   median err   p90 err   (px)")
    for n in names:
        e = per_kp_err.get(n, {})
        flag = "  <- not on the floor" if n in tpl.non_floor else ""
        print(f"   {n}       {per_kp_labelled[n]:4d}       {e.get('median_px', float('nan')):5.1f}      "
              f"{e.get('p90_px', float('nan')):6.1f}{flag}")
    print(f"label outliers (> {args.outlier_px} px): {len(outliers)}")
    print(f"leakage: {leakage}")
    print(f"game split (images): {counts}")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
