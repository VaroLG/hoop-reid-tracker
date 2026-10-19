"""Evaluate the court-keypoint detector on what matters for a homography (court chapter, C02).

Ultralytics reports pose mAP (OKS), a metric designed for human joints. For a
court we care about three other things, all measured on the TEST split (games
never seen in training):

1. **Keypoint error in metres on the court.** Each predicted point is mapped to
   the court with the homography fitted on that image's ground-truth labels and
   compared with where the template says it is. "12 px" means little; "0.3 m"
   is directly what a top-down map will inherit.
2. **Homography quality.** A homography is fitted from the PREDICTED points only
   (RANSAC); the ground-truth labels are then mapped to the court with it.
   Median error per image, and the share of images where it is usable (< 0.5 m).
3. **Coverage.** Share of labelled points the model finds (confidence >= thr).

Optionally (``--frames``), the court is drawn through the predicted homography
on SportsMOT frames: a visual check of NCAA / NBA / FIBA courts the model never
saw in training.

Usage:
    python scripts/court/eval_keypoints.py --weights outputs/court_runs/c02_yolo26n_pose/weights/best.pt
    python scripts/court/eval_keypoints.py --weights ... --frames v_00HRwkvvjtQ_c001:300 v_-6Os86HzwCs_c009:200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from hooptrack.court import nba_template
from hooptrack.court.homography import fit_homography
from hooptrack.court.template import nba_court_lines


def read_label(path: Path, w: int, h: int, k: int) -> np.ndarray:
    """(K, 3) keypoints in pixels from a YOLO-pose label line."""
    v = np.array(path.read_text().split(), float)
    kp = v[5:5 + 3 * k].reshape(k, 3)
    kp[:, 0] *= w
    kp[:, 1] *= h
    return kp


def predict(model, img: np.ndarray, imgsz: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Keypoints (K, 2) and confidences (K,) of the most confident court, or None."""
    r = model.predict(img, imgsz=imgsz, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return None
    i = int(r.boxes.conf.argmax())
    return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()


def draw_court(img: np.ndarray, hom, names, xy, conf, thr) -> np.ndarray:
    out = img.copy()
    if hom is not None:
        for line in nba_court_lines():
            p = hom.to_image(line).astype(np.int32)
            cv2.polylines(out, [p], False, (255, 255, 0), 2, cv2.LINE_AA)
    for n, (x, y), c in zip(names, xy, conf, strict=True):
        if c >= thr:
            cv2.circle(out, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(out, n, (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/court_keypoints/yolo_v22/data.yaml"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5, help="keypoint confidence threshold")
    ap.add_argument("--good-m", type=float, default=0.5, help="an image's homography is 'usable' below this error")
    ap.add_argument("--frames", nargs="*", default=[], help="SportsMOT frames to visualise, as <sequence>:<frame>")
    ap.add_argument("--sportsmot", type=Path, default=Path("data/SportsMOT/dataset"))
    ap.add_argument("--out", type=Path, default=Path("experiments/court/C02_keypoint_detector"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tpl = nba_template()
    data = yaml.safe_load(args.data.read_text())
    names = data["kpt_names"][0]
    k = len(names)
    model = YOLO(str(args.weights))
    root = args.data.parent

    kp_err_m = {n: [] for n in names}                # 1. per-keypoint error in metres
    img_err_m, found, labelled, usable, no_h = [], 0, 0, 0, 0
    examples = []
    images = sorted((root / "images" / args.split).glob("*.jpg"))
    for path in images:
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gt = read_label(root / "labels" / args.split / f"{path.stem}.txt", w, h, k)
        gt_h = fit_homography(names, gt[:, :2], gt[:, 2] / 2, tpl)          # vis 2 -> conf 1
        pred = predict(model, img, args.imgsz)
        if pred is None or gt_h is None:
            no_h += 1
            continue
        xy, conf = pred
        vis = gt[:, 2] > 0
        labelled += int(vis.sum())
        found += int((vis & (conf >= args.conf)).sum())
        # 1. predicted points -> court through the GROUND-TRUTH homography
        court_pred = gt_h.to_court(xy)
        for i, n in enumerate(names):
            if vis[i] and conf[i] >= args.conf and n not in tpl.non_floor:
                kp_err_m[n].append(float(np.linalg.norm(court_pred[i] - tpl.points[n])))
        # 2. homography from PREDICTIONS, evaluated on the ground-truth labels
        ph = fit_homography(names, xy, conf, tpl, args.conf)
        if ph is None:
            no_h += 1
            continue
        floor = [i for i in range(k) if vis[i] and names[i] not in tpl.non_floor]
        court_gt = ph.to_court(gt[floor, :2])
        e = float(np.median(np.linalg.norm(court_gt - np.array([tpl.points[names[i]] for i in floor]), axis=1)))
        img_err_m.append(e)
        usable += e < args.good_m
        if len(examples) < 6 and len(img_err_m) % 15 == 1:
            examples.append(cv2.resize(draw_court(img, ph, names, xy, conf, args.conf), (640, 360)))

    report = {
        "weights": str(args.weights), "split": args.split, "images": len(images), "conf_thr": args.conf,
        "keypoint_coverage_%": round(100 * found / max(labelled, 1), 1),
        "keypoint_error_m": {n: {"n": len(v), "median": round(float(np.median(v)), 3),
                                 "p90": round(float(np.percentile(v, 90)), 3)} for n, v in kp_err_m.items() if v},
        "homography": {
            "images_with_H": len(img_err_m), "images_without_H": no_h,
            "median_error_m": round(float(np.median(img_err_m)), 3) if img_err_m else None,
            "p90_error_m": round(float(np.percentile(img_err_m, 90)), 3) if img_err_m else None,
            f"usable_%(<{args.good_m} m)": round(100 * usable / max(len(images), 1), 1),
        },
    }
    all_kp = [x for v in kp_err_m.values() for x in v]
    report["keypoint_error_m_all"] = {"median": round(float(np.median(all_kp)), 3),
                                      "p90": round(float(np.percentile(all_kp, 90)), 3)}
    (args.out / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2))
    if examples:
        while len(examples) % 2:
            examples.append(np.zeros_like(examples[0]))
        cv2.imwrite(str(args.out / f"examples_{args.split}.jpg"),
                    np.vstack([np.hstack(examples[i:i + 2]) for i in range(0, len(examples), 2)]))

    # 3. visual check on SportsMOT frames (no labels: only the picture can judge)
    tiles = []
    for spec in args.frames:
        seq, f = spec.split(":")
        found_paths = list(args.sportsmot.glob(f"*/{seq}/img1/{int(f):06d}.jpg"))
        if not found_paths:
            print(f"frame not found: {spec}")
            continue
        img = cv2.imread(str(found_paths[0]))
        pred = predict(model, img, args.imgsz)
        ph = fit_homography(names, *pred, tpl, args.conf) if pred else None
        tile = draw_court(img, ph, names, *(pred or (np.zeros((k, 2)), np.zeros(k))), args.conf)
        label = f"{seq}:{f}  " + (f"H ok, {len(ph.inliers)} pts, {ph.reproj_px:.1f}px" if ph else "no homography")
        cv2.putText(tile, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        tiles.append(cv2.resize(tile, (640, 360)))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(args.out / "sportsmot_frames.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))

    print(json.dumps({k_: v for k_, v in report.items() if k_ != "keypoint_error_m"}, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
