"""B04 · Ball detection measured against the hand clicks of the VALIDATION clips (``b02_label_ball.py --split val``).

For every labelled frame, the detector's most confident box (above ``--conf``) is the answer:

* ball visible and answer within one ball diameter of the click: **found**;
* ball visible, answer elsewhere: **wrong place** (a false alarm and a miss);
* ball visible, no answer: **missed**;
* ball not visible, an answer anyway: **false alarm**.

Reported per clip and overall: recall (found / visible), precision (found / answers), median distance of the found
balls in pixels, and the same over a sweep of confidences (``sweep.csv``) to see the trade-off.

The SportsMOT validation clips are never used to train or tune a ball detector: this is its test.

Usage (repo root):
    python scripts/ball/b04_eval_ball.py --weights yolo26m.pt --classes 32 --imgsz 1280 --name coco_1280
    python scripts/ball/b04_eval_ball.py --weights weights/ball_yolo26s_1280.pt --imgsz 1280 --name ball_b03
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.ball.labels import ball_diameter, load_labels
from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config


def score(rows: pd.DataFrame, conf: float) -> dict:
    ans = rows["conf"] >= conf
    vis = rows["visible"]
    found = vis & ans & (rows["dist"] <= rows["diam"])
    n_ans = int(ans.sum())
    return {"conf": conf, "frames": len(rows), "visible": int(vis.sum()), "found": int(found.sum()),
            "wrong_place": int((vis & ans & ~(rows["dist"] <= rows["diam"])).sum()),
            "missed": int((vis & ~ans).sum()), "false_alarm_no_ball": int((~vis & ans).sum()),
            "recall": round(found.sum() / max(1, vis.sum()), 3),
            "precision": round(found.sum() / max(1, n_ans), 3),
            "median_px": round(float(rows.loc[found, "dist"].median()), 1) if found.any() else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--labels", type=Path, default=Path("data/ball_labels/val.csv"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--classes", nargs="*", type=int, default=None, help="COCO: 32 (sports ball); own model: all")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence for the headline numbers")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/b04"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    labels = load_labels(args.labels)
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=val", "data.only=null"]))}
    rows = []
    for clip, lab in labels.groupby("clip"):
        seq = seqs[clip]
        gt = seq.load_gt()
        for _, r in lab.iterrows():
            f = int(r["frame"])
            res = model.predict(cv2.imread(str(seq.frame_path(f))), imgsz=args.imgsz, conf=0.01,
                                classes=args.classes, verbose=False)[0]
            boxes, confs = res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()
            row = {"clip": clip, "frame": f, "visible": bool(r["visible"]), "conf": 0.0, "dist": np.inf, "diam": 0.0}
            if len(boxes):
                k = int(confs.argmax())
                cx, cy = (boxes[k, 0] + boxes[k, 2]) / 2, (boxes[k, 1] + boxes[k, 3]) / 2
                row["conf"] = float(confs[k])
                if r["visible"]:
                    row["dist"] = float(np.hypot(cx - r["x"], cy - r["y"]))
            if r["visible"]:
                row["diam"] = ball_diameter(r["x"], r["y"], gt.get(f, (None, np.zeros((0, 4))))[1])
            rows.append(row)
        print(f"{clip}: {len(lab)} frames")
    df = pd.DataFrame(rows)
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "frames.csv", index=False)
    per_clip = pd.DataFrame([{"clip": c, **score(g, args.conf)} for c, g in df.groupby("clip")]
                            + [{"clip": "ALL", **score(df, args.conf)}])
    sweep = pd.DataFrame([score(df, c) for c in np.round(np.arange(0.05, 0.96, 0.05), 2)])
    per_clip.to_csv(out / "per_clip.csv", index=False)
    sweep.to_csv(out / "sweep.csv", index=False)
    print(f"\n{args.name}: {args.weights}, imgsz {args.imgsz}, conf {args.conf}\n" + per_clip.to_string(index=False))
    print("\nConfidence sweep (all clips):\n" + sweep[["conf", "recall", "precision", "found", "wrong_place",
                                                       "missed", "false_alarm_no_ball"]].to_string(index=False))


if __name__ == "__main__":
    main()
