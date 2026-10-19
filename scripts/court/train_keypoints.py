"""Train the court-keypoint detector (YOLO26-pose) on CPU within a time budget (court chapter, C02).

Why these settings
------------------
* ``yolo26n-pose.pt`` — the smallest model, pretrained on COCO person keypoints.
  Its backbone already detects edges, lines and textures; only the head has to
  learn our 33 court points (the head is re-initialised because 33 != 17).
  Small is also what will later run on a Jetson.
* ``time=<hours>`` — Ultralytics stops at the budget AND stretches the learning-
  rate schedule to it, instead of cutting a 100-epoch schedule halfway.
* ``imgsz=640`` — frames are 1024 x 576; 640 keeps the court lines visible and
  costs ~2.5x less than 1024.
* ``mosaic=0`` — mosaic glues 4 images together: it would cut the court, the
  single object we predict, into pieces that never occur in a broadcast.
* ``fliplr=0.5`` with ``flip_idx`` in data.yaml — a flipped frame is a valid
  frame of the other half of the court; ``flipud=0`` because courts are never
  upside down. ``degrees=0`` (broadcast cameras do not roll).
* ``patience=20`` — early stopping if validation stops improving.
* ``cache=ram`` — decode each JPEG once instead of every epoch (~0.6 GB of RAM).
* ``--freeze 10`` (optional) — keep the COCO-pretrained backbone fixed and train
  only neck + head: much less back-propagation on CPU. Court lines are simple
  features the pretrained backbone already sees.

Usage (from the repo root, venv active):
    python scripts/court/train_keypoints.py --probe                       # a few minutes: measures speed
    python scripts/court/train_keypoints.py --probe --imgsz 512 --freeze 10
    python scripts/court/train_keypoints.py --hours 2.75 --imgsz 512 --freeze 10   # real run, < 3 h
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolved_data_yaml(data: Path) -> Path:
    """Copy of data.yaml whose ``path`` is the absolute dataset folder on THIS machine.

    The dataset may have been converted on another machine (or inside a VM), so
    the ``path`` written at conversion time is not trusted.
    """
    cfg = yaml.safe_load(data.read_text())
    cfg["path"] = str(data.parent.resolve())
    out = data.parent / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/court_keypoints/yolo_v22/data.yaml"))
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--hours", type=float, default=2.75, help="training budget (Ultralytics 'time')")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0,
                    help="freeze the first N layers (backbone = 10): faster on CPU, keeps COCO features")
    ap.add_argument("--cache", default="ram", help="'ram' decodes each image once (faster epochs), 'false' = off")
    ap.add_argument("--device", default="cpu", help="'cpu' or a GPU index such as 0")
    ap.add_argument("--project", type=Path, default=Path("outputs/court_runs"))
    ap.add_argument("--name", default="c02_yolo26n_pose")
    ap.add_argument("--probe", action="store_true", help="1 epoch on 10 %% of the data to measure speed")
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"{args.data} not found: run python scripts/court/convert_to_yolo.py first")

    common = dict(
        data=str(resolved_data_yaml(args.data)), imgsz=args.imgsz, batch=args.batch,
        workers=args.workers, device=args.device, freeze=args.freeze or None,
        cache=False if str(args.cache).lower() == "false" else args.cache,
        project=str(args.project.resolve()), seed=0, deterministic=True, plots=True,
        # augmentation (see docstring)
        mosaic=0.0, mixup=0.0, fliplr=0.5, flipud=0.0, degrees=0.0, translate=0.1, scale=0.3,
    )
    model = YOLO(args.model)

    if args.probe:
        # Two short epochs on 10 % of the data, no validation. The FIRST epoch carries
        # one-off costs (scanning labels, caching images, building the model), so the
        # speed is taken from the SECOND one only: (t_epoch2 - t_epoch1) / 0.1.
        # Validation is timed separately because it always runs on the full val split.
        res = model.train(**common, epochs=2, fraction=0.1, name=f"{args.name}_probe", exist_ok=True, val=False)
        rows = (Path(res.save_dir) / "results.csv").read_text().strip().splitlines()[1:]
        t_cum = [float(r.split(",")[1]) for r in rows]          # column 'time' = cumulative seconds
        t_train = (t_cum[1] - t_cum[0]) / 0.1
        t0 = time.time()
        model.val(data=common["data"], imgsz=args.imgsz, batch=args.batch, device=args.device, plots=False)
        t_val = time.time() - t0
        per_epoch = (t_train + t_val) / 60
        epochs_in_budget = int(args.hours * 60 / per_epoch)
        print(f"\nProbe: one full epoch ~{t_train / 60:.1f} min of training + {t_val:.0f} s of validation.")
        print(f"Estimate: ~{per_epoch:.1f} min per epoch -> ~{epochs_in_budget} epochs in {args.hours} h.")
        if epochs_in_budget < 15:
            print("Fewer than ~15 epochs: try --imgsz 512 --freeze 10, or train on a GPU (Colab).")
        return

    t0 = time.time()
    model.train(**common, epochs=300, time=args.hours, patience=20, name=args.name, exist_ok=False)
    run_dir = args.project / args.name
    (run_dir / "train_meta.json").write_text(json.dumps(
        {"hours_budget": args.hours, "wall_minutes": round((time.time() - t0) / 60, 1),
         "model": args.model, "imgsz": args.imgsz, "freeze": args.freeze, "device": args.device}, indent=2))
    print(f"Done in {(time.time() - t0) / 60:.0f} min. Best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
