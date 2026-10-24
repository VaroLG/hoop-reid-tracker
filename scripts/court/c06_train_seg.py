"""Train the court-AREA segmentation model (YOLO26-seg) on CPU within a time budget (court chapter, C06).

Same recipe as C02 (``train_keypoints.py``), adapted to segmentation:

* ``yolo26n-seg.pt`` — smallest segmentation model, COCO-pretrained.
* ``mosaic=0`` — the court is one large object; mosaic would cut it into
  pieces that never occur in a broadcast.
* ``fliplr=0.5`` — a mirrored court is still a court (no keypoint order to fix).
* **Strong colour augmentation** (``hsv_h=0.5, hsv_s=0.7, hsv_v=0.4``) — the
  lesson of 003/009: colour is what does NOT generalise across arenas. Changing
  hue by up to half the colour wheel forces the model to use shape and lines.
* ``scale=0.5, translate=0.2`` — zoomed-in and zoomed-out views.
* ``time=<hours>`` — stops at the budget and stretches the LR schedule to it.
* Validation = real arenas only (DeepSport val arenas + v22 val), so the saved
  "best" model is the best on REAL courts not seen in training.

Usage (repo root, venv active):
    python scripts/court/c06_train_seg.py --probe                     # a few minutes: measures speed
    python scripts/court/c06_train_seg.py --hours 2.75 --imgsz 512    # real run (< 3 h on CPU)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolved_data_yaml(data: Path) -> Path:
    """Copy of data.yaml whose ``path`` is the absolute dataset folder on THIS machine."""
    cfg = yaml.safe_load(data.read_text())
    cfg["path"] = str(data.parent.resolve())
    if not cfg.get("val"):
        raise SystemExit("data.yaml has no validation images: build the deepsport and/or v22 sources first")
    out = data.parent / "data_resolved.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data/court_seg/data.yaml"))
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--hours", type=float, default=2.75)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze", type=int, default=0, help="freeze the first N layers (backbone = 10)")
    ap.add_argument("--cache", default="false", help="'ram' only if you have > 8 GB free (thousands of images)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", type=Path, default=Path("outputs/court_runs"))
    ap.add_argument("--name", default="c06_yolo26n_seg")
    ap.add_argument("--probe", action="store_true", help="2 short epochs on 5 %% of the data to measure speed")
    args = ap.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"{args.data} not found: run scripts/court/c06_build_dataset.py first")
    common = dict(
        data=str(resolved_data_yaml(args.data)), imgsz=args.imgsz, batch=args.batch, workers=args.workers,
        device=args.device, freeze=args.freeze or None,
        cache=False if str(args.cache).lower() == "false" else args.cache,
        project=str(args.project.resolve()), seed=0, deterministic=True, plots=True, single_cls=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0, fliplr=0.5, flipud=0.0, degrees=0.0,
        translate=0.2, scale=0.5, hsv_h=0.5, hsv_s=0.7, hsv_v=0.4,
    )
    model = YOLO(args.model)

    if args.probe:
        frac = 0.05
        res = model.train(**common, epochs=2, fraction=frac, name=f"{args.name}_probe", exist_ok=True, val=False)
        rows = (Path(res.save_dir) / "results.csv").read_text().strip().splitlines()[1:]
        t_cum = [float(r.split(",")[1]) for r in rows]
        t_train = (t_cum[1] - t_cum[0]) / frac
        t0 = time.time()
        model.val(data=common["data"], imgsz=args.imgsz, batch=args.batch, device=args.device, plots=False)
        t_val = time.time() - t0
        per_epoch = (t_train + t_val) / 60
        n = int(args.hours * 60 / per_epoch)
        print(f"\nProbe: one full epoch ~{t_train / 60:.1f} min of training + {t_val:.0f} s of validation.")
        print(f"Estimate: ~{per_epoch:.1f} min per epoch -> ~{n} epochs in {args.hours} h.")
        if n < 10:
            print("Fewer than ~10 epochs: use --imgsz 416, --freeze 10, fewer synthetic images, or a GPU.")
        return

    t0 = time.time()
    model.train(**common, epochs=300, time=args.hours, patience=15, name=args.name, exist_ok=False)
    run_dir = args.project / args.name
    (run_dir / "train_meta.json").write_text(json.dumps(
        {"hours_budget": args.hours, "wall_minutes": round((time.time() - t0) / 60, 1), "model": args.model,
         "imgsz": args.imgsz, "freeze": args.freeze, "device": args.device,
         "data": yaml.safe_load(args.data.read_text())}, indent=2))
    print(f"Done in {(time.time() - t0) / 60:.0f} min. Best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
