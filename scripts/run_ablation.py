"""Ablation study: add one component at a time and measure what it buys.

Each row turns ON one more piece of the tracker:

    1. motion-only        IoU + Kalman, two-stage (≈ ByteTrack)
    2. + GMC              camera motion compensation
    3. + ReID             appearance in the cost + appearance re-acquisition
    4. + memory (full)    long-term identity gallery
    5. full + GT boxes    same tracker on perfect detections: the ceiling of
                          the association logic, independent of the detector

Run detection ONCE (``hooptrack detect``) and this script re-uses the cache,
so the whole study takes minutes, not hours.

Usage:
    python scripts/run_ablation.py --config configs/default.yaml --hota
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hooptrack.cli import cmd_evaluate, cmd_track
from hooptrack.config import apply_overrides, load_config

# "{reid}" is replaced by --reid: every variant sets every component explicitly,
# so the ablation does not depend on what configs/default.yaml happens to enable.
VARIANTS: list[tuple[str, list[str]]] = [
    ("1_motion_only", ["reid.type=none", "tracker.gmc=none", "tracker.memory_enabled=false"]),
    ("2_plus_gmc", ["reid.type=none", "tracker.gmc=sparse", "tracker.memory_enabled=false"]),
    ("3_plus_reid", ["reid.type={reid}", "tracker.gmc=sparse", "tracker.memory_enabled=false"]),
    ("4_full", ["reid.type={reid}", "tracker.gmc=sparse", "tracker.memory_enabled=true"]),
    ("5_full_gt_boxes", ["reid.type={reid}", "detector.type=gt", "tracker.gmc=sparse",
                         "tracker.memory_enabled=true"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to every variant")
    ap.add_argument("--out", type=Path, default=Path("outputs/ablation"))
    ap.add_argument("--hota", action="store_true")
    ap.add_argument("--reid", default="resnet", help="extractor for the variants that use ReID")
    ap.add_argument("--only", nargs="*", default=None, help="run only these variants (by name)")
    ap.add_argument("--no-cache", action="store_true", help="run YOLO live instead of the cache")
    args = ap.parse_args()

    base = load_config(args.config, args.set)
    if not args.no_cache and base["detector"]["type"] == "yolo":
        base = apply_overrides(base, ["detector.type=cached"])

    rows = {}
    for name, overrides in VARIANTS:
        if args.only and name not in args.only:
            continue
        overrides = [o.replace("{reid}", args.reid) for o in overrides]
        print(f"\n=== {name} ===")
        cfg = apply_overrides(base, overrides + [f"output.dir={args.out / name}"])
        run_dir = cmd_track(cfg)
        rows[name] = cmd_evaluate(cfg, run_dir, hota=args.hota)

    df = pd.DataFrame(rows).T
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "ablation.csv", float_format="%.2f")
    try:
        md = df.to_markdown(floatfmt=".1f")   # needs the optional `tabulate` package
    except ImportError:
        md = df.to_string(float_format=lambda v: f"{v:.1f}")
    (args.out / "ablation.md").write_text(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
