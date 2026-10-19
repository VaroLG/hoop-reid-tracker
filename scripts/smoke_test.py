"""End-to-end smoke test on a rendered synthetic clip (no download, no GPU).

It writes a tiny dataset to disk in the exact SportsMOT/MOTChallenge layout —
JPEG frames, gt.txt, seqinfo.ini, split files — with:

  * a panning camera (tests GMC),
  * two teams with the same jersey colour per team (tests appearance),
  * players leaving the frame and coming back (tests the identity memory),

then runs the real CLI path: ``track`` (oracle detector + histogram ReID) and
``evaluate``. If this passes, every piece of plumbing works; only the model
weights change on the real data.

Usage:
    python scripts/smoke_test.py [--out outputs/smoke]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from hooptrack.cli import cmd_evaluate, cmd_track
from hooptrack.config import load_config
from hooptrack.synthetic import SceneConfig, make_scene

SEQ = "v_synthetic_c001"
W, H = 1280, 720


def render_dataset(root: Path, n_frames: int = 240, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    scene = make_scene(SceneConfig(
        # (player, start, duration): 50 / 40 frames are recovered by stage 4,
        # 110 frames exceeds max_time_lost=60 and needs the long-term memory.
        n_players=8, n_frames=n_frames, exits=[(0, 60, 50), (3, 100, 110), (5, 30, 40)],
        miss_rate=0.0, fp_rate=0.0, seed=seed,
    ))

    # Wide textured "court" so the camera can pan over it.
    pad = 200
    court = np.full((H + 2 * pad, W + 2 * pad, 3), (60, 110, 160), np.uint8)
    for _ in range(400):
        c = (int(rng.integers(0, W + 2 * pad)), int(rng.integers(0, H + 2 * pad)))
        cv2.circle(court, c, int(rng.integers(3, 25)), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)

    team_colors = [(30, 30, 200), (230, 230, 230)]                    # red / white jerseys
    shorts = [tuple(int(v) for v in rng.integers(0, 255, 3)) for _ in range(8)]

    seq_dir = root / "dataset" / "val" / SEQ
    (seq_dir / "img1").mkdir(parents=True, exist_ok=True)
    (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
    gt_lines = []
    for f, fr in enumerate(scene, start=1):
        ox = int(pad + 80 * np.sin(f / 35))       # camera pan (px)
        img = court[pad : pad + H, ox : ox + W].copy()
        for gid, (x1, y1, x2, y2) in zip(fr.gt_ids, fr.gt_boxes, strict=True):
            # world -> image: the camera moved right by (ox - pad)
            x1, x2 = x1 - (ox - pad), x2 - (ox - pad)
            x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
            mid = y1 + int(0.55 * (y2 - y1))
            cv2.rectangle(img, (x1, y1), (x2, mid), team_colors[(gid - 1) % 2], -1)
            cv2.rectangle(img, (x1, mid), (x2, y2), shorts[gid - 1], -1)
            gt_lines.append(f"{f},{gid},{x1},{y1},{x2 - x1},{y2 - y1},1,1,1")
        cv2.imwrite(str(seq_dir / "img1" / f"{f:06d}.jpg"), img)

    (seq_dir / "gt" / "gt.txt").write_text("\n".join(gt_lines) + "\n")
    (seq_dir / "seqinfo.ini").write_text(
        f"[Sequence]\nname={SEQ}\nimDir=img1\nframeRate=25\nseqLength={n_frames}\n"
        f"imWidth={W}\nimHeight={H}\nimExt=.jpg\n"
    )
    splits = root / "splits_txt"
    splits.mkdir(parents=True, exist_ok=True)
    for name in ("basketball", "val"):
        (splits / f"{name}.txt").write_text(SEQ + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/smoke"))
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    shutil.rmtree(args.out, ignore_errors=True)
    data = args.out / "data"
    render_dataset(data)

    common = [
        f"data.root={data / 'dataset'}", f"data.splits_dir={data / 'splits_txt'}",
        "data.split=val", "detector.type=gt", "detector.box_noise=2.0", "detector.drop_rate=0.03",
        "reid.type=histogram",
        # The synthetic clip has no real court, and the variants below must not
        # depend on what configs/default.yaml currently enables.
        "filters.court_mask=false", "tracker.gmc=sparse",
    ]
    results = {}
    variants = (
        ("motion_only", ["reid.type=none", "tracker.memory_enabled=false"]),
        ("reid_no_memory", ["tracker.memory_enabled=false"]),
        ("full", ["tracker.memory_enabled=true"]),
    )
    for name, extra in variants:
        cfg = load_config(args.config, common + extra + [f"output.dir={args.out / name}"])
        results[name] = cmd_evaluate(cfg, cmd_track(cfg), hota=False)

    print("\nSMOKE TEST SUMMARY")
    print(json.dumps(results, indent=2))
    full, base = results["full"], results["motion_only"]
    assert full["IDF1"] > base["IDF1"], "ReID + memory should beat motion-only"
    assert full["reentry_recovered_%"] == 100.0, "every re-entry should get its id back"
    assert full["reentry_stolen"] == 0, "no identity should be stolen on the synthetic clip"
    print("OK")


if __name__ == "__main__":
    main()
