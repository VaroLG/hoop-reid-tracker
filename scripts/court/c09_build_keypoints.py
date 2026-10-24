"""Build the court-KEYPOINT dataset for the learned detector (court chapter, C09).

Same 33 keypoints, order and ``flip_idx`` as C02 (``data/court_keypoints/yolo_v22``),
so the C02 training and evaluation scripts work unchanged. What changes is the
variety: C02 saw one domain (NBA 2025 broadcasts) and did not transfer to
SportsMOT. Every label here is derived from a court homography, as in C06:

========== ===================================================== ================================
source     what                                                  homography from
========== ===================================================== ================================
v22        850 NBA 2025 broadcast frames (C01/C02)               used as is (already keypoints)
deepsport  DeepSportRadar: 15 French LNB arenas                  per-image calibration, undistorted
sportsmot  SportsMOT TRAIN clips, every 10th frame               C05 automatic, **kept only where the
                                                                 C07 clip camera accepts the frame**
synth      domain-randomised courts (``court.synth``)            the render itself
========== ===================================================== ================================

The SportsMOT VALIDATION clips are never used: they are the test set
(``test_set.json``, hand anchors). DeepSport is split by arena (C06's split).

A keypoint is labelled visible (2) when it projects inside the image and in
front of the camera, even if a player or the score bar hides it: its position
is known exactly from the homography, and the detector must learn to infer it.
The hoop centres (09, 33) are points in the AIR in the v22 labels; a floor
homography cannot place them, so they are left unlabelled (0) here.

Usage (repo root, venv active):
    python scripts/court/c09_build_keypoints.py deepsport
    python scripts/court/c09_build_keypoints.py sportsmot
    python scripts/court/c09_build_keypoints.py synth --n 2000
    python scripts/court/c09_build_keypoints.py yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from c06_build_dataset import _resize, _split_groups  # noqa: E402  (C06: same resizing, same arena split)

from hooptrack.court.leagues import SPECS, keypoints, lines  # noqa: E402
from hooptrack.court.masks import court_polygon  # noqa: E402

V22 = Path("data/court_keypoints/yolo_v22")
SOURCES = ("deepsport", "sportsmot", "synth")
NON_FLOOR = {"09", "33"}
# C07 output folders holding verified homographies of SportsMOT TRAIN clips
VERIFIED_TRAIN = [Path("outputs/court_verified/train"), Path("outputs/court_verified/auto_final")]


def kpt_names() -> list[str]:
    d = yaml.safe_load((V22 / "data.yaml").read_text())
    names = d["kpt_names"][0] if isinstance(d["kpt_names"], dict) else d["kpt_names"]
    return [f"{int(n):02d}" if not isinstance(n, str) else n for n in names]


def label_line(H: np.ndarray, spec, size, names: list[str]) -> str:
    """YOLO-pose line: class, court box, then (x, y, v) for each keypoint (normalised). Empty if no court."""
    w, h = size
    poly = court_polygon(H, spec, size)
    if len(poly) < 3:
        return ""
    x0, y0 = np.clip(poly.min(0), 0, [w, h])
    x1, y1 = np.clip(poly.max(0), 0, [w, h])
    # sign of "in front of the camera" (court_polygon's trick): the floor at the bottom centre is in front
    c = np.linalg.inv(H) @ np.array([w / 2, h - 1, 1.0])
    sign = np.sign(H[2] @ (c / c[2])) or 1.0
    kp = keypoints(spec)
    vals = []
    for n in names:
        q = H @ np.array([*kp[n], 1.0])
        u, v = q[:2] / q[2]
        vis = n not in NON_FLOOR and sign * q[2] > 0 and 0 <= u < w and 0 <= v < h
        vals += [u / w, v / h, 2] if vis else [0.0, 0.0, 0]
    box = [(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h]
    return "0 " + " ".join(f"{b:.6f}" for b in box) + " " + " ".join(
        f"{x:.6f} {y:.6f} {int(v)}" for x, y, v in zip(vals[0::3], vals[1::3], vals[2::3], strict=True))


def _write(out: Path, split: str, name: str, img, H, spec, names, counts: dict) -> None:
    for d in ("images", "labels"):
        (out / d / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    line = label_line(H, spec, img.shape[1::-1], names)
    (out / "labels" / split / f"{name}.txt").write_text(line + "\n" if line else "")
    counts[split] = counts.get(split, 0) + 1


def _preview(img, H, spec, names, path: Path) -> None:
    """Court lines and the labelled keypoints drawn on the image: the check that labels are right."""
    v = img.copy()
    for ln in lines(spec):
        p = cv2.perspectiveTransform(ln[None].astype(np.float64), H)[0]
        cv2.polylines(v, [np.clip(p, -1e4, 1e4).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
    lab = label_line(H, spec, img.shape[1::-1], names).split()
    if lab:
        k = np.array(lab[5:], float).reshape(-1, 3)
        for n, (x, y, vis) in zip(names, k, strict=True):
            if vis:
                c = (int(x * img.shape[1]), int(y * img.shape[0]))
                cv2.circle(v, c, 5, (0, 0, 255), -1)
                cv2.putText(v, n, (c[0] + 5, c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), v)


def build_deepsport(root: Path, out: Path, names, preview: int) -> dict:
    calibs = sorted(root.rglob("camcourt*_*.json"))
    arenas = {c: c.relative_to(root).parts[0] for c in calibs}
    split_of = _split_groups(arenas.values())
    spec = SPECS["FIBA"]                                   # LNB Pro A plays FIBA rules
    counts = {}
    for i, c in enumerate(calibs):
        img = cv2.imread(str(c.with_name(c.stem + "_0.png")))
        if img is None:
            continue
        cal = json.loads(c.read_text())
        cal = cal.get("calibration", cal)
        K = np.array(cal["KK"], float).reshape(3, 3)
        R = np.array(cal["R"], float).reshape(3, 3)
        T = np.array(cal["T"], float).reshape(3)
        img = cv2.undistort(img, K, np.array(cal.get("kc", [0, 0, 0, 0, 0]), float))
        H = K @ np.c_[R[:, 0], R[:, 1], T] @ np.diag([100.0, 100.0, 1.0])     # calibration in cm
        img, H = _resize(img, H)
        name = f"ds_{arenas[c]}_{c.parent.name}_{c.stem}"
        _write(out, split_of[arenas[c]], name, img, H, spec, names, counts)
        if i < preview:
            _preview(img, H, spec, names, out / "preview" / f"{name}.jpg")
    return {"images": counts}


def build_sportsmot(out: Path, names, stride: int, preview: int) -> dict:
    from _clip import Clip
    counts, per_clip, shown = {}, {}, 0
    for base in VERIFIED_TRAIN:
        for npz in sorted(base.glob("*/homographies.npz")):
            seq = npz.parent.name
            if seq in per_clip:
                continue
            clip = Clip(seq)
            if "/val/" in clip.seq.root.as_posix():
                continue                                    # validation clips are the test set
            z = np.load(npz)
            spec = SPECS[str(z["league"])]
            n = 0
            for f, H in zip(z["frames"], z["H"], strict=True):
                if (int(f) - 1) % stride:
                    continue
                img, H2 = _resize(clip.frame(int(f)), H)
                name = f"smot_{seq}_{int(f):06d}"
                _write(out, "train", name, img, H2, spec, names, counts)
                n += 1
                if shown < preview:
                    _preview(img, H2, spec, names, out / "preview" / f"{name}.jpg")
                    shown += 1
            per_clip[seq] = n
    return {"images": counts, "per_clip": per_clip}


def build_synth(out: Path, names, n: int, seed: int, preview: int) -> dict:
    from hooptrack.court.synth import render
    rng = np.random.default_rng(seed)
    counts = {}
    for i in range(n):
        s = render(rng, (1280, 720))
        name = f"syn_{seed}_{i:05d}"
        _write(out, "train", name, s.image, s.H, SPECS[s.spec], names, counts)   # model selection on real only
        if i < preview:
            _preview(s.image, s.H, SPECS[s.spec], names, out / "preview" / f"{name}.jpg")
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return {"images": counts, "seed": seed}


def write_yaml(root: Path) -> Path:
    """data.yaml over every source (v22 read in place). Validation and test: real images only."""
    v22 = yaml.safe_load((V22 / "data.yaml").read_text())
    rel = Path("..") / V22.relative_to("data")
    split = {"train": [], "val": [], "test": []}
    for s in ("v22", *SOURCES):
        base = rel if s == "v22" else Path(s)
        for k in split:
            d = (V22 if s == "v22" else root / s) / "images" / k
            if d.is_dir() and any(d.iterdir()):
                split[k].append((base / "images" / k).as_posix())
    cfg = {"path": ".", **split, "names": {0: "court"}, "kpt_shape": v22["kpt_shape"],
           "flip_idx": v22["flip_idx"], "kpt_names": v22["kpt_names"]}
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=(*SOURCES, "yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/court_kp"))
    ap.add_argument("--root", type=Path, default=Path("data/basketball-instants-dataset"))
    ap.add_argument("--stride", type=int, default=10, help="sportsmot: one frame every N")
    ap.add_argument("--n", type=int, default=2000, help="synth: number of images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=12)
    args = ap.parse_args()
    names = kpt_names()
    if args.source != "yaml":
        out = args.out / args.source
        if args.source == "deepsport":
            report = build_deepsport(args.root, out, names, args.preview)
        elif args.source == "sportsmot":
            report = build_sportsmot(out, names, args.stride, args.preview)
        else:
            report = build_synth(out, names, args.n, args.seed, args.preview)
        out.mkdir(parents=True, exist_ok=True)
        (out / "build_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    path = write_yaml(args.out)
    print(f"wrote {path}: {yaml.safe_load(path.read_text())}")


if __name__ == "__main__":
    main()
