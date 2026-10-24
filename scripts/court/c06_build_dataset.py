"""Build the court-SEGMENTATION dataset (court chapter, C06).

One class, ``court``: the playing area inside the sidelines and baselines.
Labels are YOLO segmentation polygons (``0 x1 y1 x2 y2 ...``, normalised), all
derived from a court homography, so every source gives the SAME definition of
the mask (``hooptrack.court.masks``).

Sources (each in its own folder under ``--out``, each with train/val/test):

========== ============================================ ==============================================
source     what                                         homography from
========== ============================================ ==============================================
deepsport  DeepSportRadar basketball instants: 15       per-image camera calibration (exact). Images
           French LNB arenas, fixed elevated cameras    are undistorted first (lens kc), so the
                                                        court is a true homography of the floor.
v22        Roboflow court keypoints (C01): 850 NBA      the image's own keypoint labels (C01 verified
           broadcast frames, 2025                       the template to ~1-2 px); bad fits dropped
sportsmot  SportsMOT TRAIN clips with automatic         C05 chains, only the frames listed (checked
           homographies checked by eye                  on overlay videos)
synth      domain-randomised courts (``court.synth``)   the render itself (exact)
========== ============================================ ==============================================

**Leak-free splits.** DeepSport is split by ARENA (md5 order, ~70/15/15), v22
keeps its by-game split (C01), SportsMOT train clips go to train only. The
SportsMOT VALIDATION clips (NCAA + the three FIBA arenas of 009) are never used
here: they are the final test (``c06_eval.py``).

Usage (repo root, venv active):
    python scripts/court/c06_build_dataset.py synth --n 3000
    python scripts/court/c06_build_dataset.py v22
    python scripts/court/c06_build_dataset.py deepsport --root data/basketball-instants-dataset --preview 12
    python scripts/court/c06_build_dataset.py sportsmot
    python scripts/court/c06_build_dataset.py yaml          # writes data/court_seg/data.yaml from what exists
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from hooptrack.court import nba_template
from hooptrack.court.homography import fit_homography
from hooptrack.court.leagues import SPECS, lines
from hooptrack.court.masks import court_polygon, yolo_seg_line

LONG_SIDE = 1280          # images are stored with this long side (16:9 broadcast -> 1280 x 720)
SOURCES = ("deepsport", "v22", "sportsmot", "synth")

# SportsMOT TRAIN clips whose automatic homographies were checked on the overlay videos (C05).
# (sequence, homographies.npz, first frame, last frame): only frames in [first, last] are used.
SPORTSMOT_VERIFIED = [
    ("v_2j7kLB-vEEk_c002", "outputs/court_auto_final/v_2j7kLB-vEEk_c002/homographies.npz", 7, 900),
    ("v_-6Os86HzwCs_c009", "outputs/court_auto_final/v_-6Os86HzwCs_c009/homographies.npz", 1, 253),
]


# ----------------------------------------------------------------------------- helpers
def _md5_order(keys) -> list[str]:
    return sorted(keys, key=lambda k: hashlib.md5(k.encode()).hexdigest())


def _split_groups(groups, fractions=(0.7, 0.15, 0.15)) -> dict[str, str]:
    """group -> split, deterministic and leak-free (whole groups go to one split)."""
    order = _md5_order(set(groups))
    n = len(order)
    n_val = max(1, round(fractions[1] * n)) if n >= 3 else 0
    n_test = max(1, round(fractions[2] * n)) if n >= 3 else 0
    out = {}
    for i, g in enumerate(order):
        out[g] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
    return out


def _resize(img: np.ndarray, H: np.ndarray):
    """Resize so the long side is LONG_SIDE, and update H (court -> image) accordingly."""
    h, w = img.shape[:2]
    s = LONG_SIDE / max(h, w)
    if abs(s - 1) > 1e-6:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        H = np.diag([s, s, 1.0]) @ H
    return img, H


def _write(out: Path, split: str, name: str, img: np.ndarray, H: np.ndarray, spec, counts: dict) -> bool:
    size = img.shape[1::-1]
    poly = court_polygon(H, spec, size)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # an image with no court visible is a valid NEGATIVE example: empty label file
    (out / "labels" / split / f"{name}.txt").write_text(yolo_seg_line(poly, size) + "\n" if len(poly) >= 3 else "")
    counts[split] = counts.get(split, 0) + 1
    return len(poly) >= 3


def _preview(img: np.ndarray, H: np.ndarray, spec, path: Path) -> None:
    """Court lines + mask outline drawn on the image: the check that labels are right."""
    v = img.copy()
    for ln in lines(spec):
        p = cv2.perspectiveTransform(ln[None].astype(np.float64), H)[0]
        cv2.polylines(v, [np.clip(p, -1e4, 1e4).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)
    poly = court_polygon(H, spec, img.shape[1::-1])
    if len(poly) >= 3:
        cv2.polylines(v, [poly.astype(np.int32)], True, (0, 255, 0), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), v)


# ----------------------------------------------------------------------------- sources
def build_deepsport(root: Path, out: Path, rule: str, preview: int, limit: int = 0) -> dict:
    """DeepSportRadar instants: <arena>/<game>/camcourt<N>_<timestamp>.json + ..._0.png."""
    calibs = sorted(root.rglob("camcourt*_*.json"))
    if limit:
        calibs = calibs[::max(len(calibs) // limit, 1)][:limit]           # a spread-out sample, for checks
    if not calibs:
        raise SystemExit(f"no camcourt*.json calibration files under {root}")
    arenas = {c: c.relative_to(root).parts[0] for c in calibs}
    split_of = _split_groups(arenas.values())
    spec = SPECS[rule]
    counts, skipped, shown = {}, 0, 0
    for c in calibs:
        img_path = c.with_name(c.stem + "_0.png")
        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue
        cal = json.loads(c.read_text())
        cal = cal.get("calibration", cal)
        K = np.array(cal["KK"], float).reshape(3, 3)
        R = np.array(cal["R"], float).reshape(3, 3)
        T = np.array(cal["T"], float).reshape(3)
        kc = np.array(cal.get("kc", [0, 0, 0, 0, 0]), float)
        # 1. remove lens distortion: afterwards the floor maps to the image by a homography
        img = cv2.undistort(img, K, kc)
        # 2. floor (z = 0, centimetres) -> image:  H = K [r1 r2 T];  metres: scale x, y by 100
        H = K @ np.c_[R[:, 0], R[:, 1], T] @ np.diag([100.0, 100.0, 1.0])
        img, H = _resize(img, H)
        split = split_of[arenas[c]]
        name = f"ds_{arenas[c]}_{c.parent.name}_{c.stem}"
        _write(out, split, name, img, H, spec, counts)
        if shown < preview:
            _preview(img, H, spec, out / "preview" / f"{name}.jpg")
            shown += 1
    arena_split = {a: split_of[a] for a in sorted(set(arenas.values()))}
    return {"images": counts, "skipped_no_image": skipped, "arenas": arena_split}


def build_v22(src: Path, out: Path, max_reproj: float, preview: int) -> dict:
    """Roboflow v22 in YOLO-pose format (C02): the keypoint labels give each image's homography."""
    data = yaml.safe_load((src / "data.yaml").read_text())
    names = data["kpt_names"][0] if isinstance(data["kpt_names"], dict) else data["kpt_names"]
    k = len(names)
    tpl = nba_template()
    counts, dropped, shown = {}, 0, 0
    for split in ("train", "val", "test"):
        for img_path in sorted((src / "images" / split).glob("*.jpg")):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            v = np.array((src / "labels" / split / f"{img_path.stem}.txt").read_text().split(), float)
            kp = v[5:5 + 3 * k].reshape(k, 3)
            hom = fit_homography(names, kp[:, :2] * [w, h], kp[:, 2] / 2, tpl)
            if hom is None or hom.reproj_px > max_reproj or len(hom.inliers) < 6:
                dropped += 1                      # not enough good points to trust the mask
                continue
            img2, H = _resize(img, hom.H)
            _write(out, split, f"v22_{img_path.stem}", img2, H, SPECS["NBA"], counts)
            if shown < preview:
                _preview(img2, H, SPECS["NBA"], out / "preview" / f"v22_{img_path.stem}.jpg")
                shown += 1
    return {"images": counts, "dropped_bad_fit": dropped}


def build_sportsmot(out: Path, stride: int, preview: int) -> dict:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _clip import Clip

    counts, shown, per_clip = {}, 0, {}
    for seq, npz, first, last in SPORTSMOT_VERIFIED:
        if not Path(npz).is_file():
            print(f"skip {seq}: {npz} not found (run auto_homographies.py first)")
            continue
        z = np.load(npz)
        spec = SPECS[str(z["league"])]
        clip = Clip(seq)
        n = 0
        for f, H in zip(z["frames"], z["H"], strict=True):
            if not first <= f <= last or (f - first) % stride:
                continue
            img, H2 = _resize(clip.frame(int(f)), H)
            _write(out, "train", f"smot_{seq}_{int(f):06d}", img, H2, spec, counts)
            n += 1
            if shown < preview:
                _preview(img, H2, spec, out / "preview" / f"smot_{seq}_{int(f):06d}.jpg")
                shown += 1
        per_clip[seq] = n
    return {"images": counts, "per_clip": per_clip}


def build_synth(out: Path, n: int, seed: int, preview: int) -> dict:
    from hooptrack.court.synth import render
    rng = np.random.default_rng(seed)
    counts = {}
    for i in range(n):
        s = render(rng, (LONG_SIDE, LONG_SIDE * 9 // 16))
        split = "val" if i % 10 == 0 else "train"             # 10 % synthetic validation (sanity only)
        _write(out, split, f"syn_{seed}_{i:05d}", s.image, s.H, SPECS[s.spec], counts)
        if i < preview:
            _preview(s.image, s.H, SPECS[s.spec], out / "preview" / f"syn_{i:05d}.jpg")
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n}", flush=True)
    return {"images": counts, "seed": seed}


def write_yaml(root: Path) -> Path:
    """data.yaml listing every source that exists. Validation = REAL images only (deepsport + v22)."""
    train, val, test = [], [], []
    for s in SOURCES:
        for split, lst in (("train", train), ("val", val), ("test", test)):
            d = root / s / "images" / split
            if d.is_dir() and any(d.iterdir()):
                if split == "val" and s == "synth":
                    continue                      # model selection on real arenas, not on our own renders
                lst.append(f"{s}/images/{split}")
    cfg = {"path": ".", "train": train, "val": val, "test": test, "names": {0: "court"}}
    path = root / "data.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=(*SOURCES, "yaml"))
    ap.add_argument("--out", type=Path, default=Path("data/court_seg"))
    ap.add_argument("--root", type=Path, default=Path("data/basketball-instants-dataset"), help="deepsport folder")
    ap.add_argument("--rule", default="FIBA", help="deepsport court rules (LNB Pro A plays FIBA rules)")
    ap.add_argument("--v22", type=Path, default=Path("data/court_keypoints/yolo_v22"))
    ap.add_argument("--max-reproj", type=float, default=4.0, help="v22: drop images whose keypoint fit is worse")
    ap.add_argument("--stride", type=int, default=10, help="sportsmot: one frame every N")
    ap.add_argument("--n", type=int, default=3000, help="synth: number of images")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--preview", type=int, default=12, help="save N images with the label drawn, to check it")
    ap.add_argument("--limit", type=int, default=0, help="deepsport: only N images (quick check)")
    args = ap.parse_args()

    if args.source == "yaml":
        print(f"wrote {write_yaml(args.out)}")
        print(yaml.safe_load((args.out / 'data.yaml').read_text()))
        return
    out = args.out / args.source
    if args.source == "deepsport":
        report = build_deepsport(args.root, out, args.rule, args.preview, args.limit)
    elif args.source == "v22":
        report = build_v22(args.v22, out, args.max_reproj, args.preview)
    elif args.source == "sportsmot":
        report = build_sportsmot(out, args.stride, args.preview)
    else:
        report = build_synth(out, args.n, args.seed, args.preview)
    out.mkdir(parents=True, exist_ok=True)
    (out / "build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"check the label previews in {out / 'preview'} before training")
    write_yaml(args.out)


if __name__ == "__main__":
    main()
