"""Measure court homographies against the hand-made test set (court chapter, C09).

Truth: the hand anchors of ``test_set.json`` (``annotate_anchors.py``), each
fitted and polished on the painted lines like C04's anchors; plus, for the NCAA
clip, the C04 hand-anchored homographies of every frame.

For every method (a folder ``<dir>/<sequence>/homographies.npz``: C05 raw,
C07-verified, C09 detector raw or verified, ...) and every test frame:

* **covered** — the method gives a homography for the frame;
* **error** — median distance between its court and the true one on the
  visible court, in metres (``propagate.disagreement``);
* **wrong** — covered with an error above ``--wrong-m`` (1 m): a confidently
  wrong court, the failure that matters most for the map.

Usage:
    python scripts/court/eval_court_test.py \
        --method c05=outputs/court_auto_edges c05+c07=outputs/court_verified/auto_edges \
                 c09=outputs/court_c09 c09+c07=outputs/court_verified/c09 \
        --out experiments/court/C09_learned_keypoints/eval_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _clip import Clip

from hooptrack.court.anchors import AnchorFile, fit_anchor
from hooptrack.court.leagues import SPECS
from hooptrack.court.propagate import disagreement

ROOT = Path("experiments/court/C09_learned_keypoints")


def truth(manifest: Path, anchors_dir: Path, ncaa_step: int) -> dict[tuple[str, int], tuple[np.ndarray, str]]:
    """(sequence, frame) -> (true H, league)."""
    out = {}
    m = json.loads(manifest.read_text())
    for seq, npz in m.get("reference_full_clip", {}).items():
        z = np.load(npz)
        for f, H in zip(z["frames"], z["H"], strict=True):
            if (int(f) - 1) % ncaa_step == 0:
                out[(seq, int(f))] = (H, str(z["league"]))
    for c in m["annotate"]:
        path = anchors_dir / f"anchors_{c['sequence']}.json"
        if not path.is_file():
            print(f"not annotated yet: {c['sequence']}")
            continue
        af = AnchorFile.load(path)
        clip = Clip(af.sequence)
        for a in af.anchors:
            if len(a.points) >= 4:
                H, _, after = fit_anchor(clip.frame(a.frame), a, af.spec, clip.people(a.frame), ignore_below=575)
                out[(af.sequence, a.frame)] = (H, af.league)
    return out


def evaluate(method_dir: Path, gt, wrong_m: float) -> dict:
    per_seq, all_err, wrong, covered = {}, [], 0, 0
    for (seq, f), (Ht, league) in sorted(gt.items()):
        s = per_seq.setdefault(seq, {"frames": 0, "covered": 0, "errors": []})
        s["frames"] += 1
        npz = method_dir / seq / "homographies.npz"
        if not npz.is_file():
            continue
        z = np.load(npz)
        Hm = dict(zip(z["frames"].tolist(), z["H"], strict=True)).get(f)
        if Hm is None:
            continue
        e = disagreement(Hm, Ht, SPECS[league], (1280, 575))[1]
        e = e if np.isfinite(e) else 99.0                  # no visible court in common: as wrong as it gets
        s["covered"] += 1
        s["errors"].append(round(float(e), 3))
        covered += 1
        all_err.append(e)
        wrong += int(e > wrong_m)
    n = sum(s["frames"] for s in per_seq.values())
    return {"frames": n, "covered": covered, "coverage": round(covered / max(n, 1), 3),
            "median_m": round(float(np.median(all_err)), 3) if all_err else None,
            "share_below_0.5m_of_covered": round(float(np.mean(np.array(all_err) < 0.5)), 3) if all_err else None,
            "wrong": wrong, "per_sequence": per_seq}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", nargs="+", required=True, metavar="NAME=DIR")
    ap.add_argument("--manifest", type=Path, default=ROOT / "test_set.json")
    ap.add_argument("--anchors", type=Path, default=ROOT / "test_anchors")
    ap.add_argument("--ncaa-step", type=int, default=50, help="NCAA reference: one frame every N")
    ap.add_argument("--wrong-m", type=float, default=1.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    gt = truth(args.manifest, args.anchors, args.ncaa_step)
    print(f"{len(gt)} test frames over {len({s for s, _ in gt})} clips")
    res = {}
    for m in args.method:
        name, d = m.split("=", 1)
        res[name] = evaluate(Path(d), gt, args.wrong_m)
        r = res[name]
        print(f"{name:12s} covered {r['covered']:3d}/{r['frames']} ({r['coverage']:.0%})  median {r['median_m']} m  "
              f"<0.5 m {r['share_below_0.5m_of_covered']}  WRONG (> {args.wrong_m} m): {r['wrong']}")
    if args.out:
        args.out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
