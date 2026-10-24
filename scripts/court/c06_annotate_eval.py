"""Ground-truth court masks for the C06 test clips (SportsMOT validation), by hand + propagation.

The C06 model is tested on the four SportsMOT VALIDATION clips — never used in
training. To measure a mask we need the true court on those frames:

* NCAA ``v_00HRwkvvjtQ_c001``: the hand-anchored homographies of C04.
* the three FIBA arenas of experiment 009: a few frames per clip annotated by
  hand in ``eval_anchors.json``, in one of two ways:
  - ``points``: court point in metres <-> image pixel, read on zoomed pixel
    grids; polished on the painted lines and propagated ``--span`` frames both
    ways with the C04 chain (cm-level over short spans);
  - ``polygon``: the court AREA itself, from the visible boundary lines, when
    the lines inside the court do not give a reliable homography (Istanbul).
  Every result is checked on overlay images.

Writes ``<out>/<sequence>/<frame>.png`` (ground-truth masks, 0/1) for every
clip, plus overlays to check. ``--ncaa`` also writes the NCAA masks from C04.

Usage:
    python scripts/court/c06_annotate_eval.py --anchors experiments/court/C06_court_segmentation/eval_anchors.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.anchors import homography_from_court_points
from hooptrack.court.leagues import SPECS, lines
from hooptrack.court.masks import court_mask
from hooptrack.court.propagate import track_homography
from hooptrack.court.refine import line_evidence, refine, sample_lines, score


def overlay(img, H, spec, text):
    v = img.copy()
    for ln in lines(spec):
        p = cv2.perspectiveTransform(ln[None].astype(np.float64), H)[0]
        cv2.polylines(v, [np.clip(p, -1e4, 1e4).astype(np.int32)], False, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(v, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    return v


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/court_eval"))
    ap.add_argument("--span", type=int, default=20, help="propagate this many frames each way")
    ap.add_argument("--step", type=int, default=5, help="keep one propagated frame every N")
    ap.add_argument("--only", nargs="*", help="only these sequences")
    ap.add_argument("--no-refine", action="store_true", help="use the hand points as they are")
    ap.add_argument("--ncaa", action="store_true", help="also write NCAA masks from the C04 homographies")
    ap.add_argument("--ncaa-step", type=int, default=25)
    args = ap.parse_args()

    cfg = json.loads(args.anchors.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.ncaa:
        z = np.load("outputs/court_homographies/v_00HRwkvvjtQ_c001/homographies.npz")
        d = args.out / "v_00HRwkvvjtQ_c001"
        d.mkdir(exist_ok=True)
        n = 0
        for f, H in zip(z["frames"], z["H"], strict=True):
            if (f - 1) % args.ncaa_step == 0:
                cv2.imwrite(str(d / f"{int(f):06d}.png"), court_mask(H, SPECS["NCAA"], (1280, 720)))
                n += 1
        print(f"NCAA: {n} masks from C04")
    for seq, d in cfg.items():
        if args.only and seq not in args.only:
            continue
        spec = SPECS[d["league"]]
        clip = Clip(seq)
        S, T = sample_lines(lines(spec))
        ev_kw = {"edges": True}
        res = {}
        mdir = args.out / seq
        mdir.mkdir(exist_ok=True)
        for a in d["anchors"]:
            f = a["frame"]
            img = clip.frame(f)
            if "polygon" in a:                        # court area annotated directly
                m = np.zeros(img.shape[:2], np.uint8)
                cv2.fillPoly(m, [np.array(a["polygon"], np.int32)], 1)
                cv2.imwrite(str(mdir / f"{f:06d}.png"), m)
                v = img.copy()
                cv2.polylines(v, [np.array(a["polygon"], np.int32)], True, (0, 255, 0), 3)
                cv2.imwrite(str(args.out / f"{seq}_{f:06d}_polygon.jpg"), v)
                print(f"{seq} {f}: polygon")
                continue
            H0 = homography_from_court_points(a["points"])
            ev = line_evidence(img, None, clip.people(f), **ev_kw)
            ev.valid[a.get("ignore_below", 720):, :] = 0
            a0 = score(H0, S, ev, tangents=T)[0]
            H = H0 if args.no_refine else refine(H0, S, ev, T, taus=tuple(a.get("taus", (8, 4))))
            a1 = score(H, S, ev, tangents=T)[0]
            res[f] = H
            cv2.imwrite(str(args.out / f"{seq}_{f:06d}_anchor.jpg"),
                        np.vstack([overlay(img, H0, spec, f"{f} hand  agree {a0:.2f}"),
                                   overlay(img, H, spec, f"{f} refined agree {a1:.2f}")]))
            print(f"{seq} {f}: line agreement {a0:.2f} (hand) -> {a1:.2f} (refined)")
            for stop in (max(1, f - args.span), min(clip.length, f + args.span)):
                if stop == f:
                    continue
                for st in track_homography(clip.frames(f, stop), H, spec, clip.people, evidence_kw=ev_kw):
                    if (st.frame - f) % args.step == 0 and st.frame not in res:
                        res[st.frame] = st.H
            for g in (max(1, f - args.span), min(clip.length, f + args.span)):     # check the ends
                if g in res:
                    v = overlay(clip.frame(g), res[g], spec, f"{g} (from {f})")
                    cv2.imwrite(str(args.out / f"{seq}_{g:06d}_prop.jpg"), v)
        for f, H in res.items():
            cv2.imwrite(str(mdir / f"{f:06d}.png"), court_mask(H, spec, (1280, 720)))
        print(f"{seq}: {len(list(mdir.glob('*.png')))} ground-truth masks -> {mdir}")


if __name__ == "__main__":
    main()
