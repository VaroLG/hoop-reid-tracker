"""Complete and polish a clip's verified court, frame by frame (court chapter, C09).

After C07 + the temporal filter (``verify_clip_camera.py``, ``H_smooth``) two things are left:

* **holes**: frames where the keypoint detector finds too few points (midcourt views,
  close-ups under the basket) — up to 45 frames in a row;
* **misalignment**: C07 re-draws every frame with ONE camera centre, but a zoom lens moves its
  optical centre a little, so on hard clips the re-drawn court sits 0.25–0.3 m away from the
  painted lines; and a temporal filter lags behind fast pans and zooms.

This script, per clip:

1. **fills the holes by following the camera**: from the last good frame before a hole (and the
   first after it) the court is carried frame by frame with the floor's optical flow and
   re-aligned with the painted lines at every step (``propagate.track_homography``, C04); a chain
   that loses the lines is cut (``propagate.trusted``) and where both directions reach a frame the
   court passes gradually from the forward chain to the backward one (``propagate.blend``);
2. **polishes every frame on its painted lines** (``refine.refine``), kept only if the lines agree
   better and the court moved less than ``--max-shift-m``; the image motion between consecutive
   frames is measured on the way (``propagate.floor_motion``);
3. **checks the camera**: a frame the clip camera cannot explain within ``--tol-m`` is dropped;
4. **averages each frame with its neighbours, carried by the image motion**
   (``propagate.flow_smooth``): the median of the polished courts of the ``--k`` frames on each
   side, each moved to this frame with the measured motion. No lag in a fast pan or zoom (the
   motion is measured, not assumed smooth), and no per-frame trembling or isolated jumps.

Output: the input npz with ``H_smooth`` / ``frames_smooth`` replaced by the result (the map and the
court mask read it) and ``H_polish`` / ``frames_polish`` (steps 2-3, frame by frame) for comparison.

Usage:
    python scripts/court/c09_complete_court.py --in outputs/court_verified/c09_pipeline \
        --sequence v_00HRwkvvjtQ_c001 --out outputs/court_complete/c09_pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.clipcam import court_homographies, fit_ptz
from hooptrack.court.leagues import lines, spec_from_npz
from hooptrack.court.propagate import (
    Chain,
    blend,
    court_polygon,
    disagreement,
    floor_motion,
    flow_smooth,
    track_homography,
    trusted,
)
from hooptrack.court.refine import line_evidence, refine, sample_lines, score


def holes(have: set[int], n: int) -> list[tuple[int, int]]:
    out, start = [], None
    for f in range(1, n + 2):
        if f <= n and f not in have:
            start = start or f
        elif start:
            out.append((start, f - 1))
            start = None
    return out


def floor_mask(shape, spec, H, people, ignore_below: int) -> np.ndarray:
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.clip(court_polygon(spec, H), -1e4, 1e4).astype(np.int32)], 255)
    for x1, y1, x2, y2 in np.asarray(people if people is not None else np.zeros((0, 4)), int).reshape(-1, 4):
        mask[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = 0
    mask[ignore_below:, :] = 0
    return mask


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="C07 output: <in>/<sequence>/homographies.npz")
    ap.add_argument("--sequence", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-track", type=int, default=90, help="follow the camera at most this many frames into a hole")
    ap.add_argument("--max-shift-m", type=float, default=1.0, help="a polish may move the court at most this much")
    ap.add_argument("--tol-m", type=float, default=0.5, help="clip-camera check, as in C07")
    ap.add_argument("--k", type=int, default=7, help="neighbours on each side averaged by flow_smooth; 0 = off")
    ap.add_argument("--ignore-below", type=int, default=575, help="score bar: rows below are not court evidence")
    args = ap.parse_args()

    for seq in args.sequence:
        t0 = time.time()
        z = np.load(args.inp / seq / "homographies.npz")
        spec, C = spec_from_npz(z), z["camera"]
        clip = Clip(seq)
        S, T = sample_lines(lines(spec))
        Hs = court_homographies(z)                                       # C07 camera, filtered in time

        # 1. holes: follow the camera from both sides
        filled = 0
        for a, b in holes(set(Hs), clip.length):
            fwd, bwd = {}, {}
            if a - 1 in Hs:
                steps = track_homography(clip.frames(a - 1, min(b, a - 1 + args.max_track)), Hs[a - 1], spec,
                                         clip.people, args.ignore_below)
                fwd = {st.frame: st.H for st in trusted(Chain("forward", a - 1, steps)).steps}
            if b + 1 in Hs:
                steps = track_homography(clip.frames(b + 1, max(a, b + 1 - args.max_track)), Hs[b + 1], spec,
                                         clip.people, args.ignore_below)
                bwd = {st.frame: st.H for st in trusted(Chain("backward", b + 1, steps)).steps}
            both = sorted(set(fwd) & set(bwd))
            for f in range(a, b + 1):
                if f in fwd and f in bwd:            # hand over gradually (one chain per frame makes the court jump)
                    Hs[f] = blend(fwd[f], bwd[f], (f - both[0] + 1) / (len(both) + 1))
                elif f in fwd or f in bwd:
                    Hs[f] = fwd.get(f, bwd.get(f))
                else:
                    continue
                filled += 1

        # 2. polish on the painted lines (+ image motion between frames), 3. clip-camera check
        pol, motion, polished, prev, last = {}, {}, 0, None, None
        for f in sorted(Hs):
            H, img, people = Hs[f], clip.frame(f), clip.people(f)
            if last is not None and last[0] == f - 1:
                floor = floor_mask(img.shape, spec, last[2], last[3], args.ignore_below)
                motion[f], _ = floor_motion(last[1], img, floor)
            last = (f, img, H, people)
            ev = line_evidence(img, None, people)
            ev.valid[args.ignore_below:, :] = 0
            Hr = refine(H, S, ev, T, taus=(8.0, 4.0))
            if (score(Hr, S, ev, tangents=T)[0] >= score(H, S, ev, tangents=T)[0]
                    and disagreement(Hr, H, spec, (1280, args.ignore_below))[1] <= args.max_shift_m):
                H, polished = Hr, polished + 1
            p = fit_ptz(H, C, spec, visible_rows=args.ignore_below, x0=prev)
            if p is None or p.residual_m > args.tol_m:
                continue
            pol[f], prev = H, np.r_[p.rotvec, np.log(p.focal)]

        # 4. each frame averaged with its neighbours, carried by the image motion
        fs = sorted(pol)
        Hp = [pol[f] / pol[f][2, 2] for f in fs]
        final = flow_smooth(fs, Hp, motion, args.k) if args.k else Hp

        out = args.out / seq
        out.mkdir(parents=True, exist_ok=True)
        keep = {k: z[k] for k in z.files if k not in ("frames_smooth", "H_smooth")}
        np.savez(out / "homographies.npz", **keep, frames_smooth=np.array(fs), H_smooth=np.array(final),
                 frames_polish=np.array(fs), H_polish=np.array(Hp))
        report = {"sequence": seq, "source": str(args.inp), "frames": clip.length,
                  "frames_in": int(len(z["frames_smooth"])) if "frames_smooth" in z.files else int(len(z["frames"])),
                  "filled_by_tracking": filled, "polished": polished, "frames_out": len(fs),
                  "coverage": round(len(fs) / clip.length, 3), "seconds": round(time.time() - t0)}
        src = json.loads((args.inp / seq / "report.json").read_text())
        (out / "report.json").write_text(json.dumps({**src, "complete": report}, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
