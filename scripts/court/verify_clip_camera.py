"""Keep only the court homographies that one fixed camera explains (court chapter, C07).

Reads ``<in>/<sequence>/homographies.npz`` (C04 hand anchors or C05 automatic),
fits the clip camera (``hooptrack.court.clipcam``) and writes
``<out>/<sequence>/homographies.npz`` in the same format, with only the
accepted frames, plus:

* ``H_ptz``: the same frames re-drawn by the clip camera (physically
  consistent, smoother — for the top-down map), ``camera``: its centre;
* ``frames_smooth`` / ``H_smooth`` (C09): the camera's pan / tilt / zoom filtered in
  time, outliers dropped and gaps of up to ``--fill`` frames filled (``--smooth 0 --fill 0``: off);
* ``report.json``: verdict (``ok`` / ``review``), camera, share of frames
  explained, residuals.

A clip whose verdict is ``review`` gets NO homographies.npz: the tracker falls
back to the colour mask and the clip is listed for a person to anchor (C04).

Usage:
    python scripts/court/verify_clip_camera.py --in outputs/court_auto_edges --sequence v_BgwzTUxJaeU_c008
    python scripts/court/verify_clip_camera.py --in outputs/court_auto_edges --sequence v_00HRwkvvjtQ_c001 \
        --reference outputs/court_homographies/v_00HRwkvvjtQ_c001/homographies.npz
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from hooptrack.court.clipcam import ClipCameraConfig, fit_clip_camera, ptz_homography, smooth_ptz
from hooptrack.court.leagues import spec_from_npz, spec_json
from hooptrack.court.propagate import disagreement


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="<in>/<sequence>/homographies.npz")
    ap.add_argument("--sequence", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs/court_verified"))
    ap.add_argument("--tol-m", type=float, default=0.5)
    ap.add_argument("--min-share", type=float, default=0.6)
    ap.add_argument("--ignore-below", type=int, default=575, help="score bar: rows below are not court evidence")
    ap.add_argument("--smooth", type=int, default=15, help="Savitzky-Golay window (frames) on the camera; 0 = off")
    ap.add_argument("--fill", type=int, default=15, help="fill gaps of at most this many frames from the camera")
    ap.add_argument("--reference", type=Path, help="homographies to compare with (measurement only, one sequence)")
    args = ap.parse_args()
    cfg = ClipCameraConfig(tol_m=args.tol_m, min_share=args.min_share)

    for seq in args.sequence:
        t0 = time.time()
        z = np.load(args.inp / seq / "homographies.npz")
        spec = spec_from_npz(z)
        frames, Hs = z["frames"], z["H"]
        cam = fit_clip_camera(frames, Hs, spec, cfg, visible_rows=args.ignore_below)
        res = np.array([cam.ptz[int(f)].residual_m if int(f) in cam.ptz else np.nan for f in frames])
        report = {
            "sequence": seq, "source": str(args.inp), "league": str(z["league"]),
            "verdict": cam.verdict, "camera_plausible": cam.plausible,
            "camera": None if cam.position is None else [round(float(v), 2) for v in cam.position],
            "frames_in": int(len(frames)), "frames_accepted": len(cam.accepted), "share": round(cam.share, 3),
            "residual_median_m": round(float(np.nanmedian(res)), 3) if np.isfinite(res).any() else None,
            "config": vars(cfg), "seconds": 0,
        }
        out = args.out / seq
        out.mkdir(parents=True, exist_ok=True)
        (out / "homographies.npz").unlink(missing_ok=True)
        if cam.verdict == "ok":
            keep = np.isin(frames, cam.accepted)
            Hp = np.array([ptz_homography(cam.position, cam.ptz[int(f)]) for f in frames[keep]])
            extra = {}
            if args.smooth or args.fill:                 # C09: the camera as a smooth signal, short gaps filled
                sm = smooth_ptz(cam.ptz, cam.accepted, args.smooth, args.fill)
                fs = np.array(sorted(sm), int)
                extra = {"frames_smooth": fs, "H_smooth": np.array([ptz_homography(cam.position, sm[f]) for f in fs])}
                report["frames_smooth"] = int(len(fs))
                report["frames_filled"] = int(len(fs) - np.isin(fs, cam.accepted).sum())
            np.savez(out / "homographies.npz", frames=frames[keep], H=Hs[keep], H_ptz=Hp,
                     agreement=z["agreement"][keep], league=z["league"], spec=spec_json(spec),
                     camera=cam.position, residual_m=res[keep], **extra)
            if args.reference:
                r = np.load(args.reference)
                ref = dict(zip(r["frames"].tolist(), r["H"], strict=True))
                size = (1280, args.ignore_below)
                for name, arr in (("H", Hs[keep]), ("H_ptz", Hp)):
                    e = np.array([disagreement(H, ref[int(f)], spec, size)[1]
                                  for f, H in zip(frames[keep], arr, strict=True) if int(f) in ref])
                    report[f"vs_reference_{name}"] = {"frames": int(len(e)),
                                                      "median_m": round(float(np.nanmedian(e)), 3),
                                                      "p90_m": round(float(np.nanpercentile(e, 90)), 3)}
        report["seconds"] = round(time.time() - t0)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({k: v for k, v in report.items() if k != "config"}), flush=True)


if __name__ == "__main__":
    main()
