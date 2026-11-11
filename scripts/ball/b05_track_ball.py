"""B05 · Ball tracking over the detector's candidates, measured like B04 on the hand-clicked test frames.

Reads ``b05_detect_all.py``'s candidates, runs ``hooptrack.ball.track.track_ball`` (one ball, physics-limited
moves, isolated false alarms dropped, gaps filled), writes ``<out>/<name>/<clip>_track.csv`` and compares on the
labelled frames:

* **detector alone**: the most confident candidate of the frame, above 0.25 (B04's headline);
* **tracker**: the tracked position of the frame (detection or interpolation), if any.

Same scoring as B04 (``found`` = within one ball diameter of the click).

Usage (repo root):
    python scripts/ball/b05_track_ball.py --dets outputs/ball/dets_b03 --name b05
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b04_eval_ball import score  # noqa: E402

from hooptrack.ball.labels import ball_diameter, load_labels  # noqa: E402
from hooptrack.ball.track import BallTrackConfig, track_ball  # noqa: E402
from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--labels", type=Path, default=Path("data/ball_labels/val.csv"))
    ap.add_argument("--dets", type=Path, default=Path("outputs/ball/dets_b03"))
    ap.add_argument("--name", default="b05")
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/b05"))
    ap.add_argument("--save-tracks", type=Path, default=None,
                    help="also write <dir>/<clip>.csv (frame, cx, cy, d, source): the input of possession and the demo")
    args = ap.parse_args()

    cfg = BallTrackConfig()
    labels = load_labels(args.labels)
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=val", "data.only=null"]))}
    out = args.out / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows = {"detector": [], "tracker": []}
    for clip, lab in labels.groupby("clip"):
        src = args.dets / f"{clip}.csv"
        if not src.is_file():
            print(f"{clip}: no candidates in {args.dets} - skipped")
            continue
        c = pd.read_csv(src)
        dets = {int(f): np.c_[g[["cx", "cy", "d", "conf"]].to_numpy(), np.zeros(len(g))] for f, g in c.groupby("frame")}
        track = track_ball(dets, cfg)
        pd.DataFrame([(f, *v) for f, v in sorted(track.items())],
                     columns=["frame", "cx", "cy", "d", "source"]).to_csv(out / f"{clip}_track.csv", index=False)
        if args.save_tracks is not None:
            args.save_tracks.mkdir(parents=True, exist_ok=True)
            rows_t = [(f, *v) for f, v in sorted(track.items())]
            pd.DataFrame(rows_t, columns=["frame", "cx", "cy", "d", "source"]).to_csv(
                args.save_tracks / f"{clip}.csv", index=False)
        gt = seqs[clip].load_gt()
        for _, r in lab.iterrows():
            f, vis = int(r["frame"]), bool(r["visible"])
            diam = ball_diameter(r["x"], r["y"], gt.get(f, (None, np.zeros((0, 4))))[1]) if vis else 0.0
            for name in rows:
                row = {"clip": clip, "frame": f, "visible": vis, "conf": 0.0, "dist": np.inf, "diam": diam}
                if name == "detector" and f in dets:
                    d = dets[f][np.argmax(dets[f][:, 3])]
                    row["conf"] = float(d[3])
                    if vis:
                        row["dist"] = float(np.hypot(d[0] - r["x"], d[1] - r["y"]))
                if name == "tracker" and f in track:
                    row["conf"] = 1.0
                    if vis:
                        row["dist"] = float(np.hypot(track[f][0] - r["x"], track[f][1] - r["y"]))
                rows[name].append(row)
        n_int = sum(v[3] == "interp" for v in track.values())
        print(f"{clip}: {len(c)} candidates, ball tracked in {len(track)} of {seqs[clip].length} frames "
              f"({n_int} interpolated)")
    table = []
    for name, rs in rows.items():
        df = pd.DataFrame(rs)
        for clip, g in list(df.groupby("clip")) + [("ALL", df)]:
            table.append({"method": name, "clip": clip, **score(g, 0.25 if name == "detector" else 0.5)})
    t = pd.DataFrame(table)
    t.to_csv(out / "per_clip.csv", index=False)
    print("\n" + t[["method", "clip", "visible", "found", "wrong_place", "missed", "false_alarm_no_ball", "recall",
                   "precision", "median_px"]].to_string(index=False))
    print(f"\ntracker config: {cfg}")


if __name__ == "__main__":
    main()
