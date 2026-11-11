"""B06 · Save the final ball tracks (B05c: static filter on the court plan + weak segments out) of some clips.

The tracker's values are the ones chosen on the held-out train clips in B05b (``min_frames`` = 50,
``min_segment_score`` = 8). Output: ``<out>/<clip>.csv`` (frame, cx, cy, d, source), one row per frame with a ball;
the input of possession (P01).

Usage (repo root):
    python scripts/ball/b06_save_tracks.py --dets outputs/ball/dets_b03 --out outputs/ball/tracks_b05c
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hooptrack.ball.track import BallTrackConfig, track_ball
from hooptrack.integrations.ball_static import load_court_h, remove_static

COURT_DIRS = ["outputs/court_complete/c09_pipeline", "outputs/court_verified/train",
              "outputs/court_verified/auto_final", "outputs/court_verified/c09_pipeline"]
MIN_FRAMES, MIN_SEGMENT_SCORE = 50, 8.0            # chosen on train in B05b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", type=Path, default=Path("outputs/ball/dets_b03"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ball/tracks_b05c"))
    ap.add_argument("--court-dirs", nargs="*", default=COURT_DIRS)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for src in sorted(args.dets.glob("*.csv")):
        c = pd.read_csv(src)
        dets = {int(f): np.c_[g[["cx", "cy", "d", "conf"]].to_numpy(), np.zeros(len(g))] for f, g in c.groupby("frame")}
        kept, removed = remove_static(dets, load_court_h(args.court_dirs, src.stem), MIN_FRAMES)
        track = track_ball(kept, BallTrackConfig(min_segment_score=MIN_SEGMENT_SCORE))
        pd.DataFrame([(f, *v) for f, v in sorted(track.items())],
                     columns=["frame", "cx", "cy", "d", "source"]).to_csv(args.out / src.name, index=False,
                                                                          float_format="%.1f")
        print(f"{src.stem}: {removed} static candidates removed, ball in {len(track)} frames")


if __name__ == "__main__":
    main()
