"""P01 · Who has the ball in every frame, from the ball tracks (B05c) and the player tracks (038).

Writes ``<out>/<clip>_holder.csv`` (frame, holder: player track id, -1 = ball in the air or loose, -2 = no ball) and
``<out>/<clip>_events.csv`` (frame, event, from, to), with ``hooptrack.ball.possession`` and its fixed rules.

Usage (repo root):
    python scripts/possession/p01_possession.py --tracks outputs/ball/tracks_b05c `
        --players outputs/runs/038_continuity `
        --out outputs/possession/p01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_hota import result_file  # noqa: E402

from hooptrack.ball.possession import PossessionConfig, events, possession  # noqa: E402
from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--tracks", type=Path, default=Path("outputs/ball/tracks_b05c"))
    ap.add_argument("--players", default="outputs/runs/038_continuity",
                    help="a run folder with the player tracks, or 'gt' for the ground-truth boxes (train clips)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--set", nargs="*", default=[], help="PossessionConfig values, e.g. max_drop=10")
    ap.add_argument("--out", type=Path, default=Path("outputs/possession/p01"))
    args = ap.parse_args()
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            [f"data.split={args.split}", "data.only=null"]))}
    args.out.mkdir(parents=True, exist_ok=True)
    def parse(k: str, v: str):
        t = type(getattr(PossessionConfig, k))
        return v.lower() in ("1", "true", "yes") if t is bool else t(v)

    cfg = PossessionConfig(**{k: parse(k, v) for k, v in (a.split("=") for a in args.set)})
    for src in sorted(args.tracks.glob("*.csv")):
        clip, seq = src.stem, seqs[src.stem]
        t = pd.read_csv(src)
        ball = {int(r.frame): (r.cx, r.cy, r.d) for r in t.itertuples()}
        players = seq.load_gt() if args.players == "gt" else load_mot_txt(result_file(Path(args.players), clip))
        h = possession(ball, players, seq.length, cfg)
        pd.DataFrame({"frame": range(1, seq.length + 1), "holder": h}).to_csv(args.out / f"{clip}_holder.csv",
                                                                             index=False)
        ev = pd.DataFrame(events(h), columns=["frame", "event", "from", "to"])
        ev.to_csv(args.out / f"{clip}_events.csv", index=False)
        print(f"{clip}: held in {(h >= 0).mean():.0%} of the frames, in the air {(h == -1).mean():.0%}, "
              f"no ball {(h == -2).mean():.0%}; {len(ev)} events")
    print(f"config: {cfg}")


if __name__ == "__main__":
    main()
