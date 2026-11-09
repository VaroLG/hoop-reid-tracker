"""B09 figure · Where the ball is found and where it is missed (B08 on the 4 test clips).

Crops around hand-clicked balls, in three rows: found in the air / on the floor, found over a player, missed over
players. White ring = the hand click, orange dot = B08's answer (none = missed). Picked evenly over the clicks of
each kind (no hand choice).

Usage (repo root):
    python scripts/ball/b09_context_sheet.py --out experiments/ball/B09_fused_candidates/context_sheet.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--tracks", type=Path, default=Path("outputs/ball/tracks_b08"))
    ap.add_argument("--per-row", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            ["data.split=val", "data.only=null"]))}
    lab = pd.read_csv("data/ball_labels/val.csv")
    rows = {"found, in the air / on the floor": [], "found, over a player": [], "missed, over players": []}
    for clip, g in lab[lab["visible"] == 1].groupby("clip"):
        if not (args.tracks / f"{clip}.csv").is_file():
            continue
        tr = pd.read_csv(args.tracks / f"{clip}.csv").set_index("frame")
        gt = seqs[clip].load_gt()
        for r in g.itertuples():
            ids, boxes = gt.get(int(r.frame), (np.zeros(0), np.zeros((0, 4))))
            inside = sum(b[0] <= r.x <= b[2] and b[1] <= r.y <= b[3] for b in boxes)
            h = np.median(boxes[:, 3] - boxes[:, 1]) if len(boxes) else 150
            d = float(np.clip(0.12 * h, 6, 48))
            ans = tr.loc[int(r.frame)] if int(r.frame) in tr.index else None
            found = ans is not None and np.hypot(ans["cx"] - r.x, ans["cy"] - r.y) <= d
            item = (clip, int(r.frame), r.x, r.y, None if ans is None else (ans["cx"], ans["cy"]))
            if found and inside == 0:
                rows["found, in the air / on the floor"].append(item)
            elif found and inside >= 1:
                rows["found, over a player"].append(item)
            elif not found and inside >= 1:
                rows["missed, over players"].append(item)
    S, out_rows = 200, []
    for title, items in rows.items():
        pick = [items[k] for k in np.linspace(0, len(items) - 1, args.per_row).round().astype(int)] if items else []
        tiles = []
        for clip, f, x, y, ans in pick:
            img = cv2.imread(str(seqs[clip].frame_path(f)))
            H, W = img.shape[:2]
            x0, y0 = int(np.clip(x - 80, 0, W - 160)), int(np.clip(y - 80, 0, H - 160))
            crop = img[y0:y0 + 160, x0:x0 + 160].copy()
            cv2.circle(crop, (int(x - x0), int(y - y0)), 13, (255, 255, 255), 2, cv2.LINE_AA)
            if ans is not None:
                cv2.circle(crop, (int(ans[0] - x0), int(ans[1] - y0)), 5, (0, 140, 255), -1, cv2.LINE_AA)
            tiles.append(cv2.resize(crop, (S, S), interpolation=cv2.INTER_CUBIC))
        band = np.full((34, S * args.per_row, 3), 24, np.uint8)
        cv2.putText(band, f"{title} ({len(items)} clicks)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2,
                    cv2.LINE_AA)
        out_rows += [band, np.hstack(tiles + [np.zeros((S, S, 3), np.uint8)] * (args.per_row - len(tiles)))]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(out_rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
