"""Label the jersey number of every ground-truth player of the TRAIN clips, one number per player (041).

Why
---
The jersey reader of 033 was trained on NBA footage only and reads few numbers on our NCAA / FIBA clips (domain
gap). SportsMOT's ground truth already groups every box of a clip by player, so ONE number typed per player labels
hundreds of crops of that player. Only the train split is labelled: the validation clips stay unseen, so the
reader trained on these labels is measured, as everything else, on games it has never seen.

How it works
------------
For each train clip and each ground-truth player, a window shows 12 crops of him: the tallest boxes that no other
player overlaps (the number is most likely readable there), spread over the clip, enlarged. Type the number:

* digits, then ``Enter``: the number (``0`` and ``00`` are different numbers: type what is printed);
* ``n`` then ``Enter``: no number readable in any crop (back never seen, too small);
* ``Esc``: skip this player for now; close the window to quit.

Labels are saved after every player in ``--out`` (CSV: clip, gt_id, number), and a new session continues where the
last one stopped (labelled players are not shown again; delete a line of the CSV to redo it).

Usage (repo root, needs a screen):
    python scripts/label_jersey_numbers.py --out data/jersey_labels/train_numbers.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.tracking.matching import coverage_matrix

TILE = (128, 256)                          # width, height of each crop on screen
N_CROPS = 12


def player_crops(seq, gt: dict, k: int = N_CROPS) -> dict[int, list[tuple[int, np.ndarray]]]:
    """{gt id: up to k (frame, box)}: clean boxes (no other player covers >= 10 %), tallest 30 %, spread in time."""
    cands: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for f, (ids, boxes) in gt.items():
        boxes = np.asarray(boxes, float)
        cov = coverage_matrix(boxes).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
        for i, b, c in zip(ids, boxes, cov, strict=True):
            if c < 0.1:
                cands.setdefault(int(i), []).append((int(f), b, b[3] - b[1]))
    out = {}
    for i, lst in cands.items():
        h = np.array([x[2] for x in lst])
        tall = sorted((x for x in lst if x[2] >= np.percentile(h, 70)), key=lambda x: x[0])
        pick = np.unique(np.linspace(0, len(tall) - 1, min(k, len(tall))).round().astype(int))
        out[i] = [(tall[j][0], tall[j][1]) for j in pick]
    return out


def sheet(seq, picks: list[tuple[int, np.ndarray]]) -> np.ndarray:
    tiles = []
    for f, (x1, y1, x2, y2) in picks:
        img = cv2.imread(str(seq.frame_path(f)))
        H, W = img.shape[:2]
        crop = img[max(0, int(y1)):min(H, int(y2)), max(0, int(x1)):min(W, int(x2))]
        blank = np.zeros((TILE[1], TILE[0], 3), np.uint8)
        tiles.append(cv2.resize(crop, TILE, interpolation=cv2.INTER_CUBIC) if crop.size else blank)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 6)
    return np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--out", type=Path, default=Path("data/jersey_labels/train_numbers.csv"))
    args = ap.parse_args()

    # tkinter (Python's own GUI) instead of cv2.imshow: the OpenCV wheel in many environments is "headless"
    import tkinter as tk

    from PIL import Image, ImageTk

    cfg = load_config(args.config, [])
    seqs = _sequences(apply_overrides(cfg, ["data.split=train", "data.only=null"]))
    done = set()
    if args.out.is_file():
        with args.out.open() as fh:
            done = {(r["clip"], int(r["gt_id"])) for r in csv.DictReader(fh)}
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("clip,gt_id,number\n")

    def players():                                        # lazily: one clip's crops at a time
        for seq in seqs:
            crops = None
            for gid in sorted(np.unique(np.loadtxt(seq.gt_path, delimiter=",", ndmin=2)[:, 1]).astype(int)):
                if (seq.name, gid) in done:
                    continue
                crops = crops or player_crops(seq, seq.load_gt())
                if gid in crops:
                    yield seq, gid, crops[gid]

    todo = sum(1 for s in seqs for i in np.unique(np.loadtxt(s.gt_path, delimiter=",", ndmin=2)[:, 1])
               if (s.name, int(i)) not in done)
    print(f"{len(done)} players labelled, {todo} to go")
    it = players()

    root = tk.Tk()
    root.lift()                                         # open in front of the terminal, then behave normally
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.title("Jersey numbers: type the number + Enter  |  n + Enter = none readable  |  Esc = skip  |  close = quit")
    info = tk.Label(root, font=("Arial", 14))
    info.pack()
    picture = tk.Label(root)
    picture.pack()
    entry = tk.Entry(root, font=("Arial", 22), width=6, justify="center")
    entry.pack(pady=6)
    state = {"n": 0}

    def show_next() -> None:
        try:
            seq, gid, picks = next(it)
        except StopIteration:
            print(f"all players labelled: {args.out}")
            root.destroy()
            return
        state.update(seq=seq, gid=gid, n=state["n"] + 1)
        img = cv2.cvtColor(sheet(seq, picks), cv2.COLOR_BGR2RGB)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(img))      # keep a reference, or tk drops it
        picture.configure(image=state["photo"])
        info.configure(text=f"{seq.name}   player {gid}   ({state['n']} / {todo})")
        entry.delete(0, tk.END)
        entry.focus_set()

    def save(_event=None) -> None:
        text = entry.get().strip().lower()
        if text == "n":
            text = "none"
        if not (text == "none" or (text.isdigit() and len(text) <= 2)):
            entry.delete(0, tk.END)                        # not a number: type again
            return
        with args.out.open("a", newline="") as fh:
            csv.writer(fh).writerow([state["seq"].name, state["gid"], text])
        show_next()

    entry.bind("<Return>", save)
    entry.bind("<Escape>", lambda _e: show_next())
    show_next()
    root.mainloop()
    print(f"saved in {args.out}")


if __name__ == "__main__":
    main()
