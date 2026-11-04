"""B02 · Label the ball by hand: one click on its centre per frame (ball chapter).

Why
---
SportsMOT has no ball labels, and the COCO detector is not good enough to pseudo-label them (B01: ball found in
8-43 % of the frames, confident false alarms in the stands). A ball is almost a point: one click on its centre is
enough; the box size is derived later from the players around it (a ball is ~1/8 of a player's height), so nobody
draws boxes.

Two label sets, never mixed:

* ``--split train`` (default: every basketball train clip): the training labels of the ball detector;
* ``--split val`` (default: the 4 validation clips of B01, one per arena): the evaluation labels. The detector is
  never trained on them.

How to use
----------
A window shows one frame every ``--stride`` frames:

* **left click** on the centre of the ball: saved, next frame;
* **space** (or ``n``): the ball is not visible in this frame (hidden, out of the picture, in a player's hands so
  that it cannot be seen): saved as not visible, next frame;
* **Backspace**: go back one frame (its label is removed and asked again);
* close the window to stop. Labels are saved after every frame in ``--out`` (CSV: clip, frame, x, y, visible) and
  the next session continues where this one stopped.

Usage (repo root):
    python scripts/ball/b02_label_ball.py --split train --out data/ball_labels/train.csv
    python scripts/ball/b02_label_ball.py --split val --out data/ball_labels/val.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config

VAL_CLIPS = ["v_00HRwkvvjtQ_c001", "v_4r8QL_wglzQ_c001", "v_5ekaksddqrc_c003", "v_BgwzTUxJaeU_c012"]   # as B01


def read_labels(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return [r for r in csv.reader(fh)][1:]


def write_labels(path: Path, rows: list[list[str]]) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clip", "frame", "x", "y", "visible"])
        w.writerows(rows)
    tmp.replace(path)                                   # never a half-written file


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every train clip / the 4 val clips of B01")
    ap.add_argument("--stride", type=int, default=10, help="one frame in every N")
    ap.add_argument("--out", type=Path, default=None, help="default: data/ball_labels/<split>.csv")
    args = ap.parse_args()
    out = args.out or Path(f"data/ball_labels/{args.split}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    import tkinter as tk

    from PIL import Image, ImageTk

    cfg = load_config(args.config, [])
    seqs = _sequences(apply_overrides(cfg, [f"data.split={args.split}", "data.only=null"]))
    wanted = args.clips or ([s.name for s in seqs] if args.split == "train" else VAL_CLIPS)
    seqs = [s for s in seqs if s.name in wanted]
    todo = [(s, f) for s in seqs for f in range(1, s.length + 1, args.stride)]
    rows = read_labels(out)
    done = {(r[0], int(r[1])) for r in rows}
    queue = [(s, f) for s, f in todo if (s.name, f) not in done]
    print(f"{len(done)} frames labelled, {len(queue)} to go ({len(seqs)} clips, 1 frame in {args.stride})")

    root = tk.Tk()
    root.lift()                                         # open in front of the terminal, then behave normally
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.title("Ball: click its centre | space = not visible | Backspace = back | close = stop")
    scale = min(1.0, (root.winfo_screenwidth() - 40) / 1280, (root.winfo_screenheight() - 140) / 720)
    info = tk.Label(root, font=("Arial", 13))
    info.pack()
    canvas = tk.Canvas(root, width=int(1280 * scale), height=int(720 * scale), cursor="crosshair",
                       highlightthickness=0)
    canvas.pack()
    state = {"i": 0, "history": []}

    def show() -> None:
        if state["i"] >= len(queue):
            print(f"all frames labelled: {out}")
            root.destroy()
            return
        seq, f = queue[state["i"]]
        img = cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2RGB)
        state["sx"], state["sy"] = img.shape[1] / (1280 * scale), img.shape[0] / (720 * scale)
        img = cv2.resize(img, (int(1280 * scale), int(720 * scale)), interpolation=cv2.INTER_AREA)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=state["photo"])
        info.configure(text=f"{seq.name}  frame {f}   ({len(done) + state['i'] + 1} / {len(todo)})")

    def save(x: float | None, y: float | None) -> None:
        seq, f = queue[state["i"]]
        row = [seq.name, str(f), "" if x is None else f"{x:.1f}", "" if y is None else f"{y:.1f}",
               "0" if x is None else "1"]
        rows.append(row)
        write_labels(out, rows)
        state["history"].append(row)
        state["i"] += 1
        show()

    def click(e) -> None:
        save(e.x * state["sx"], e.y * state["sy"])        # back to the frame's own pixels

    def not_visible(_e=None) -> None:
        save(None, None)

    def back(_e=None) -> None:
        if not state["history"]:
            return
        rows.remove(state["history"].pop())
        write_labels(out, rows)
        state["i"] -= 1
        show()

    canvas.bind("<Button-1>", click)
    root.bind("<space>", not_visible)
    root.bind("n", not_visible)
    root.bind("<BackSpace>", back)
    show()
    root.mainloop()
    print(f"saved in {out}")


if __name__ == "__main__":
    main()
