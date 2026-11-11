"""P02 · Mark by hand when the ball is caught and released, on the test clips (possession chapter).

What is labelled
----------------
Two kinds of events, at the frame where they happen:

* ``c`` **catch**: from this frame a player HAS the ball (he catches a pass, picks up a rebound or a loose ball);
* ``r`` **release**: from this frame nobody has it (a pass, a shot, the ball knocked away, lost out of bounds).

A dribble is possession: the bounces in a dribble are NOT releases. Between a catch and the next release a player
has the ball; between a release and the next catch the ball is in the air or loose. The frames in between are
deduced, so only the changes are clicked.

How to use
----------
The window plays the clip; the tracked ball (B05c) is circled to help find it (``b`` hides it):

* ``Right`` / ``Left``: one frame; ``Up`` / ``Down``: 10 frames; ``space``: play / pause;
* ``c`` / ``r``: catch / release at the current frame (shown as green / red marks on the timeline);
* ``Backspace``: delete the event nearest to the current frame (within 5 frames);
* ``Page Down`` / ``Page Up``, the buttons under the timeline, or the keys ``1`` - ``4``: change clip. Close the
  window to stop.
* ``n`` / ``m``: jump to the next / previous **suggestion** (grey marks on the timeline): moments where P01 thinks the
  ball changes hands. They only help to find the moments; mark the events where YOU see them, also where there is no
  suggestion (a suggestion with no real event: just skip it).

Every change is saved at once in ``--out`` (CSV: clip, frame, event); opening the tool again continues.

Usage (repo root):
    python scripts/possession/p02_label_events.py --out data/possession_labels/val_events.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config

TEST_CLIPS = ["v_00HRwkvvjtQ_c001", "v_4r8QL_wglzQ_c001", "v_5ekaksddqrc_c003", "v_BgwzTUxJaeU_c012"]   # as B04
HELD_OUT = ["v_-6Os86HzwCs_c009", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c013"]   # train clips not in the ball weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--clips", nargs="*", default=None, help="default: the 4 test clips (val) / 3 held-out (train)")
    ap.add_argument("--tracks", type=Path, default=Path("outputs/ball/tracks_b05c"), help="ball tracks to draw")
    ap.add_argument("--out", type=Path, default=Path("data/possession_labels/val_events.csv"))
    ap.add_argument("--suggestions", type=Path, default=Path("outputs/possession/p01"),
                    help="<dir>/<clip>_events.csv of P01, drawn in grey on the timeline")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import tkinter as tk

    from PIL import Image, ImageTk

    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            [f"data.split={args.split}", "data.only=null"]))}
    clips = [seqs[c] for c in (args.clips or (TEST_CLIPS if args.split == "val" else HELD_OUT))]
    events: list[list] = []
    if args.out.is_file():
        with args.out.open() as fh:
            events = [[r["clip"], int(r["frame"]), r["event"]] for r in csv.DictReader(fh)]

    def save() -> None:
        tmp = args.out.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["clip", "frame", "event"])
            w.writerows(sorted(events))
        tmp.replace(args.out)

    balls = {}
    for s in clips:
        p = args.tracks / f"{s.name}.csv"
        balls[s.name] = pd.read_csv(p).set_index("frame") if p.is_file() else None

    sugg = {}
    for s in clips:
        p = args.suggestions / f"{s.name}_events.csv"
        sugg[s.name] = sorted(set(pd.read_csv(p)["frame"].astype(int))) if p.is_file() else []

    root = tk.Tk()
    root.lift()
    root.attributes("-topmost", True)
    root.after(500, lambda: root.attributes("-topmost", False))
    root.title("Possession: c = catch | r = release | arrows = move | space = play | n/m = next/prev suggestion"
               " | PgDn/PgUp = clip | b = ball")
    scale = min(1.0, (root.winfo_screenwidth() - 40) / 1280, (root.winfo_screenheight() - 200) / 720)
    Wd, Hd = int(1280 * scale), int(720 * scale)
    info = tk.Label(root, font=("Arial", 13))
    info.pack()
    canvas = tk.Canvas(root, width=Wd, height=Hd, highlightthickness=0)
    canvas.pack()
    timeline = tk.Canvas(root, width=Wd, height=36, bg="#222", highlightthickness=0)
    timeline.pack(pady=4)
    bar = tk.Frame(root)                              # buttons too: not every laptop has Page Up / Page Down
    bar.pack(pady=2)
    st = {"clip": 0, "f": 1, "play": False, "ball": True}

    def clip_events(name: str) -> list[list]:
        return sorted(e for e in events if e[0] == name)

    def state_at(name: str, f: int) -> str:
        last = [e for e in clip_events(name) if e[1] <= f]
        return "-" if not last else ("POSSESSION" if last[-1][2] == "catch" else "AIR / LOOSE")

    def draw() -> None:
        seq = clips[st["clip"]]
        f = st["f"]
        img = cv2.imread(str(seq.frame_path(f)))
        b = balls[seq.name]
        if st["ball"] and b is not None and f in b.index:
            r = b.loc[f]
            cv2.circle(img, (int(r["cx"]), int(r["cy"])), int(max(8, r["d"])), (0, 255, 255), 2)
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (Wd, Hd), interpolation=cv2.INTER_AREA)
        st["photo"] = ImageTk.PhotoImage(Image.fromarray(img))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=st["photo"])
        ev = clip_events(seq.name)
        here = [e[2] for e in ev if e[1] == f]
        info.configure(text=f"clip {st['clip'] + 1}/{len(clips)}  {seq.name}   frame {f}/{seq.length}   "
                            f"state: {state_at(seq.name, f)}   events in clip: {len(ev)}"
                            + (f"   << {here[0].upper()} here" if here else ""))
        timeline.delete("all")
        for g in sugg[seq.name]:
            x = int((g - 1) / max(1, seq.length - 1) * (Wd - 1))
            timeline.create_line(x, 12, x, 24, fill="#888", width=1)
        for e in ev:
            x = int((e[1] - 1) / max(1, seq.length - 1) * (Wd - 1))
            timeline.create_line(x, 4, x, 32, fill="#3c3" if e[2] == "catch" else "#e33", width=2)
        x = int((f - 1) / max(1, seq.length - 1) * (Wd - 1))
        timeline.create_line(x, 0, x, 36, fill="white", width=1)

    def move(df: int) -> None:
        st["f"] = min(max(1, st["f"] + df), clips[st["clip"]].length)
        draw()

    def mark(kind: str) -> None:
        name = clips[st["clip"]].name
        events[:] = [e for e in events if not (e[0] == name and e[1] == st["f"])]
        events.append([name, st["f"], kind])
        save()
        draw()

    def delete(_e=None) -> None:
        name = clips[st["clip"]].name
        near = [e for e in events if e[0] == name and abs(e[1] - st["f"]) <= 5]
        if near:
            events.remove(min(near, key=lambda e: abs(e[1] - st["f"])))
            save()
        draw()

    def jump(forward: bool) -> None:
        sg = sugg[clips[st["clip"]].name]
        nxt = [g for g in sg if g > st["f"] + 2] if forward else [g for g in sg if g < st["f"] - 2]
        if nxt:
            st["f"] = max(1, (nxt[0] if forward else nxt[-1]) - 8)      # a little before, to see it happen
            draw()

    def change_clip(d: int) -> None:
        st["clip"] = min(max(0, st["clip"] + d), len(clips) - 1)
        st["f"], st["play"] = 1, False
        draw()

    def toggle_play(_e=None) -> None:
        st["play"] = not st["play"]
        if st["play"]:
            tick()

    def tick() -> None:
        if not st["play"]:
            return
        if st["f"] >= clips[st["clip"]].length:
            st["play"] = False
            return
        move(1)
        root.after(60, tick)                          # ~15 frames per second, a bit slower than real time

    root.bind("<Right>", lambda e: move(1))
    root.bind("<Left>", lambda e: move(-1))
    root.bind("<Up>", lambda e: move(10))
    root.bind("<Down>", lambda e: move(-10))
    root.bind("<space>", toggle_play)
    root.bind("c", lambda e: mark("catch"))
    root.bind("r", lambda e: mark("release"))
    root.bind("<BackSpace>", delete)
    tk.Button(bar, text="<< previous clip", command=lambda: change_clip(-1)).pack(side="left", padx=6)
    tk.Button(bar, text="next clip >>", command=lambda: change_clip(1)).pack(side="left", padx=6)
    for k in range(len(clips)):                       # keys 1..4: go to that clip
        root.bind(str(k + 1), lambda e, k=k: change_clip(k - st["clip"]))
    root.bind("<Next>", lambda e: change_clip(1))
    root.bind("<Prior>", lambda e: change_clip(-1))
    root.bind("b", lambda e: (st.update(ball=not st["ball"]), draw()))
    root.bind("n", lambda e: jump(True))
    root.bind("m", lambda e: jump(False))
    draw()
    root.mainloop()
    print(f"saved in {args.out}: {len(events)} events")


if __name__ == "__main__":
    main()
