"""Demo video of the best configuration: players (038) by team, ball (B08), possession (P04) and a court minimap.

Broadcast-style drawing:

* **teams**: each player track gets a team from its kit colour (median torso colour of its clean boxes; the 10 longest
  tracks are split into the two groups of 5 with the most uniform kits, every other track joins the closer group).
  Players are drawn in their team's real kit colour, as an ellipse at their feet and a small tag ``A3`` / ``B1``
  (team letter + player number within the team, by first appearance);
* **possession**: the player who has the ball (P04) gets a highlighted ring and a marker over his head; the panel
  top-left says who has it (or "ball in the air") and a running possession bar per team;
* **ball** (B08): a glowing dot with a short trail (dimmer when interpolated);
* **minimap** bottom-right: the court (C09 homographies) with every player's feet in team colours, short trails and
  the ball holder ringed.

Drawing modes (``--mode``):

* ``showcase`` (default): the broadcast look above, for the top of the README. It shows the tracker's identities as
  they are, but small team tags and feet ellipses make identity switches hard to see;
* ``showcase --hide-crowd-tags``: while a player's box is mostly covered by another one (``--crowd-coverage``, 70 %: a
  pile, where the identity is uncertain) and for ``--crowd-hold`` seconds after (0 by default), his tag is not drawn
  and his feet ellipse is neutral instead of a team colour: the video does not claim an identity when the tracker
  cannot know it. Ids and metrics are unchanged;
* ``tracking``: the players chapter's view: one thin box and the tracker's ``#id`` per identity, each identity in its
  own colour, nothing else (no ball, panels or map);
* ``ball``: the ball chapter's view: the ball with a longer trail, the player who has it highlighted and the
  possession panel; the other players only as faint grey ellipses;
* ``honest``: the same video for the evaluation part of the README. Every player gets a **box** whose colour is the
  kit seen in THAT frame (corner brackets: torso colour of a clean box, nearest team; thin brackets in the id's colour
  when another box covers it and the kit cannot be measured), and a tag with the team label AND the tracker's raw id
  (``A3 · 17``), whose colour is the id's team over the clip: an id that jumps to an opponent shows a tag of one
  colour on a box of the other. Every identity switch measured against the ground truth (same counting as
  ``scripts/render_compare.py``) flashes the box red with "ID SWITCH" for half a second (full red frame + "SWITCH"),
  and the panel keeps count of switches and ids used. Uses the ground truth: evaluation only.

Text uses a TrueType font (Segoe UI on Windows, Lato / DejaVu on Linux). Writes H.264 through ffmpeg if present.

Usage (repo root):
    python scripts/possession/p06_render_demo.py --clip v_00HRwkvvjtQ_c001 --out outputs/videos/demo_best_c001.mp4
    python scripts/possession/p06_render_demo.py --clip v_00HRwkvvjtQ_c001 --mode honest `
        --out outputs/videos/demo_honest_c001.mp4
"""

from __future__ import annotations

import argparse
import itertools
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_hota import result_file  # noqa: E402

from hooptrack.cli import _sequences  # noqa: E402
from hooptrack.config import apply_overrides, load_config  # noqa: E402
from hooptrack.court.leagues import lines  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt  # noqa: E402
from hooptrack.integrations.topdown import TopDownConfig, load_homographies  # noqa: E402
from hooptrack.tracking.matching import coverage_matrix  # noqa: E402
from hooptrack.viz import id_color  # noqa: E402

SWITCH = (40, 40, 235)                       # BGR red, honest mode
NEUTRAL = (120, 120, 120)                    # BGR light grey, showcase --hide-crowd-tags

FONTS = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
         "/usr/share/fonts/truetype/lato/Lato-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
BALL = (0, 140, 255)                         # BGR orange
HIGHLIGHT = (60, 230, 255)                   # BGR yellow


def font(size: int):
    for f in FONTS:
        if Path(f).is_file():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


# ------------------------------------------------------------------ teams from kit colours
def kit_colours(seq, players: dict, per_track: int = 15) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{track: (median Lab, median BGR)} of the torso of its clean, tall boxes."""
    picks: dict[int, list[tuple[int, np.ndarray]]] = {}
    for f, (ids, boxes) in players.items():
        cov = coverage_matrix(np.asarray(boxes, float)).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
        for i, b, c in zip(ids, boxes, cov, strict=True):
            if c < 0.1 and b[3] - b[1] > 60:
                picks.setdefault(int(i), []).append((f, b))
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for i, lst in picks.items():
        for k in np.linspace(0, len(lst) - 1, min(per_track, len(lst))).round().astype(int):
            by_frame.setdefault(lst[k][0], []).append((i, lst[k][1]))
    samples: dict[int, list[np.ndarray]] = {}
    for f, lst in by_frame.items():
        img = cv2.imread(str(seq.frame_path(f)))
        for i, (x1, y1, x2, y2) in lst:
            w, h = x2 - x1, y2 - y1
            torso = img[int(y1 + 0.2 * h):int(y1 + 0.45 * h), int(x1 + 0.3 * w):int(x2 - 0.3 * w)]
            if torso.size:
                samples.setdefault(i, []).append(torso.reshape(-1, 3))
    out = {}
    for i, s in samples.items():
        px = np.concatenate(s).astype(np.uint8)
        out[i] = (np.median(cv2.cvtColor(px[None], cv2.COLOR_BGR2LAB)[0].astype(float), axis=0),
                  np.median(px.astype(float), axis=0))
    return out


def split_teams(players: dict, kits: dict) -> tuple[dict[int, int], list[tuple[int, int, int]], list[np.ndarray]]:
    """{track: 0 | 1}, the two display colours (BGR) and the two kit centres (Lab): 10 longest tracks split 5/5 by
    kit, the rest by distance."""
    length = {}
    for ids, _ in players.values():
        for i in ids:
            length[int(i)] = length.get(int(i), 0) + 1
    main = [i for i in sorted(kits, key=lambda i: -length.get(i, 0))][:10]
    if len(main) < 2:
        return {i: 0 for i in length}, [(220, 220, 220), (40, 40, 200)], []
    lab = np.array([kits[i][0] for i in main])
    best, split = np.inf, None
    for a in itertools.combinations(range(len(main)), len(main) // 2):
        if 0 not in a:
            continue
        b = [k for k in range(len(main)) if k not in a]
        cost = sum(((lab[list(g)] - lab[list(g)].mean(0)) ** 2).sum() for g in (a, b))
        if cost < best:
            best, split = cost, (list(a), b)
    centres = [lab[g].mean(0) for g in split]
    team = {main[k]: t for t, g in enumerate(split) for k in g}
    for i in length:
        if i not in team:
            team[i] = int(np.argmin([np.linalg.norm(kits[i][0] - c) for c in centres])) if i in kits else -1
    colours = []
    for t in (0, 1):
        bgr = np.median([kits[main[k]][1] for k in split[t]], axis=0).astype(np.uint8)
        L, a_, b_ = cv2.cvtColor(bgr[None, None], cv2.COLOR_BGR2LAB)[0, 0].astype(int)
        if np.hypot(a_ - 128, b_ - 128) < 20 and L > 130:   # a white / light grey kit: draw it white
            colours.append((240, 240, 240))
            continue
        hsv = cv2.cvtColor(bgr[None, None], cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        hsv[1] = min(255, int(hsv[1] * 1.4))          # a bit more vivid than the kit under arena light
        hsv[2] = max(170, hsv[2])
        colours.append(tuple(int(v) for v in cv2.cvtColor(np.uint8(hsv)[None, None], cv2.COLOR_HSV2BGR)[0, 0]))
    return team, colours, centres


def corners(img: np.ndarray, box: tuple[int, int, int, int], col, thick: int = 2, frac: float = 0.22) -> None:
    """Honest mode: a box drawn as four corner brackets (lighter on the image than a full rectangle)."""
    x1, y1, x2, y2 = box
    lx, ly = max(6, int(frac * (x2 - x1))), max(6, int(frac * (y2 - y1)))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * lx, cy), col, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ly), col, thick, cv2.LINE_AA)


def dashed_rect(img: np.ndarray, box: tuple[int, int, int, int], col, dash: int = 8) -> None:
    """Honest mode: a box whose kit cannot be measured in this frame (another box covers it)."""
    x1, y1, x2, y2 = box
    for a, b in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) // dash))
        for k in range(0, n, 2):
            p = (int(a[0] + (b[0] - a[0]) * k / n), int(a[1] + (b[1] - a[1]) * k / n))
            q = (int(a[0] + (b[0] - a[0]) * min(k + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(k + 1, n) / n))
            cv2.line(img, p, q, col, 2, cv2.LINE_AA)


def box_kits(img: np.ndarray, boxes: np.ndarray, centres: list[np.ndarray]) -> list[int]:
    """Honest mode: the team of the kit seen in this frame, per box (0 / 1), -1 if covered by another box."""
    if not centres or not len(boxes):
        return [-1] * len(boxes)
    boxes = np.asarray(boxes, float)
    cov = coverage_matrix(boxes).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out = []
    for (x1, y1, x2, y2), c in zip(boxes, cov, strict=True):
        w, h = x2 - x1, y2 - y1
        t = lab[max(0, int(y1 + 0.2 * h)):int(y1 + 0.45 * h), max(0, int(x1 + 0.3 * w)):int(x2 - 0.3 * w)]
        if c >= 0.2 or t.size < 30:
            out.append(-1)
            continue
        m = np.median(t.reshape(-1, 3).astype(float), axis=0)
        out.append(int(np.argmin([np.linalg.norm(m - k) for k in centres])))
    return out


# ------------------------------------------------------------------ drawing helpers
def text_colour(bgr) -> tuple[int, int, int]:
    return (15, 15, 15) if 0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2] > 150 else (250, 250, 250)


def rgb(bgr) -> tuple[int, int, int]:
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def court_panel(spec, scale: float):
    """Static minimap: dark rounded panel, court in slate, light lines. Returns (RGBA image, metres -> px)."""
    pad = 14
    w, h = int(spec.length * scale) + 2 * pad, int(spec.width * scale) + 2 * pad
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=14, fill=(18, 22, 30, 215))
    d.rectangle([pad, pad, w - pad, h - pad], fill=(46, 54, 68, 255))

    def to_px(p):
        p = np.asarray(p, float)
        return p * scale + pad

    for pl in lines(spec):
        pts = [tuple(v) for v in to_px(pl)]
        d.line(pts, fill=(205, 212, 225, 255), width=2)
    return panel, to_px


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--players", type=Path, default=Path("outputs/runs/038_continuity"))
    ap.add_argument("--ball", type=Path, default=Path("outputs/ball/tracks_b08"))
    ap.add_argument("--possession", type=Path, default=Path("outputs/possession/p04"))
    ap.add_argument("--court-dir", default="outputs/court_complete/c09_pipeline")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--hide-crowd-tags", action="store_true",
                    help="showcase: no tag and a neutral ellipse while a box is covered by another (and just after)")
    ap.add_argument("--crowd-hold", type=float, default=0.0, help="seconds the tag stays hidden after a crowd")
    ap.add_argument("--crowd-coverage", type=float, default=0.7,
                    help="a box is 'in a crowd' when it covers / is covered by another this much (0.2 hid 59 %% of "
                         "the tags on c001: boxes touch all the time in basketball)")
    ap.add_argument("--start", type=float, default=0.0, help="seconds: first frame drawn")
    ap.add_argument("--end", type=float, default=None, help="seconds: last frame drawn (default: the whole clip)")
    ap.add_argument("--mode", choices=["showcase", "honest", "tracking", "ball"], default="showcase",
                    help="showcase: broadcast look; honest: boxes, raw ids, kit per frame and ID switches flashed")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seq = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                           ["data.split=val", "data.only=null"]))}[args.clip]
    players = load_mot_txt(result_file(args.players, args.clip))
    ball = pd.read_csv(args.ball / f"{args.clip}.csv").set_index("frame")
    holder = pd.read_csv(args.possession / f"{args.clip}_holder.csv").set_index("frame")["holder"]
    court = load_homographies(TopDownConfig(dir=args.court_dir), args.clip)

    team, colours, centres = split_teams(players, kit_colours(seq, players))
    honest = args.mode == "honest"
    tracking = args.mode == "tracking"           # players chapter: a box and an id per identity, nothing else
    ballmode = args.mode == "ball"               # ball chapter: the ball, its trail and the holder; players faint
    flash: dict[int, list[int]] = {}             # frame -> run ids flashing (honest mode)
    switch_frames: list[int] = []
    if honest:                                   # identity switches against the ground truth (evaluation only)
        from render_compare import switch_records

        from hooptrack.eval.gt_teams import player_teams

        recs = switch_records(players, seq.load_gt(), seq.length, player_teams(seq))
        for r in recs:
            switch_frames.append(r["frame"])
            for g in range(r["frame"], min(seq.length, r["frame"] + int(seq.fps / 2)) + 1):
                flash.setdefault(g, []).append(r["run_id"])
        print(f"{len(recs)} identity switches against the ground truth")
    first_seen: dict[int, int] = {}
    for f in sorted(players):
        for i in players[f][0]:
            first_seen.setdefault(int(i), f)
    label = {}
    for t, letter in ((0, "A"), (1, "B"), (-1, "?")):
        for k, i in enumerate(sorted((i for i in first_seen if team.get(i, -1) == t), key=first_seen.get)):
            label[i] = f"{letter}{k + 1}" + (f" · {i}" if honest else "")
    colour = {i: colours[team[i]] if team.get(i, -1) >= 0 else (160, 160, 160) for i in first_seen}

    first = cv2.imread(str(seq.frame_path(1)))
    H0, W0 = first.shape[:2]
    W, H = args.width, int(round(H0 * args.width / W0 / 2)) * 2
    panel, to_px = court_panel(court[0], scale=8.0) if court else (None, None)
    f_tag, f_hud, f_small = font(15), font(17), font(12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                 "-s", f"{W}x{H}", "-r", str(seq.fps), "-i", "-", "-c:v", "libx264", "-preset",
                                 "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", str(args.out)], stdin=subprocess.PIPE)
        write, close = proc.stdin.write, lambda: (proc.stdin.close(), proc.wait())
    else:
        vw = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (W, H))
        write, close = (lambda b: vw.write(np.frombuffer(b, np.uint8).reshape(H, W, 3))), vw.release

    trail_ball: deque = deque(maxlen=25 if ballmode else 10)
    trail_map: dict[int, deque] = {}
    held = [0, 0]
    seen_ids: set[int] = set()
    last_crowd: dict[int, int] = {}              # id -> last frame its box was covered / covering (hide mode)
    hold = int(round(args.crowd_hold * seq.fps))
    hide = args.hide_crowd_tags and args.mode != "honest"
    f0 = max(1, int(round(args.start * seq.fps)) + 1)
    f1 = seq.length if args.end is None else min(seq.length, int(round(args.end * seq.fps)))
    for f in range(f0, f1 + 1):
        img = cv2.imread(str(seq.frame_path(f)))
        h = int(holder.get(f, -2))
        ids, boxes = players.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        ht = team.get(h, -1) if h >= 0 else -1
        if ht >= 0:
            held[ht] += 1

        if honest:                               # boxes in the kit seen in this frame, red when a switch just happened
            kits_now = box_kits(img, boxes, centres)
            for i, (x1, y1, x2, y2), k in zip(ids, boxes, kits_now, strict=True):
                hit = int(i) in flash.get(f, [])
                box = (int(x1), int(y1), int(x2), int(y2))
                if hit:                          # a switch: full red frame, thicker
                    cv2.rectangle(img, box[:2], box[2:], SWITCH, 3, cv2.LINE_AA)
                elif k >= 0:                     # corner brackets in the kit seen in this frame
                    corners(img, box, colours[k], 2)
                else:                            # covered by another box: kit not measurable -> thin brackets
                    corners(img, box, colour[int(i)], 1, frac=0.15)
        hidden: set[int] = set()
        if hide and len(boxes):                  # crowd: covered by or covering another box by >= 20 %
            cov = coverage_matrix(np.asarray(boxes, float)) if len(boxes) > 1 else np.zeros((1, 1))
            crowd = (np.maximum(cov, cov.T).max(axis=1) >= args.crowd_coverage if len(boxes) > 1
                     else np.zeros(1, bool))
            for i, c in zip(ids, crowd, strict=True):
                if c:
                    last_crowd[int(i)] = f
                if int(i) in last_crowd and f - last_crowd[int(i)] <= hold:
                    hidden.add(int(i))
        col_now = {int(i): (NEUTRAL if int(i) in hidden or ballmode else colour[int(i)]) for i in ids}
        if tracking:                             # one colour per identity: full thin box
            for i, (x1, y1, x2, y2) in zip(ids, boxes, strict=True):
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), id_color(int(i)), 2, cv2.LINE_AA)
        # feet ellipses: filled, translucent, then outline (the holder: thick, highlighted)
        over = img.copy()
        for i, (x1, _, x2, y2) in zip(ids, boxes, strict=True):
            c = (int((x1 + x2) / 2), int(y2))
            ax = (max(12, int(0.55 * (x2 - x1))), max(5, int(0.16 * (x2 - x1))))
            cv2.ellipse(over, c, ax, 0, 0, 360, col_now[int(i)], -1, cv2.LINE_AA)
        a_fill = 0.0 if (honest or tracking) else (0.18 if ballmode else 0.35)
        img = cv2.addWeighted(over, a_fill, img, 1 - a_fill, 0)
        for i, (x1, y1, x2, y2) in zip(ids, boxes, strict=True):
            c = (int((x1 + x2) / 2), int(y2))
            ax = (max(12, int(0.55 * (x2 - x1))), max(5, int(0.16 * (x2 - x1))))
            if int(i) == h and not tracking:
                cv2.ellipse(img, c, (ax[0] + 4, ax[1] + 3), 0, 0, 360, HIGHLIGHT, 3, cv2.LINE_AA)
                tip = (int((x1 + x2) / 2), int(y1) - 30)        # marker over the head, above the tag
                tri = np.array([[tip[0] - 11, tip[1] - 15], [tip[0] + 11, tip[1] - 15], [tip[0], tip[1]]], np.int32)
                cv2.fillPoly(img, [tri], HIGHLIGHT, cv2.LINE_AA)
            if not (honest or tracking):
                cv2.ellipse(img, c, ax, 0, 0, 360, col_now[int(i)], 1 if ballmode else 2, cv2.LINE_AA)

        # ball: short fading trail and a glowing dot
        if tracking:
            pass
        elif f in ball.index:
            b = ball.loc[f]
            trail_ball.append((int(b["cx"]), int(b["cy"]), b["source"]))
        else:
            trail_ball.clear()
        for k in range(1, len(trail_ball)):
            a = k / len(trail_ball)
            cv2.line(img, trail_ball[k - 1][:2], trail_ball[k][:2], BALL, max(1, int((6 if ballmode else 4) * a)),
                     cv2.LINE_AA)
        if trail_ball and f in ball.index and not tracking:
            x, y, src = trail_ball[-1]
            r = int(max(6, ball.loc[f]["d"] * 0.45))
            glow = img.copy()
            cv2.circle(glow, (x, y), r + 7, BALL, -1, cv2.LINE_AA)
            img = cv2.addWeighted(glow, 0.35 if src == "det" else 0.18, img, 1 - (0.35 if src == "det" else 0.18), 0)
            cv2.circle(img, (x, y), r, (255, 255, 255) if src == "det" else (200, 200, 200), 2, cv2.LINE_AA)

        # everything else with PIL (text and translucent panels)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        for i, (x1, y1, x2, _) in zip(ids, boxes, strict=True):       # player tags
            if int(i) in hidden or ballmode:
                continue                         # hide mode: no identity claimed in a crowd; ball mode: no tags
            if tracking:                         # the tracker's id, in its colour
                t = f"#{int(i)}"
                tw = d.textlength(t, font=f_small)
                d.rectangle([x1, y1 - 17, x1 + tw + 8, y1], fill=rgb(id_color(int(i))) + (235,))
                d.text((x1 + 4, y1 - 16), t, font=f_small, fill=rgb(text_colour(id_color(int(i)))))
                continue
            t = label[int(i)]
            tw = d.textlength(t, font=f_tag)
            cx, top = (x1 + x2) / 2, y1 - 24
            d.rounded_rectangle([cx - tw / 2 - 6, top, cx + tw / 2 + 6, top + 20], radius=6,
                                fill=rgb(colour[int(i)]) + (230,))
            d.text((cx - tw / 2, top + 1), t, font=f_tag, fill=rgb(text_colour(colour[int(i)])))
            if honest and int(i) in flash.get(f, []):
                sw = d.textlength("SWITCH", font=f_small)
                d.rounded_rectangle([cx - sw / 2 - 5, top - 19, cx + sw / 2 + 5, top - 3], radius=5,
                                    fill=rgb(SWITCH) + (235,))
                d.text((cx - sw / 2, top - 18), "SWITCH", font=f_small, fill=(255, 255, 255))
        # HUD: possession and running share
        if tracking:
            pil = Image.alpha_composite(pil, layer)
            out = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            write(cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA).tobytes())
            continue
        d.rounded_rectangle([16, 14, 376, 54 if ballmode else 104], radius=12, fill=(18, 22, 30, 200))
        if h >= 0 and int(h) in label:
            d.text((30, 22), "POSSESSION", font=f_hud, fill=(170, 178, 192))
            c = colour[int(h)]
            shown = label[int(h)] if int(h) not in hidden else ("TEAM " + label[int(h)][0])
            lw = d.textlength(shown, font=f_hud)
            d.rounded_rectangle([150, 21, 164 + lw, 45], radius=6, fill=rgb(c) + (255,))
            d.text((157, 22), shown, font=f_hud, fill=rgb(text_colour(c)))
        else:
            d.text((30, 22), "BALL IN THE AIR" if h == -1 else "BALL NOT SEEN", font=f_hud, fill=(170, 178, 192))
        d.text((300, 22), f"{f / seq.fps:4.1f}s", font=f_hud, fill=(170, 178, 192))
        tot = max(1, held[0] + held[1])
        share = held[0] / tot
        x0, x1b, yb = 30, 362, 62
        if ballmode:                             # ball view: no running share (it would start at the window)
            pil = Image.alpha_composite(pil, layer)
            out = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            write(cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA).tobytes())
            continue
        d.rounded_rectangle([x0, yb, x1b, yb + 14], radius=7, fill=rgb(colours[1]) + (255,))
        d.rounded_rectangle([x0, yb, x0 + max(14, (x1b - x0) * share), yb + 14], radius=7,
                            fill=rgb(colours[0]) + (255,))
        d.text((30, 78), f"A {100 * share:.0f}%", font=f_tag, fill=(225, 230, 238))
        d.text((322, 78), f"{100 * (1 - share):.0f}% B", font=f_tag, fill=(225, 230, 238))
        if honest:                               # running count of switches and ids used so far
            n_sw = sum(g <= f for g in switch_frames)
            seen_ids.update(int(i) for i in ids)
            n_ids = len(seen_ids)
            d.rounded_rectangle([16, 112, 376, 146], radius=10, fill=(18, 22, 30, 200))
            d.text((30, 119), f"ID SWITCHES {n_sw}", font=f_hud, fill=rgb(SWITCH) if n_sw else (170, 178, 192))
            d.text((210, 119), f"IDS USED {n_ids} / 10", font=f_hud, fill=(170, 178, 192))
        pil = Image.alpha_composite(pil, layer)

        # minimap
        if panel is not None and not ballmode:
            m = panel.copy()
            dm = ImageDraw.Draw(m)
            if f in court[1] and len(ids):
                feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]].astype(np.float64)
                pts = cv2.perspectiveTransform(feet[None], np.linalg.inv(court[1][f]))[0]
                for i, p in zip(ids, pts, strict=True):
                    q = to_px(p)
                    if not (0 <= q[0] < m.width and 0 <= q[1] < m.height):
                        continue
                    tr = trail_map.setdefault(int(i), deque(maxlen=20))
                    tr.append(tuple(q))
                    if len(tr) > 1:
                        dm.line(list(tr), fill=rgb(col_now[int(i)]) + (110,), width=2)
                    if int(i) == h:
                        dm.ellipse([q[0] - 11, q[1] - 11, q[0] + 11, q[1] + 11], outline=rgb(HIGHLIGHT) + (255,),
                                   width=3)
                        dm.ellipse([q[0] + 5, q[1] - 13, q[0] + 13, q[1] - 5], fill=rgb(BALL) + (255,))
                    dm.ellipse([q[0] - 6, q[1] - 6, q[0] + 6, q[1] + 6], fill=rgb(col_now[int(i)]) + (255,),
                               outline=(255, 255, 255, 255), width=1)
            pil.alpha_composite(m, (W0 - m.width - 16, H0 - m.height - 16))
        out = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        write(cv2.resize(out, (W, H), interpolation=cv2.INTER_AREA).tobytes())
    close()
    print(f"saved {args.out}  (teams by kit: A = {colours[0]}, B = {colours[1]} BGR)")


if __name__ == "__main__":
    main()
