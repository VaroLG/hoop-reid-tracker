"""The pipeline phase by phase on ONE play: a short video that shows what each model adds (for a post or a talk).

The same 7 seconds of a validation clip (default: NCAA 2015, ``v_00HRwkvvjtQ_c001``, 34-41 s, never used for
training) are drawn six times, one per phase, each after a title card that names the model and the number it
moved:

1. **Detection** — first a generic COCO "person" detector (it also sees the crowd and the bench), then the player
   detector (011) with the learned court mask (C06): the boxes kept in green, and in red what the generic detector
   saw that is no longer there (crowd, bench, referees);
2. **Online tracking** (011b) — one box and id per track with its recent path (the motion the Kalman filter
   follows); identity switches against the ground truth flash in red, and the first one is frozen and zoomed in;
3. **Re-identification** (038 + 048) — first how: the tracks cut into clean pieces wherever players overlap (each
   piece in its own colour, a cut flagged), the 10 players as a row of thumbnails, and for each new piece its crop
   and its real appearance distances (fine-tuned SigLIP) to the closest players; then the result: the same frames
   with the final identities, the switches gone;
4. **Court** (C09 + C07) — the court lines from the homography of each frame and the players on the court plan;
5. **Ball** (B08) — the ball and its trail;
6. **Possession and the final video** (P04) — the showcase video of ``p06_render_demo.py``.

It reads results that the experiments already computed (detection caches, tracker runs, court, ball, possession):
nothing is re-run except the C06 mask of the detection phase and the showcase render of phase 6.

Usage (repo root):
    python scripts/make_phases_video.py --out outputs/videos/phases_c001.mp4

Licences: the frames are SportsMOT (CC BY-NC 4.0, Cui et al., ICCV 2023): non-commercial use, with attribution,
which the closing card shows.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "possession")]

import p06_render_demo as p06  # noqa: E402  (fonts, kit colours, teams, minimap)
from render_compare import switch_records  # noqa: E402

from hooptrack.court.clipcam import court_homographies  # noqa: E402
from hooptrack.court.leagues import lines, spec_from_npz  # noqa: E402
from hooptrack.court.propagate import court_polygon  # noqa: E402
from hooptrack.data.sportsmot import load_mot_txt, read_seqinfo  # noqa: E402
from hooptrack.eval.gt_teams import player_teams  # noqa: E402
from hooptrack.tracking.matching import coverage_matrix, iou_matrix  # noqa: E402
from hooptrack.tracking.stitch import split_at_crossings  # noqa: E402
from hooptrack.viz import id_color  # noqa: E402

GREEN, RED, GREY, CYAN, BALL = (90, 220, 90), (60, 60, 235), (185, 185, 185), (255, 230, 0), (0, 140, 255)
BG = (24, 20, 16)                                         # BGR of the title cards: near-black blue

# The six phases: title, the model(s), and the measured contribution (README, "Final results" and the chapters).
PHASES = [
    ("Detection", "player detector (YOLO26m fine-tuned) + learned court mask (C06)",
     ["a generic detector also boxes the crowd and the bench", "player detector: IDF1 53.1 -> 74.6 on unseen arenas",
      "court mask: false positives 6,897 -> 3,989"]),
    ("Online tracking", "a tracker written from scratch, frame by frame (011b)",
     ["a Kalman filter predicts each player's next box, corrected for the camera's pan",
      "each detection is matched to a prediction by overlap and appearance (ResNet-50)",
      "a player who leaves the frame is remembered and recognised when he returns",
      "weak point: when two players overlap, their ids can swap · IDF1 72.7"]),
    ("Re-identification", "offline, on the whole clip: exactly 10 players (038, 048)",
     ["1. every track is cut where players overlap: 18 tracks -> 170 clean pieces",
      "2. a SigLIP model fine-tuned on players gives each piece an appearance vector",
      "3. each piece goes to the closest of the 10 players not on screen at the same time",
      "IDF1 72.7 -> 89.6 · HOTA 79.8 on 15 validation clips"]),
    ("Court", "learned court keypoints + court type + one-camera check (C09, C07)",
     ["the court found in every frame, no annotation", "players placed on the court plan, in metres",
      "0.28 m median error"]),
    ("Ball", "WASB fine-tuned on 1,245 hand clicks + a one-ball tracker (B08)",
     ["a generic detector finds 11 % of the ball", "ours: 71 % found, 85 % right when it answers", "F1 0.78"]),
    ("Possession", "who has the ball: closest player moving with it, last state kept (P04)",
     ["86 % of the frames right", "the final video: players, court, ball and possession together", ""]),
]


def put(d: ImageDraw.ImageDraw, xy, text: str, size: int, fill=(235, 238, 242), anchor="la") -> None:
    d.text(xy, text, font=p06.font(size), fill=fill, anchor=anchor)


def card(w: int, h: int, n: int, title: str, model: str, notes: list[str]) -> np.ndarray:
    """A title card: phase number, title, model, and what it changes."""
    pil = Image.new("RGB", (w, h), p06.rgb(BG))
    d = ImageDraw.Draw(pil)
    put(d, (90, 190), f"{n} / 6" if n else "", 34, (120, 200, 255))
    size = 76
    while size > 40 and d.textlength(title, font=p06.font(size)) > w - 180:   # fit the title on one line
        size -= 4
    put(d, (90, 240 + (76 - size) // 2), title, size)
    put(d, (92, 340), model, 26, (175, 185, 200))
    bullet = "•  " if n else ""                              # phases get bullets; the opening / closing cards not
    lines_ = [bullet + x for x in notes if x]
    fs = 34
    while fs > 20 and max((d.textlength(x, font=p06.font(fs)) for x in lines_), default=0) > w - 190:
        fs -= 2                                                  # fit the longest line
    for k, line in enumerate(lines_):
        put(d, (92, 420 + int(1.55 * fs) * k), line, fs)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def band(img: np.ndarray, n: int, title: str, model: str, counter: str = "") -> np.ndarray:
    """The caption kept on screen during a phase: phase and model bottom-left, a live counter top-right."""
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
    layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = pil.size
    d.rounded_rectangle([16, h - 96, min(w - 16, 40 + 11 * max(len(model), 2 * len(title))), h - 16], radius=14,
                        fill=(18, 22, 30, 215))
    put(d, (34, h - 88), f"{n} · {title.upper()}", 30, (120, 200, 255))
    put(d, (34, h - 48), model, 20, (215, 220, 230))
    if counter:
        tw = d.textlength(counter, font=p06.font(24))
        d.rounded_rectangle([w - tw - 52, 16, w - 16, 62], radius=12, fill=(18, 22, 30, 215))
        put(d, (w - 34, 26), counter, 24, anchor="ra")
    pil = Image.alpha_composite(pil, layer)
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def court_mask_fn(weights: str, imgsz: int):
    """C06 court-area segmentation, frame by frame (needs Ultralytics; this is the pipeline's own court filter)."""
    from hooptrack.court.segment import CourtSegmenter

    seg = CourtSegmenter(weights, imgsz, 0.25)
    return lambda img, f: seg(img)


def homography_mask_fn(H: dict, spec, shape):
    """Without Ultralytics: the court rectangle from the homography (a stand-in, for previews only)."""
    def fn(img, f):
        m = np.zeros(shape[:2], np.uint8)
        if f in H:
            cv2.fillPoly(m, [court_polygon(spec, H[f]).astype(np.int32)], 1)
        return m
    return fn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="v_00HRwkvvjtQ_c001")
    ap.add_argument("--start", type=float, default=34.0, help="seconds")
    ap.add_argument("--end", type=float, default=41.0, help="seconds")
    ap.add_argument("--data", default="data/SportsMOT/dataset/val")
    ap.add_argument("--coco", default="outputs/detections/yolo26m", help="COCO 'person' detection cache")
    ap.add_argument("--player-det", default="outputs/detections/yolo26m_player", help="player detector cache (011)")
    ap.add_argument("--online", default="outputs/runs/011b_no_referee_filter/segmentation", help="online run (011b)")
    ap.add_argument("--ids038", default="outputs/runs/038_continuity", help="038 run: pieces' embeddings")
    ap.add_argument("--final", default="outputs/runs/048_body_smooth", help="final identities (048)")
    ap.add_argument("--court-dir", default="outputs/court_complete/c09_pipeline")
    ap.add_argument("--ball", default="outputs/ball/tracks_b08")
    ap.add_argument("--possession", default="outputs/possession/p04")
    ap.add_argument("--seg-weights", default="outputs/court_runs/c06_yolo26n_seg/weights/best.pt")
    ap.add_argument("--mask", choices=["c06", "homography"], default="c06",
                    help="court mask of phase 1: c06 (the pipeline's, needs Ultralytics) or a homography stand-in")
    ap.add_argument("--min-conf", type=float, default=0.4, help="detection boxes drawn in phase 1")
    ap.add_argument("--card", type=float, default=2.2, help="seconds of each title card")
    ap.add_argument("--out", type=Path, default=Path("outputs/videos/phases_c001.mp4"))
    args = ap.parse_args()

    c = args.clip
    seq = read_seqinfo(Path(args.data) / c)
    fps = seq.fps
    f0, f1 = int(args.start * fps) + 1, min(seq.length, int(args.end * fps))
    frames = range(f0, f1 + 1)
    W, H = seq.width, seq.height

    def mot(run: str) -> dict:
        nested = Path(run) / c / "results" / f"{c}.txt"
        return load_mot_txt(nested if nested.is_file() else Path(run) / "results" / f"{c}.txt")

    def dets(cache: str) -> dict:                         # detection cache: frame,-1,x,y,w,h,score
        a = np.loadtxt(Path(cache) / f"{c}.txt", delimiter=",", ndmin=2)
        a = a[a[:, 6] >= args.min_conf]
        return {int(f): np.c_[g[:, 2:4], g[:, 2:4] + g[:, 4:6]] for f in np.unique(a[:, 0])
                for g in [a[a[:, 0] == f]]}

    coco, pdet = dets(args.coco), dets(args.player_det)
    online, final = mot(args.online), mot(args.final)
    gt, teams_gt = seq.load_gt(), player_teams(seq)
    z = np.load(Path(args.court_dir) / c / "homographies.npz")
    spec, Hs = spec_from_npz(z), court_homographies(z, use_ptz=True, smooth=True)
    ball = pd.read_csv(Path(args.ball) / f"{c}.csv").set_index("frame")
    mask_fn = (court_mask_fn(args.seg_weights, 512) if args.mask == "c06"
               else homography_mask_fn(Hs, spec, (H, W)))
    print(f"{c}: frames {f0}-{f1}, court mask of phase 1: {args.mask}")

    # identity switches against the ground truth, for phases 2 and 3 (the run id flashes for half a second)
    def flashes(pred: dict) -> tuple[dict[int, list[int]], list[int]]:
        out: dict[int, list[int]] = {}
        recs = [r for r in switch_records(pred, gt, seq.length, teams_gt) if f0 <= r["frame"] <= f1]
        for r in recs:
            for g in range(r["frame"], r["frame"] + int(fps / 2)):
                out.setdefault(g, []).append(r["run_id"])
        return out, [r["frame"] for r in recs]

    # teams of the final identities, for the minimap of phase 4
    team, colours, _ = p06.split_teams(final, p06.kit_colours(seq, final))
    colours = p06.distinct(colours)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                             f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf",
                             "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.out)],
                            stdin=subprocess.PIPE)

    def emit(img: np.ndarray, times: int = 1) -> None:
        for _ in range(times):
            proc.stdin.write(img.tobytes())

    def frame(f: int) -> np.ndarray:
        return cv2.imread(str(seq.frame_path(f)))

    n_card = int(args.card * fps)
    emit(card(W, H, 0, "From a broadcast to who has the ball",
              "the same 7 seconds, one model at a time · NCAA 2015, a validation game never used for training",
              ["detection", "tracking", "identities", "court", "ball", "possession"]), int(3 * fps))

    # 1 · detection: COCO 'person' for the first half, then player detector + C06 court mask
    title, model, notes = PHASES[0]
    emit(card(W, H, 1, title, model, notes), n_card)
    half = f0 + (f1 - f0) // 2
    for f in frames:
        img = frame(f)
        if f < half:
            for x1, y1, x2, y2 in coco.get(f, np.zeros((0, 4))).astype(int):
                cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 2, cv2.LINE_AA)
            img = band(img, 1, title, "a generic detector (COCO 'person'): players, crowd, bench, referees",
                       f"people boxed: {len(coco.get(f, []))}")
        else:
            m = mask_fn(img, f)
            tint = img.copy()
            tint[m > 0] = (0.65 * tint[m > 0] + 0.35 * np.array(GREEN)).astype(np.uint8)
            img = tint
            kept = []
            for x1, y1, x2, y2 in pdet.get(f, np.zeros((0, 4))).astype(int):
                fx, fy = int((x1 + x2) / 2), min(H - 1, int(y2))
                if m[min(H - 1, max(0, fy - 3)), min(W - 1, max(0, fx))] > 0:     # feet on the court: kept
                    kept.append((x1, y1, x2, y2))
            # what the generic detector boxed and the final detection no longer has: crowd, bench, referees
            gone = [b for b in coco.get(f, np.zeros((0, 4))).astype(int)
                    if not kept or iou_matrix(np.array([b], float), np.array(kept, float)).max() < 0.3]
            for x1, y1, x2, y2 in gone:
                cv2.rectangle(img, (x1, y1), (x2, y2), RED, 2, cv2.LINE_AA)
                cv2.line(img, (x1, y1), (x2, y2), RED, 1, cv2.LINE_AA)
                cv2.line(img, (x1, y2), (x2, y1), RED, 1, cv2.LINE_AA)
            for x1, y1, x2, y2 in kept:
                cv2.rectangle(img, (x1, y1), (x2, y2), GREEN, 3, cv2.LINE_AA)
            img = band(img, 1, title, "player detector (011) + court mask (C06): players only",
                       f"players {len(kept)} · discarded {len(gone)} (crowd, bench, referees)")
        emit(img)

    def boxes_at(pred: dict, f: int):
        return pred.get(f, (np.zeros(0), np.zeros((0, 4))))

    def id_box(img, i: int, box, col, text: str, thick: int = 2) -> None:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, thick, cv2.LINE_AA)
        cv2.putText(img, text, (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

    def ids_so_far(pred: dict, f: int) -> int:
        return len({int(i) for g in range(1, f + 1) if g in pred for i in pred[g][0]})

    # 2 · online tracker: boxes, ids and recent paths; switches flash, the first one is frozen and zoomed
    title, model, notes = PHASES[1]
    emit(card(W, H, 2, title, model, notes), int(1.6 * n_card))
    flash, sw = flashes(online)
    sw_recs = [r for r in switch_records(online, gt, seq.length, teams_gt) if f0 <= r["frame"] <= f1]

    def overlap(r) -> float:                                     # how much the switched box overlaps another one
        ids, boxes = boxes_at(online, r["frame"])
        k = [int(i) for i in ids].index(r["run_id"])
        cov = coverage_matrix(np.asarray(boxes, float))
        cov = np.maximum(cov, cov.T)
        np.fill_diagonal(cov, 0)
        return float(cov[k].max()) if len(boxes) > 1 else 0.0

    freeze_at = max(sw_recs, key=overlap)["frame"] if sw_recs else -1    # the clearest crossing of the play
    paths: dict[int, deque] = {}
    frozen = False
    for f in frames:
        img = frame(f)
        ids, boxes = boxes_at(online, f)
        for i, b in zip(ids, boxes, strict=True):
            pth = paths.setdefault(int(i), deque(maxlen=12))
            pth.append((int((b[0] + b[2]) / 2), int(b[3])))
            col = id_color(int(i))
            for k in range(1, len(pth)):                         # the path the motion model follows
                cv2.line(img, pth[k - 1], pth[k], col, 2, cv2.LINE_AA)
        for i, b in zip(ids, boxes, strict=True):
            bad = int(i) in flash.get(f, [])
            id_box(img, int(i), b, RED if bad else id_color(int(i)), ("SWITCH  " if bad else "") + f"#{int(i)}",
                   4 if bad else 2)
        counter = f"ids used: {ids_so_far(online, f)} for 10 players · switches: {sum(s <= f for s in sw)}"
        out = band(img, 2, title, "predict (Kalman + camera motion) -> match (overlap + appearance) -> memory",
                   counter)
        emit(out)
        rec = next((r for r in sw_recs if r["frame"] == f), None)
        if rec is not None and f == freeze_at and not frozen:    # freeze the clearest switch, zoomed in
            frozen = True
            k = [int(i) for i in ids].index(rec["run_id"])
            x1, y1, x2, y2 = boxes[k]
            touching = [b for b in boxes if iou_matrix(np.array([b], float), np.array([boxes[k]], float))[0, 0] > 0]
            ux1, uy1 = min(b[0] for b in touching), min(b[1] for b in touching)
            ux2, uy2 = max(b[2] for b in touching), max(b[3] for b in touching)
            cx, cy, s = (ux1 + ux2) / 2, (uy1 + uy2) / 2, max(ux2 - ux1, (uy2 - uy1) * 1.4) * 0.75 + 40
            crop = img[int(max(0, cy - s / 1.4)):int(min(H, cy + s / 1.4)), int(max(0, cx - s)):int(min(W, cx + s))]
            zoom = cv2.resize(crop, (int(crop.shape[1] * 380 / crop.shape[0]), 380))[:, :560]
            zx = W - 24 - zoom.shape[1] if cx < W / 2 else 24           # the zoom goes on the other side
            still = out.copy()
            still[80:80 + zoom.shape[0], zx:zx + zoom.shape[1]] = zoom
            cv2.rectangle(still, (zx, 80), (zx + zoom.shape[1], 80 + zoom.shape[0]), RED, 3)
            pil = Image.fromarray(cv2.cvtColor(still, cv2.COLOR_BGR2RGB)).convert("RGBA")
            layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            ty = 80 + zoom.shape[0] + 10
            tx = int(np.clip(zx, 24, W - 24 - 620))                     # the text box stays on screen
            d.rounded_rectangle([tx, ty, tx + 620, ty + 74], radius=12, fill=(18, 22, 30, 230))
            msg = [f"two players overlap: id #{rec['run_id']} jumps to the other player",
                   "the motion and the appearance of both fit, so the matching has to guess"]
            for j, line in enumerate(msg):
                put(d, (tx + 14, ty + 10 + 30 * j), line, 21 if j == 0 else 17,
                    (255, 255, 255) if j == 0 else (210, 215, 225))
            pil = Image.alpha_composite(pil, layer).convert("RGB")
            emit(cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR), int(2.5 * fps))

    # 3 · re-identification: pieces, the 10 players, the appearance distances; then the result
    title, model, notes = PHASES[2]
    emit(card(W, H, 3, title, model, notes), int(1.8 * n_card))
    seg = split_at_crossings(online, 0.2)                        # (frame, track) -> piece, as 038 does
    piece_frames: dict[int, list[int]] = {}
    for (g, _), pc in seg.items():
        piece_frames.setdefault(pc, []).append(g)
    owner: dict[int, list[int]] = {}                             # piece -> final ids of its boxes (038 keeps the boxes)
    r38 = mot(args.ids038)
    for g, (ids, boxes) in online.items():
        if g not in r38:
            continue
        fids, fboxes = r38[g]
        for i, b in zip(ids, boxes, strict=True):
            k = int(np.argmin(np.abs(fboxes - b).sum(axis=1))) if len(fboxes) else -1
            if k >= 0:
                owner.setdefault(seg[(g, int(i))], []).append(int(fids[k]))
    player_of = {pc: max(set(v), key=v.count) for pc, v in owner.items()}
    z = np.load(Path(args.ids038) / c / "embeddings.npz", allow_pickle=True)
    vec = {int(pc): (lambda m: m / np.linalg.norm(m))(np.asarray(e).mean(axis=0)) for pc, e in
           zip(z["tids"], z["feats"], strict=True)}
    count: dict[int, int] = {}
    for pc, pl in player_of.items():
        count[pl] = count.get(pl, 0) + len(piece_frames[pc])
    roster = sorted(count, key=lambda p: -count[p])[:10]         # the 10 players, by time on screen
    number = {p: k + 1 for k, p in enumerate(roster)}

    def distances(pc: int) -> list[tuple[int, float]]:
        """Cosine distance of a piece to each player: the mean vector of the player's OTHER pieces."""
        out = []
        for p in roster:
            others = [vec[q] for q, pl in player_of.items() if pl == p and q != pc and q in vec]
            if others and pc in vec:
                m = np.mean(others, axis=0)
                out.append((p, float(1 - vec[pc] @ (m / np.linalg.norm(m)))))
        return sorted(out, key=lambda x: x[1])

    thumbs = {}                                                  # one clean, tall crop per player
    for p in roster:
        best = None
        for g, (ids, boxes) in r38.items():
            for i, b in zip(ids, boxes, strict=True):
                if int(i) == p and (best is None or b[3] - b[1] > best[2][3] - best[2][1]):
                    best = (g, int(i), b)
        x1, y1, x2, y2 = (int(v) for v in best[2])
        thumbs[p] = cv2.resize(cv2.imread(str(seq.frame_path(best[0])))[max(0, y1):y2, max(0, x1):x2], (44, 84))

    def roster_strip(img: np.ndarray) -> None:
        x = W - 16 - 54 * len(roster)
        cv2.rectangle(img, (x - 10, 70), (W - 10, 186), (30, 22, 18), -1)
        for p in roster:
            img[80:164, x:x + 44] = thumbs[p]
            cv2.rectangle(img, (x, 80), (x + 44, 164), id_color(1000 + number[p]), 2)
            cv2.putText(img, f"P{number[p]}", (x + 4, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        id_color(1000 + number[p]), 2, cv2.LINE_AA)
            x += 54

    starts = {min(fr): pc for pc, fr in piece_frames.items() if f0 <= min(fr) <= f1 and len(fr) >= 10}
    shown: tuple[int, int] | None = None                         # (piece, until frame) of the info card
    cuts_here = 0
    for f in frames:
        img = frame(f)
        ids, boxes = boxes_at(online, f)
        new_pieces = []
        for i, b in zip(ids, boxes, strict=True):
            pc = seg[(f, int(i))]
            new = min(piece_frames[pc]) == f and f > 1
            new_pieces += [pc] if new else []
            pl = player_of.get(pc)
            tag = f"piece {pc}" + (f" -> P{number[pl]}" if pl in number else "")
            id_box(img, pc, b, id_color(pc), ("CUT  " if new else "") + tag, 3 if new else 2)
        cuts_here += len(new_pieces)
        roster_strip(img)
        if f in starts and (shown is None or f > shown[1]):
            shown = (starts[f], f + int(1.6 * fps))
        if shown and f <= shown[1]:                              # the info card of the newest piece
            pc = shown[0]
            g = min(piece_frames[pc])
            ids0, boxes0 = boxes_at(online, g)
            k = [seg[(g, int(i))] for i in ids0].index(pc)
            x1, y1, x2, y2 = (int(v) for v in boxes0[k])
            crop = cv2.resize(cv2.imread(str(seq.frame_path(g)))[max(0, y1):y2, max(0, x1):x2], (60, 114))
            pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
            layer = Image.new("RGBA", pil.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            d.rounded_rectangle([W - 420, H - 270, W - 16, H - 110], radius=14, fill=(18, 22, 30, 230))
            put(d, (W - 320, H - 260), f"new piece {pc}: who is it?", 22)
            for j, (p, dist) in enumerate(distances(pc)[:3]):
                mark = "  <- given" if p == player_of.get(pc) else ""
                put(d, (W - 320, H - 222 + 30 * j), f"P{number[p]}  appearance distance {dist:.2f}{mark}", 18,
                    (120, 230, 140) if mark else (210, 215, 225))
            pil = Image.alpha_composite(pil, layer)
            img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
            img[H - 262:H - 148, W - 404:W - 344] = crop
        emit(band(img, 3, title, "cut where players overlap -> one appearance vector per piece -> closest player",
                  f"18 tracks -> 170 pieces in the clip · cuts in this play: {cuts_here}"))

    flash, sw = flashes(final)
    for f in frames:                                             # the result: final identities
        img = frame(f)
        ids, boxes = boxes_at(final, f)
        for i, b in zip(ids, boxes, strict=True):
            bad = int(i) in flash.get(f, [])
            id_box(img, int(i), b, RED if bad else id_color(int(i)), ("SWITCH  " if bad else "") + f"#{int(i)}",
                   4 if bad else 2)
        emit(band(img, 3, "Re-identification · result", "the same play with the final identities (038 + 048)",
                  f"ids used: {ids_so_far(final, f)} for 10 players · switches: {sum(s <= f for s in sw)}"))

    # 4 · court: the court lines of each frame and the players on the court plan
    title, model, notes = PHASES[3]
    emit(card(W, H, 4, title, model, notes), n_card)
    panel, to_px = p06.court_panel(spec, scale=9.0)
    trails: dict[int, deque] = {}
    for f in frames:
        img = frame(f)
        if f in Hs:
            for pl in lines(spec):
                q = cv2.perspectiveTransform(pl[None].astype(np.float64), Hs[f])[0].astype(np.int32)
                cv2.polylines(img, [q], False, CYAN, 2, cv2.LINE_AA)
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        mp = panel.copy()
        dm = ImageDraw.Draw(mp)
        ids, boxes = final.get(f, (np.zeros(0), np.zeros((0, 4))))
        if f in Hs and len(ids):
            feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]].astype(np.float64)
            for i, p in zip(ids, cv2.perspectiveTransform(feet[None], np.linalg.inv(Hs[f]))[0], strict=True):
                q = to_px(p)
                col = p06.rgb(colours[team[int(i)]]) if team.get(int(i), -1) >= 0 else (160, 160, 160)
                tr = trails.setdefault(int(i), deque(maxlen=25))
                tr.append(tuple(q))
                if len(tr) > 1:
                    dm.line(list(tr), fill=col + (120,), width=2)
                dm.ellipse([q[0] - 7, q[1] - 7, q[0] + 7, q[1] + 7], fill=col + (255,), outline=(255, 255, 255, 255))
        pil.alpha_composite(mp, (W - mp.width - 16, 76))
        img = cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        emit(band(img, 4, title, "court keypoints + court type -> homography per frame (C09), checked (C07)",
                  "players in metres on the court plan"))

    # 5 · ball: the ball and its trail, players faint
    title, model, notes = PHASES[4]
    emit(card(W, H, 5, title, model, notes), n_card)
    trail: deque = deque(maxlen=25)
    found = 0
    for f in frames:
        img = frame(f)
        if f in ball.index:
            b = ball.loc[f]
            trail.append((int(b["cx"]), int(b["cy"])))
            found += b["source"] != "interp"
        else:
            trail.clear()
        for k in range(1, len(trail)):
            cv2.line(img, trail[k - 1], trail[k], BALL, max(1, int(6 * k / len(trail))), cv2.LINE_AA)
        if trail and f in ball.index:
            glow = img.copy()
            r = int(max(8, ball.loc[f]["d"] * 0.5))
            cv2.circle(glow, trail[-1], r + 8, BALL, -1, cv2.LINE_AA)
            img = cv2.addWeighted(glow, 0.35, img, 0.65, 0)
            cv2.circle(img, trail[-1], r, BALL, 3, cv2.LINE_AA)
        emit(band(img, 5, title, "WASB fine-tuned (B08) + one ball through the clip",
                  f"ball detected in {found} of {f - f0 + 1} frames"))

    # 6 · possession: the showcase video of the same window
    title, model, notes = PHASES[5]
    emit(card(W, H, 6, title, model, notes), n_card)
    tmp = Path(tempfile.mkdtemp(prefix="phases_"))
    subprocess.run([sys.executable, str(ROOT / "scripts" / "possession" / "p06_render_demo.py"), "--clip", c,
                    "--players", args.final, "--ball", args.ball, "--possession", args.possession,
                    "--court-dir", args.court_dir, "--start", str(args.start), "--end", str(args.end),
                    "--width", str(W), "--out", str(tmp / "show.mp4")], check=True)
    cap = cv2.VideoCapture(str(tmp / "show.mp4"))
    while True:
        ok, img = cap.read()
        if not ok:
            break
        emit(band(cv2.resize(img, (W, H)), 6, title, "possession (P04) + everything above: the final video"))
    cap.release()
    shutil.rmtree(tmp, ignore_errors=True)

    # closing card: the final numbers, the code, the data's attribution
    emit(card(W, H, 0, "HoopTrack", "github.com/VaroLG/hoop-reid-tracker · code AGPL-3.0",
              ["15 validation clips: HOTA 79.8 · IDF1 89.6 · MOTA 96.6", "ball F1 0.78 · possession 86 % of frames",
               "clips: SportsMOT (Cui et al., ICCV 2023), CC BY-NC 4.0"]), int(4 * fps))
    proc.stdin.close()
    proc.wait()
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
