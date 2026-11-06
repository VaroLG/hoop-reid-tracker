"""044 · Jersey numbers the way the literature reads them: pose-guided torso crop + a pretrained text reader.

Why
---
033 and 041 failed the gate (about 30 % of the pieces numbered, reads often contradictory). Both read the number
from the WHOLE player box resized to 256 x 128, so the digits are a few pixels, and both learned to classify numbers
from a handful of games. The published pipelines (Koshkina & Elder 2024; most SoccerNet 2025 game-state entries) do
three things differently:

1. **Torso crop from pose keypoints** (shoulders and hips): the number fills the crop instead of 1/20 of it;
2. **A pretrained scene-text reader (PARSeq)**, trained on millions of words, not on our few hundred stubs: zero-shot
   it already reads ~80 % of SoccerNet tracklets right;
3. **Weighted voting over a tracklet's crops**, keeping only confident reads (here PARSeq's own confidence stands in
   for their legibility classifier, which needs labelled legible / illegible crops we do not have).

What this script does
---------------------
For every piece (online tracklet cut at crossings, as in 038; or ground-truth players with ``--players gt``):

* the ``--crops`` tallest clean boxes (no other box covering >= 20 %, >= 50 px tall), as 033;
* pose (Ultralytics YOLO pose, COCO keypoints) on the crop -> torso box between shoulders and hips, a few pixels of
  margin; no confident keypoints -> a fixed torso window (flagged as such);
* PARSeq on the torso crop -> text + confidence (product of the character probabilities); only reads made of 1-2
  digits count;
* piece vote: confidence-weighted counts per number; the piece gets the winner if its weight divided by the number
  of crops is at least ``--min-score``.

Two modes, the same methodology as the rest of the project:

* ``--split train --players gt`` on the 3 held-out train clips, against our typed numbers
  (``data/jersey_labels/train_numbers.csv``): prints coverage / accuracy for a grid of ``--min-score`` and the value
  that keeps the most players numbered with accuracy >= 90 % (criterion fixed before running);
* ``--split val`` on the online run: writes ``<out>/<clip>/pieces.txt`` and ``<out>/summary.json`` in the format of
  033, so ``scripts/jersey_coverage.py`` applies 041's gate unchanged.

Reads are cached per crop (``<out>/<clip>/reads.json``): changing ``--min-score`` is instant.

Install once (laptop, conda): ``pip install timm pytorch-lightning nltk`` (PARSeq through torch.hub) — the pose
weights download by themselves on first use.

Usage (repo root):
    python scripts/jersey_parseq.py --split train --players gt --out outputs/jersey/044_train
    python scripts/jersey_parseq.py --split val --run outputs/runs/011b_no_referee_filter/segmentation `
        --min-score <value from train> --out outputs/runs/044_jersey_parseq
    python scripts/jersey_coverage.py --config configs/player.yaml --run outputs/runs/044_jersey_parseq `
        --reference outputs/runs/029_siglip_ft
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import load_mot_txt, write_mot_txt
from hooptrack.tracking.matching import coverage_matrix
from hooptrack.tracking.stitch import split_at_crossings

TRAIN_CLIPS = ["v_-6Os86HzwCs_c009", "v_2j7kLB-vEEk_c010", "v_4LXTUim5anY_c013"]
GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12               # COCO keypoint indices


# ------------------------------------------------------------------------------------------------ crops and pieces
def pieces_of(pred: dict) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """The online run cut at crossings (038's pieces): {frame: (piece ids, TLBR)}."""
    seg = split_at_crossings(pred, 0.2)
    return {f: (np.array([seg[(f, int(t))] for t in ids]), b) for f, (ids, b) in pred.items()}


def pick_crops(work: dict, n: int, min_h: float = 50.0) -> dict[int, list[tuple[int, np.ndarray]]]:
    """{piece: up to n (frame, TLBR)}: the tallest clean boxes (no other box covering >= 20 %), as 033."""
    picks: dict[int, list] = {}
    for f, (ids, boxes) in work.items():
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(len(boxes), bool)
        for t, b, c in zip(ids, boxes, clean, strict=True):
            if c and b[3] - b[1] >= min_h:
                picks.setdefault(int(t), []).append((int(f), b))
    return {t: sorted(lst, key=lambda x: -(x[1][3] - x[1][1]))[:n] for t, lst in picks.items()}


# ------------------------------------------------------------------------------------------------ models
class TorsoReader:
    """Pose -> torso crop -> PARSeq. ``read(frame_bgr, boxes)`` -> [(text, confidence, used_pose), ...]."""

    def __init__(self, pose_weights: str, device: str | None = None, digits_only: bool = False,
                 save_dir: Path | None = None, n_save: int = 0):
        import torch
        from torchvision import transforms as T
        from ultralytics import YOLO

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pose = YOLO(pose_weights)
        self.parseq = torch.hub.load("baudm/parseq", "parseq", pretrained=True, trust_repo=True).eval().to(self.device)
        # 044b: only digits (and the end token) may be read. PARSeq is a general text reader: zero-shot it reads a
        # "5" as "S" and a "1" as "I" or "L" (044a, train). Masking the other characters' logits turns those reads
        # into the closest digit instead of throwing them away.
        itos = self.parseq.tokenizer._itos
        self.keep = None
        if digits_only:
            self.keep = torch.tensor([t == "[E]" or (len(t) == 1 and t.isdigit()) for t in itos], device=self.device)
        self.save_dir, self.n_save, self.saved = save_dir, n_save, 0
        self.tf = T.Compose([T.Resize(self.parseq.hparams.img_size, T.InterpolationMode.BICUBIC),
                             T.ToTensor(), T.Normalize(0.5, 0.5)])

    def torso(self, crop: np.ndarray) -> tuple[np.ndarray, bool]:
        """Torso window of a player crop: between shoulders and hips from pose, else a fixed window."""
        h, w = crop.shape[:2]
        res = self.pose.predict(crop, imgsz=256, conf=0.25, verbose=False, device=self.device)[0]
        if res.keypoints is not None and len(res.keypoints.data):
            # the person whose box is largest in the crop is the player (others are neighbours' limbs)
            k = int(np.argmax([(b[2] - b[0]) * (b[3] - b[1]) for b in res.boxes.xyxy.cpu().numpy()]))
            kp = res.keypoints.data[k].cpu().numpy()          # (17, 3): x, y, conf
            pts = kp[[L_SH, R_SH, L_HIP, R_HIP]]
            ok = pts[:, 2] > 0.4
            if ok[:2].all() and ok[2:].any():                 # both shoulders and at least one hip
                x1, x2 = pts[ok, 0].min(), pts[ok, 0].max()
                y1, y2 = pts[:2, 1].min(), pts[ok, 1].max()
                pad = 0.1 * max(x2 - x1, 1.0)
                x1, x2 = int(max(0, x1 - pad)), int(min(w, x2 + pad))
                y1, y2 = int(max(0, y1 - pad)), int(min(h, y2 + 0.5 * pad))
                if x2 - x1 >= 8 and y2 - y1 >= 8:
                    return crop[y1:y2, x1:x2], True
        return crop[int(0.15 * h): int(0.55 * h), int(0.15 * w): int(0.85 * w)], False

    def read(self, frame: np.ndarray, boxes: np.ndarray) -> list[tuple[str, float, bool]]:
        from PIL import Image

        H, W = frame.shape[:2]
        torsos, flags = [], []
        for x1, y1, x2, y2 in np.asarray(boxes, int):
            crop = frame[max(0, y1): min(H, y2), max(0, x1): min(W, x2)]
            t, used = self.torso(crop)
            if self.save_dir is not None and self.saved < self.n_save and t.size:   # to look at the torso crops
                self.save_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(self.save_dir / f"torso_{self.saved:03d}.png"), t)
                self.saved += 1
            torsos.append(self.tf(Image.fromarray(cv2.cvtColor(t, cv2.COLOR_BGR2RGB))))
            flags.append(used)
        with self.torch.no_grad():
            logits = self.parseq(self.torch.stack(torsos).to(self.device))
            if self.keep is not None:
                logits[..., ~self.keep[: logits.shape[-1]]] = float("-inf")
            probs = logits.softmax(-1)
            texts, confs = self.parseq.tokenizer.decode(probs)
        return [(s, float(c.prod()), u) for s, c, u in zip(texts, confs, flags, strict=True)]


# ------------------------------------------------------------------------------------------------ voting
def is_number(s: str) -> bool:
    return 1 <= len(s) <= 2 and s.isdigit()


def vote(reads: list[tuple[str, float, bool]], n_crops: int, min_score: float) -> tuple[str | None, float]:
    """Confidence-weighted vote over the digit reads; accepted if winner weight / n_crops >= min_score."""
    w: Counter = Counter()
    for s, c, _ in reads:
        if is_number(s):
            w[s.lstrip("0") or "0"] += c
    if not w:
        return None, 0.0
    num, weight = w.most_common(1)[0]
    score = weight / max(n_crops, 1)
    return (num if score >= min_score else None), round(score, 4)


def read_clip(seq, work: dict, reader: TorsoReader | None, n: int, cache: Path) -> dict[int, list]:
    """{piece: [(text, conf, used_pose), ...]}, cached per crop."""
    picks = pick_crops(work, n)
    key = lambda f, b: f"{f}_" + "_".join(str(int(round(v))) for v in b)  # noqa: E731
    known = json.loads(cache.read_text()) if cache.is_file() else {}
    todo: dict[int, list] = defaultdict(list)
    for lst in picks.values():
        for f, b in lst:
            if key(f, b) not in known:
                todo[f].append(b)
    if todo:
        if reader is None:
            raise SystemExit(f"{cache}: reads missing and no reader loaded")
        for i, f in enumerate(sorted(todo)):
            for b, r in zip(todo[f], reader.read(cv2.imread(str(seq.frame_path(f))), np.array(todo[f])), strict=True):
                known[key(f, b)] = r
            if i % 100 == 0:
                print(f"  {seq.name}: frame {i + 1}/{len(todo)}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known))
    return {t: [tuple(known[key(f, b)]) for f, b in lst] for t, lst in picks.items()}


# ------------------------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--players", default="run", choices=["run", "gt"], help="gt: ground-truth players (train mode)")
    ap.add_argument("--run", type=Path, default=Path("outputs/runs/011b_no_referee_filter/segmentation"))
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--crops", type=int, default=12)
    ap.add_argument("--min-score", type=float, default=None, help="val: the value chosen on train")
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    ap.add_argument("--digits-only", action="store_true", help="044b: PARSeq may only read digits")
    ap.add_argument("--save-crops", type=int, default=0, help="save this many torso crops in <out>/torsos to look at")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seqs = {s.name: s for s in _sequences(apply_overrides(load_config(args.config, []),
                                                            [f"data.split={args.split}", "data.only=null"]))}
    clips = args.clips or (TRAIN_CLIPS if args.split == "train" else sorted(seqs))
    caches = [args.out / c / "reads.json" for c in clips]
    reader = None if all(p.is_file() for p in caches) else TorsoReader(
        args.pose, digits_only=args.digits_only, save_dir=args.out / "torsos", n_save=args.save_crops)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.players == "gt":                          # ---- train: choose min_score against the typed numbers
        truth = pd.read_csv("data/jersey_labels/train_numbers.csv")
        truth = {(r.clip, int(r.gt_id)): str(r.number) for r in truth.itertuples() if str(r.number) != "nan"}
        all_reads = {c: read_clip(seqs[c], seqs[c].load_gt(), reader, args.crops, args.out / c / "reads.json")
                     for c in clips}
        pose_share = np.mean([u for rd in all_reads.values() for lst in rd.values() for *_, u in lst])
        print(f"torso from pose keypoints on {pose_share:.0%} of the crops")
        rows = []
        for ms in GRID:
            n_lab = n_ans = n_ok = 0
            for c, rd in all_reads.items():
                for pid, lst in rd.items():
                    if (c, pid) not in truth:
                        continue
                    n_lab += 1
                    num, _ = vote(lst, len(lst), ms)
                    if num is not None:
                        n_ans += 1
                        n_ok += num == truth[(c, pid)].lstrip("0")
            rows.append({"min_score": ms, "players": n_lab, "numbered": n_ans, "right": n_ok,
                         "coverage": round(n_ans / max(n_lab, 1), 3), "accuracy": round(n_ok / max(n_ans, 1), 3)})
        g = pd.DataFrame(rows)
        g.to_csv(args.out / "train_grid.csv", index=False)
        print(g.to_string(index=False))
        ok = g[g.accuracy >= 0.9]
        print("\nchosen: min_score = " + (f"{ok.loc[ok.coverage.idxmax(), 'min_score']}" if len(ok)
                                          else "none reaches 90 % accuracy"))
        return

    if args.min_score is None:                        # ---- val: one value, chosen on train
        raise SystemExit("--min-score: pass the value chosen on train")
    jersey = {}
    for c in clips:
        work = pieces_of(load_mot_txt(args.run / c / "results" / f"{c}.txt"))
        reads = read_clip(seqs[c], work, reader, args.crops, args.out / c / "reads.json")
        numbers, info = {}, {}
        for pid, lst in reads.items():
            num, score = vote(lst, len(lst), args.min_score)
            info[pid] = (num or "nan", score, len(lst))
            if num is not None:
                numbers[pid] = num
        write_mot_txt(args.out / c / "pieces.txt", [(f, int(t), b, 1.0) for f, (ids, bs) in work.items()
                                                     for t, b in zip(ids, bs, strict=True)])
        jersey[c] = {"numbers": numbers, "reads": info}
        print(f"{c}: {len(numbers)} of {len(reads)} pieces with a number")
    (args.out / "summary.json").write_text(json.dumps({"jersey": jersey, "merges": {}, "min_score": args.min_score,
                                                       "crops": args.crops, "pose": args.pose}, indent=2))
    print(f"\nSaved {args.out / 'summary.json'}; gate: scripts/jersey_coverage.py --run {args.out}")


if __name__ == "__main__":
    main()
