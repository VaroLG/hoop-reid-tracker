"""Join the fragments of one player after the video is processed (experiment 013).

Reads the output of a finished tracking run, describes every tracklet (all boxes
of one id) by the ReID embeddings of its cleanest crops, merges tracklets that
are the same player (``hooptrack.tracking.stitch``), writes the new output and
compares the metrics before / after.

Nothing is re-tracked: a run of the online tracker is post-processed in minutes,
so the stitching can be tried without the hours of a tracking run. The ReID
model used here is independent of the one the tracker used: the best online
configuration (011b, ResNet in the tracker) + OSNet for the stitching is allowed.

Crops: up to ``--crops`` per tracklet, spread over its life, preferring boxes that
no other box covers (a crop that mixes two players describes neither) and boxes
at least ``--min-height`` px tall. Embeddings are cached per clip
(``<out>/<clip>/embeddings.npz``): changing the thresholds afterwards is instant.

Thresholds: by default from the ReID calibration on the TRAIN split
(``outputs/calibration/<reid>.json``, written by ``scripts/calibrate_reid.py``):

* ``thresh`` = 5th percentile of the hard-negative distances (closest OTHER player):
  only 5 % of "closest other player" crop pairs are that close. A tracklet distance
  is a median over many crop pairs, so it is less noisy than one crop pair.
* ``margin`` = hard-negative p5 − p1: the lead over the second-best candidate must
  be at least the spread of the hardest negatives.

They are never tuned on the validation clips (where the numbers are reported).

Usage (repo root):
    python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
        --config configs/player.yaml --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth `
        --out outputs/runs/013_stitch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import load_mot_txt
from hooptrack.eval import clear_and_identity
from hooptrack.reid import build_reid
from hooptrack.tracking.matching import coverage_matrix
from hooptrack.tracking.stitch import StitchConfig, Tracklet, stitch


def sample_crops(pred: dict, k: int, min_height: float) -> dict[int, list[tuple[int, np.ndarray]]]:
    """id -> up to k (frame, box) picks: clean boxes first, evenly spread over the tracklet's life."""
    clean, anyb = {}, {}
    for f, (ids, boxes) in pred.items():
        cov = coverage_matrix(boxes).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
        for tid, box, c in zip(ids, boxes, cov, strict=True):
            anyb.setdefault(int(tid), []).append((f, box))
            if c < 0.2 and box[3] - box[1] >= min_height:
                clean.setdefault(int(tid), []).append((f, box))
    picks = {}
    for tid, cands in anyb.items():
        pool = sorted(clean.get(tid) or cands, key=lambda x: x[0])
        sel = np.unique(np.linspace(0, len(pool) - 1, min(k, len(pool))).round().astype(int))
        picks[tid] = [pool[i] for i in sel]
    return picks


def embed_tracklets(seq, pred: dict, reid, k: int, min_height: float, cache: Path, tag: str):
    if cache.is_file():
        z = np.load(cache, allow_pickle=True)
        if str(z["tag"]) == tag:
            return {int(t): f for t, f in zip(z["tids"], z["feats"], strict=True)}
    picks = sample_crops(pred, k, min_height)
    by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for tid, lst in picks.items():
        for f, box in lst:
            by_frame.setdefault(f, []).append((tid, box))
    feats: dict[int, list[np.ndarray]] = {}
    for f in sorted(by_frame):                                 # one image read per frame
        img = cv2.imread(str(seq.frame_path(f)))
        tids = [t for t, _ in by_frame[f]]
        emb = reid(img, np.array([b for _, b in by_frame[f]]))
        for t, e in zip(tids, emb, strict=True):
            feats.setdefault(t, []).append(e / (np.linalg.norm(e) + 1e-12))
    out = {t: np.asarray(v, np.float32) for t, v in feats.items()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, tag=tag, tids=np.array(list(out)), feats=np.array(list(out.values()), dtype=object))
    return out


def rewrite(src: Path, dst: Path, mapping: dict[int, int]) -> None:
    """Same MOT file with the ids remapped (all other columns untouched)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split(",")
        cols[1] = str(mapping.get(int(float(cols[1])), int(float(cols[1]))))
        lines.append(",".join(cols))
    dst.write_text("\n".join(lines) + "\n")


def metrics(per_seq: dict) -> pd.DataFrame:
    t = clear_and_identity(per_seq)
    return pd.DataFrame({"IDF1": (100 * t.idf1).round(1), "MOTA": (100 * t.mota).round(1),
                         "IDSW": t.num_switches.astype(int), "Frag": t.num_fragmentations.astype(int)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True,
                    help="finished run: <run>/<clip>/results/<clip>.txt or <run>/results/<clip>.txt")
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. the ReID model used to stitch")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every clip found in --run")
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/013_stitch"))
    ap.add_argument("--crops", type=int, default=12, help="crops per tracklet")
    ap.add_argument("--min-height", type=float, default=50)
    ap.add_argument("--calibration", type=Path, default=None, help="default: outputs/calibration/<reid type>.json")
    ap.add_argument("--thresh", type=float, default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--max-gap", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    rcfg = cfg["reid"]
    calib = args.calibration or Path("outputs/calibration") / f"{rcfg['type']}.json"
    thresh, margin = args.thresh, args.margin
    if thresh is None or margin is None:
        if not calib.is_file():
            raise SystemExit(f"{calib} not found: run scripts/calibrate_reid.py or pass --thresh and --margin")
        hn = json.loads(calib.read_text())["hard_neg_pct"]
        thresh = hn["p5"] if thresh is None else thresh
        margin = round(hn["p5"] - hn["p1"], 3) if margin is None else margin
    scfg = StitchConfig(thresh=thresh, margin=margin, max_gap=args.max_gap)
    tag = f"{rcfg['type']}|{rcfg.get('weights')}|{args.crops}|{args.min_height}"
    print(f"ReID {rcfg['type']} {rcfg.get('weights') or ''} | thresh {thresh} | margin {margin} | max_gap {args.max_gap}")

    seqs = {s.name: s for s in _sequences(apply_overrides(cfg, ["data.only=null"]))}
    # two layouts: run_court_source_eval.py writes <run>/<clip>/results/<clip>.txt,
    # hooptrack track writes <run>/results/<clip>.txt
    def source(clip: str) -> Path:
        nested = args.run / clip / "results" / f"{clip}.txt"
        return nested if nested.is_file() else args.run / "results" / f"{clip}.txt"

    found = [p.name for p in args.run.iterdir() if (p / "results" / f"{p.name}.txt").is_file()]
    if (args.run / "results").is_dir():
        found += [p.stem for p in (args.run / "results").glob("*.txt")]
    clips = args.clips or sorted(set(found))
    reid = build_reid(rcfg)
    before, after, rows, all_logs = {}, {}, [], {}
    for clip in clips:
        src = source(clip)
        pred = load_mot_txt(src)
        feats = embed_tracklets(seqs[clip], pred, reid, args.crops, args.min_height,
                                args.out / clip / "embeddings.npz", tag)
        frames: dict[int, list[int]] = {}
        for f, (ids, _) in pred.items():
            for t in ids:
                frames.setdefault(int(t), []).append(f)
        tracklets = [Tracklet(t, np.array(sorted(fr)), feats.get(t, np.zeros((0, 1), np.float32)))
                     for t, fr in frames.items()]
        mapping, log = stitch(tracklets, scfg)
        dst = args.out / clip / "results" / f"{clip}.txt"
        rewrite(src, dst, mapping)
        gt = seqs[clip].load_gt()
        before[clip], after[clip] = (gt, pred), (gt, load_mot_txt(dst))
        merged = [e for e in log if e["merged"]]
        rows.append({"clip": clip, "ids_before": len(frames), "ids_after": len(set(mapping.values())),
                     "merges": len(merged), "refused_ambiguous": len(log) - len(merged)})
        all_logs[clip] = log
        print(f"{clip}: {len(frames)} ids -> {len(set(mapping.values()))} ({len(merged)} merges, "
              f"{len(log) - len(merged)} refused as ambiguous)")

    mb, ma = metrics(before), metrics(after)
    table = mb.add_suffix(" before").join(ma.add_suffix(" after"))
    table = table[[c for m in ("IDF1", "MOTA", "IDSW", "Frag") for c in (f"{m} before", f"{m} after")]]
    ids = pd.DataFrame(rows).set_index("clip")
    print("\n" + table.to_string() + "\n\n" + ids.to_string())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({
        "run": str(args.run), "reid": rcfg["type"], "reid_weights": rcfg.get("weights"),
        "config": vars(scfg), "metrics": table.reset_index().to_dict(orient="records"),
        "ids": rows, "merges": all_logs}, indent=2, default=str))
    print(f"\nSaved {args.out / 'summary.json'} and the stitched results in {args.out}/<clip>/results/")


if __name__ == "__main__":
    main()
