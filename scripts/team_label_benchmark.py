"""Team-label benchmark (experiment 020): which unsupervised labeller gives teams good enough for 019?

019 cuts a tracklet where its team changes for good. It works with ground-truth teams (+7.8 IDF1 on the
4 evaluation clips) and not with OSNet labels (no gain on the 15 val clips). Two kinds of label errors
matter, and this script measures both on GROUND-TRUTH tracks, where the right answer is known:

* **accuracy**: share of labelled boxes that get their player's team. A wrong label alone is harmless (the
  cut needs >= 1 s of the other team), but errors come in runs (a player turned away, a bad cluster).
* **false cuts**: ``split_at_team_changes`` applied to the ground-truth tracks, exactly as 019 applies it
  to the tracker's tracklets. A ground-truth track is ONE player, so every cut here is wrong. A wrong cut
  is cheap only if the connector re-joins the pieces; 019 showed it does not always (NCAA c007, c011).

Crops come from the ground truth (no detector, no tracker), clean only (no other box covering >= 20 %),
one frame in 5: the same boxes 019 labels.

The labeller is CHOSEN on the train split (15 clips, 3 games). The val split (15 clips, 4 other games,
where tracking is measured) is only reported. OSNet 012 was trained on the train players: its train
numbers are optimistic and are compared with care.

Team "ground truth": SportsMOT has player ids but no teams; ``hooptrack.eval.gt_teams`` clusters the
median torso colour of each player over the clip (k = 2). Its own sanity is printed: share of frames with
more than 5 players of one team on screen (impossible in basketball, so a sign of a wrong team).
NOTE: the ``lab`` feature is the same colour as that reference, so its accuracy is flattering.

Usage (repo root):
    python scripts/team_label_benchmark.py --split train --sources lab colour006 osnet osnet_torso
    python scripts/team_label_benchmark.py --split train --sources siglip_torso `
        --siglip weights/siglip-base-patch16-224
Features are cached in ``outputs/team_bench/<split>/<clip>/<source>.npz``; the labelling variants
(k, rejection, UMAP) are recomputed from the cache in seconds.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.data.sportsmot import list_sequences
from hooptrack.detection.teams import torso_colour_features
from hooptrack.eval.gt_teams import player_teams
from hooptrack.tracking.matching import coverage_matrix
from hooptrack.tracking.stitch import split_at_team_changes
from hooptrack.tracking.team_labels import cluster_team_labels, lab_torso, torso_boxes

# colour features have dimensions with different units: standardise instead of L2-normalising
COLOUR_SOURCES = {"lab", "colour006"}


def build_source(name: str, args):
    """``fn(frame, boxes) -> (N, D)`` for one feature source."""
    if name == "lab":
        return lab_torso
    if name == "colour006":
        return torso_colour_features
    if name in ("osnet", "osnet_torso"):
        from hooptrack.reid import build_reid

        reid = build_reid({"type": "osnet", "weights": args.osnet, "device": "cpu"})
        if name == "osnet":
            return reid
        return lambda img, boxes: reid(img, torso_boxes(boxes))
    if name == "siglip_torso":
        from hooptrack.reid.siglip import SigLIPExtractor

        sig = SigLIPExtractor(args.siglip)
        return lambda img, boxes: sig(img, torso_boxes(boxes))
    raise ValueError(name)


def clean_gt_boxes(seq, stride: int):
    """(frame, id, box) of every clean ground-truth box, one frame in ``stride`` (frames 1, 1+stride...)."""
    gt = seq.load_gt()
    out = []
    for f in sorted(gt):
        if (f - 1) % stride:
            continue
        ids, boxes = gt[f]
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(len(boxes), bool)
        out += [(f, int(t), b) for t, b, c in zip(ids, boxes, clean, strict=True) if c]
    return gt, out


def features(seq, source: str, fn, stride: int, cache: Path):
    """Cached ``(keys, feats)`` of one source on one clip."""
    if cache.is_file():
        z = np.load(cache)
        return [tuple(k) for k in z["keys"].tolist()], z["feats"]
    _, items = clean_gt_boxes(seq, stride)
    by_f: dict[int, list] = {}
    for f, t, b in items:
        by_f.setdefault(f, []).append((t, b))
    keys, feats = [], []
    for f in sorted(by_f):
        img = cv2.imread(str(seq.frame_path(f)))
        e = fn(img, np.array([b for _, b in by_f[f]]))
        keys += [(f, t) for t, _ in by_f[f]]
        feats.append(np.asarray(e, np.float32))
    feats = np.concatenate(feats) if feats else np.zeros((0, 1), np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, keys=np.array(keys), feats=feats)
    return keys, feats


def opponent_crossings(gt: dict, team: dict[int, int], cov_min: float = 0.2) -> list[tuple[int, int, int]]:
    """(player a, player b, last frame) of every crossing between two players of DIFFERENT teams.

    A crossing is a run of consecutive frames where one box covers the other by ``cov_min`` or more
    (the tracker's definition of "not clean"). These are the places where an id can be stolen by an
    opponent, the case 019 repairs.
    """
    overlap: dict[tuple[int, int], list[int]] = {}
    for f, (ids, boxes) in gt.items():
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if len(boxes) < 2:
            continue
        cov = coverage_matrix(boxes)
        cov = np.maximum(cov, cov.T)
        for i, j in zip(*np.nonzero(np.triu(cov >= cov_min, 1)), strict=True):
            a, b = sorted((int(ids[i]), int(ids[j])))
            if a in team and b in team and team[a] != team[b]:
                overlap.setdefault((a, b), []).append(int(f))
    out = []
    for (a, b), frames in overlap.items():
        frames.sort()
        for k, f in enumerate(frames):
            if k + 1 == len(frames) or frames[k + 1] != f + 1:   # last frame of a run
                out.append((a, b, f))
    return out


def detected_swaps(labels: dict, keys: set, gt: dict, crossings: list, fps: float, min_run: int) -> tuple[int, int]:
    """Synthetic stolen ids: does the labeller see the team change? -> (detected, eligible).

    For each opponent crossing (a, b, t), a fake tracklet follows a for 5 s up to t and b for 5 s after,
    exactly what the tracker produces when b steals a's id. It is eligible when both sides have at least
    ``min_run`` clean boxes (so a perfect labeller could detect it), and detected when
    ``split_at_team_changes`` cuts it. Unlabelled boxes (rejected as ambiguous) count against the
    labeller: rejecting everything would otherwise look perfect on false cuts.
    """
    w = int(5 * fps)
    pos = {f: {int(t): b for t, b in zip(ids, np.asarray(bx, float).reshape(-1, 4), strict=True)}
           for f, (ids, bx) in gt.items()}
    det = elig = 0
    for a, b, t in crossings:
        pre = [f for f in range(t - w, t + 1) if a in pos.get(f, {})]
        post = [f for f in range(t + 1, t + w + 1) if b in pos.get(f, {})]
        if sum((f, a) in keys for f in pre) < min_run or sum((f, b) in keys for f in post) < min_run:
            continue
        elig += 1
        fake = {f: (np.array([0]), pos[f][a][None]) for f in pre}
        fake.update({f: (np.array([0]), pos[f][b][None]) for f in post})
        lab = {(f, 0): labels[(f, a)] for f in pre if (f, a) in labels}
        lab.update({(f, 0): labels[(f, b)] for f in post if (f, b) in labels})
        det += len(set(split_at_team_changes(fake, lab, min_run).values())) > 1
    return det, elig


def score(labels: dict, team: dict[int, int], gt: dict, n_clean: int, fps: float, stride: int,
          keys: set | None = None, crossings: list | None = None) -> dict:
    """Accuracy against the reference teams, false cuts on the ground-truth tracks, detected swaps."""
    pairs = [(lab, team[t]) for (f, t), lab in labels.items() if t in team]
    # name the two team clusters after the reference teams (best of the two assignments); "other" is
    # always wrong: the ground truth only has players
    c = Counter(pairs)
    direct = c[(0, 0)] + c[(1, 1)]
    swapped = c[(0, 1)] + c[(1, 0)]
    acc = max(direct, swapped) / max(len(pairs), 1)
    min_run = max(1, round(1.0 * fps / stride))                 # 1 s of labelled boxes, as 019
    seg = split_at_team_changes(gt, labels, min_run)
    n_tracks = len({t for _, (ids, _) in gt.items() for t in ids})
    cuts = len(set(seg.values())) - n_tracks
    minutes = sum(len(ids) for ids, _ in gt.values()) / fps / 60   # player-minutes on screen
    det, elig = detected_swaps(labels, keys or set(), gt, crossings or [], fps, min_run)
    return {"labelled": len(labels) / max(n_clean, 1), "acc": acc,
            "other": sum(lab == 2 for lab, _ in pairs) / max(len(pairs), 1),
            "false_cuts": cuts, "cuts_per_10min": 10 * cuts / max(minutes, 1e-9),
            "swaps_detected": det, "swaps_eligible": elig}


def team_sanity(gt: dict, team: dict[int, int]) -> float:
    """Share of frames where the reference puts more than 5 visible players in one team."""
    bad = 0
    for ids, _ in gt.values():
        n = Counter(team.get(int(t), -1) for t in ids)
        bad += max(n.get(0, 0), n.get(1, 0)) > 5
    return bad / max(len(gt), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data/SportsMOT/dataset")
    ap.add_argument("--splits-dir", default="data/SportsMOT/splits_txt")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--sources", nargs="+", default=["lab", "colour006", "osnet", "osnet_torso"])
    ap.add_argument("--osnet", default="weights/osnet_x1_0_sportsmot.pth")
    ap.add_argument("--siglip", default="weights/siglip-base-patch16-224")
    ap.add_argument("--k", nargs="+", type=int, default=[3, 2])
    ap.add_argument("--reject", nargs="+", type=float, default=[0.0, 0.8, 0.7],
                    help="distance ratio nearest / 2nd nearest to label a box; 0 = label every box")
    ap.add_argument("--umap", nargs="+", type=int, default=[0], help="UMAP dimensions; 0 = no UMAP")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("outputs/team_bench"))
    args = ap.parse_args()

    seqs = list_sequences(args.root, args.split, "basketball", args.splits_dir)
    if args.clips:
        seqs = [s for s in seqs if s.name in set(args.clips)]
    rows, sanity = [], {}
    for source in args.sources:
        fn = None
        for seq in seqs:
            cache = args.out / args.split / seq.name / f"{source}.npz"
            if fn is None and not cache.is_file():
                fn = build_source(source, args)
            keys, feats = features(seq, source, fn, args.stride, cache)
            gt = seq.load_gt()
            if seq.name not in sanity:
                team = player_teams(seq)
                sanity[seq.name] = (team, team_sanity(gt, team), opponent_crossings(gt, team))
            team, _, crossings = sanity[seq.name]
            for k, rej, um in itertools.product(args.k, args.reject, args.umap):
                labels = cluster_team_labels(keys, feats, k=k, reject_ratio=rej or None, umap_dims=um or None,
                                             normalise=source not in COLOUR_SOURCES)
                rows.append({"source": source, "k": k, "reject": rej, "umap": um, "clip": seq.name,
                             **score(labels, team, gt, len(keys), seq.fps, args.stride, set(keys), crossings)})
            print(f"{source:13s} {seq.name}", flush=True)

    df = pd.DataFrame(rows)
    keys = ["source", "k", "reject", "umap"]
    summary = df.groupby(keys).agg(labelled=("labelled", "mean"), acc=("acc", "mean"),
                                   acc_worst=("acc", "min"), other=("other", "mean"),
                                   false_cuts=("false_cuts", "sum"),
                                   cuts_per_10min=("cuts_per_10min", "mean"),
                                   swaps_detected=("swaps_detected", "sum"),
                                   swaps_eligible=("swaps_eligible", "sum")).round(3).reset_index()
    summary["swap_recall"] = (summary.swaps_detected / summary.swaps_eligible.clip(lower=1)).round(3)
    print(f"\n{args.split}: {len(seqs)} clips | reference teams: frames with > 5 players in one team: "
          + ", ".join(f"{v[1]:.1%}" for v in sanity.values()))
    print(summary.sort_values(["swap_recall", "false_cuts"], ascending=[False, True]).to_string(index=False))
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / f"{args.split}_per_clip.csv", index=False)
    summary.to_csv(args.out / f"{args.split}_summary.csv", index=False)
    (args.out / f"{args.split}_reference_sanity.json").write_text(
        json.dumps({k: v[1] for k, v in sanity.items()}, indent=2))
    print(f"\nSaved {args.out}/{args.split}_summary.csv and _per_clip.csv")


if __name__ == "__main__":
    main()
