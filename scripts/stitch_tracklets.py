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

Two modes (``--mode``):

* ``gta`` (default): GTA's connector (``stitch_gta``) with its published SportsMOT settings
  (merge distance 0.4, spatial factor 1.0). Reproduces the original code exactly when fed the same
  embeddings (013). Here it uses ``--crops`` sampled crops per tracklet instead of every box.
* ``conservative``: our first rules (``stitch``), with the thresholds below.

``--split-teams`` (019): before stitching, cut a tracklet where its team changes for good
(``split_at_team_changes``): a track that went on with an opponent after a crossing becomes two pieces,
and the connector gives each piece to the right player. ``--team-source`` says where the team of each clean
box comes from:

* ``osnet`` (default): the ReID embedding of every clean box, one frame in 5. The tracklets (mean embedding,
  one vote each) are clustered in 3 groups, two teams and "other" (the group with the fewest boxes, as in
  006), and each box gets the nearest group. Opponents are what OSNet separates best (016).
* ``colour``: the jersey-colour classifier of 006, fitted on the whole clip. Its labels match the true teams
  on only 67-92 % of the boxes of the evaluation clips (019 README), too few for this.
* 020: ``lab`` (median torso colour), ``osnet_torso`` (OSNet on the torso only) and ``siglip`` (SigLIP on the
  torso, ``--siglip`` folder), all with the recipe of ``hooptrack.tracking.team_labels`` and its options
  ``--team-k`` (default 3, as 019), ``--team-reject`` and ``--team-umap``. ``osnet`` with these options
  uses the same recipe; without them it is exactly 019.
* ``gt``: ORACLE, for the ceiling only. Each clean box takes the team of the ground-truth player it overlaps
  (IoU >= 0.5, reference teams of ``hooptrack.eval.gt_teams``). Same boxes (one frame in 5) as the real
  labellers, so the gap to it is what the labels cost.

``--split-crossings`` (015): before stitching, cut every tracklet at the end of each crossing
(``split_at_crossings``), so that a tracklet whose id was swapped in a crossing becomes two clean
segments that the connector can give to the right players. Metrics "before" are always the
original, uncut run.

Thresholds of the conservative mode: by default from the ReID calibration on the TRAIN split
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
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import load_mot_txt, write_mot_txt
from hooptrack.detection.teams import TeamClassifier, TeamClassifierConfig
from hooptrack.eval import clear_and_identity
from hooptrack.eval.gt_teams import player_teams
from hooptrack.reid import build_reid
from hooptrack.tracking.matching import coverage_matrix, iou_matrix
from hooptrack.tracking.stitch import (
    GTAConfig,
    StitchConfig,
    Tracklet,
    assign_closed_set,
    assign_roster,
    refine_closed_set,
    smooth_sandwiches,
    smooth_swapped_excursions,
    split_at_crossings,
    split_at_team_changes,
    stitch,
    stitch_gta,
)
from hooptrack.tracking.team_labels import cluster_team_labels, lab_torso, torso_boxes


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


def team_labels(seq, pred: dict, team_cfg: dict, stride: int = 5) -> dict[tuple[int, int], int]:
    """019: ``{(frame, id): team}`` for every CLEAN box of a finished run (others: unknown).

    Two passes with the 006 colour classifier. 1) Fit: clean boxes every ``stride`` frames over the WHOLE
    clip (online, the tracker only had the first seconds). 2) Label: every clean box of every frame.
    Occluded boxes are not labelled: their torso mixes two players.
    """
    frames = sorted(pred)
    fit_frames = frames[::stride]
    cfg = TeamClassifierConfig.from_dict({**(team_cfg or {}), "warmup_frames": len(fit_frames),
                                          "early_fit_frames": 0, "label_occluded": False})
    clf = TeamClassifier(cfg)
    for f in fit_frames:                                    # the last call fits the clusters
        _, boxes = pred[f]
        clf(cv2.imread(str(seq.frame_path(f))), boxes, np.ones(len(boxes)))
    out: dict[tuple[int, int], int] = {}
    if not clf.reliable:
        return out                                          # no clear teams: no cuts (as 006)
    for f in frames:
        ids, boxes = pred[f]
        cov = coverage_matrix(boxes).max(axis=1) if len(boxes) > 1 else np.zeros(len(boxes))
        clean = cov < cfg.max_coverage
        if not clean.any():
            continue
        lab = clf.classify(cv2.imread(str(seq.frame_path(f))), boxes[clean], others=boxes[~clean])
        for t, lb in zip(np.asarray(ids)[clean], lab, strict=True):
            out[(f, int(t))] = int(lb)
    return out


def team_labels_osnet(seq, pred: dict, reid, stride: int = 5) -> dict[tuple[int, int], int]:
    """019: ``{(frame, id): team}`` from OSNet, for the clean boxes of one frame in ``stride``.

    Why tracklet means: one box can look like anything (pose, blur), but the mean of a tracklet is
    dominated by its kit. Clustering the tracklets (not the boxes) gives each player one vote, so the
    long tracklets do not decide alone and a few odd boxes cannot create a cluster.
    """
    keys, emb = [], []
    for f in sorted(pred)[::stride]:
        ids, boxes = pred[f]
        clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(len(boxes), bool)
        if not clean.any():
            continue
        e = reid(cv2.imread(str(seq.frame_path(f))), boxes[clean])
        for t, v in zip(np.asarray(ids)[clean], e, strict=True):
            keys.append((f, int(t)))
            emb.append(v / (np.linalg.norm(v) + 1e-12))
    if len(keys) < 3:
        return {}
    emb = np.asarray(emb, np.float32)
    tid = np.array([t for _, t in keys])
    tracks = [t for t in np.unique(tid) if (tid == t).sum() >= 5]       # >= 1 s of clean boxes
    if len(tracks) < 3:
        return {}
    means = np.array([emb[tid == t].mean(0) for t in tracks], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    cv2.setRNGSeed(0)                                        # same clusters on every run
    _, _, centres = cv2.kmeans(means, 3, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    nearest = ((emb[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
    other = int(np.argmin(np.bincount(nearest, minlength=3)))   # fewest boxes: referees / staff
    teams = [k for k in range(3) if k != other]
    to_label = {teams[0]: 0, teams[1]: 1, other: 2}
    return {k: to_label[int(n)] for k, n in zip(keys, nearest, strict=True)}


def piece_kit_prob(seq, pred: dict, seg: dict | None, cache: Path, tag: str, stride: int = 5) -> dict[int, float]:
    """042: {piece: P(kit 1)}. Lab torso median of every clean box (one frame in ``stride``), 2-means over the clip
    (fixed seed), then per piece (n1 + 1) / (n + 2): a piece with few clean boxes stays close to 0.5."""
    if cache.is_file() and str(np.load(cache)["tag"]) == f"lab|{tag}":
        z = np.load(cache)
        keys, X = z["keys"], z["feats"]
    else:
        keys, X = [], []
        for f, ids, boxes in clean_boxes(pred, stride):
            v = lab_torso(cv2.imread(str(seq.frame_path(f))), boxes)
            for t, d in zip(ids, v, strict=True):
                if np.isfinite(d).all():
                    keys.append((int(f), int(t)))
                    X.append(d)
        keys, X = np.array(keys, np.int32).reshape(-1, 2), np.array(X, np.float32).reshape(-1, 3)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, tag=f"lab|{tag}", keys=keys, feats=X)
    if len(X) < 2:
        return {}
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    _, lab, _ = cv2.kmeans(X, 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    n, n1 = Counter(), Counter()
    for (f, t), k in zip(keys.tolist(), lab.ravel().tolist(), strict=True):
        pc = seg.get((f, t), t) if seg is not None else t
        n[pc] += 1
        n1[pc] += k
    return {pc: (n1[pc] + 1) / (n[pc] + 2) for pc in n}


def clean_boxes(pred: dict, stride: int):
    """(frame, ids, boxes) of the clean boxes (no other box covering >= 20 %), one frame in ``stride``."""
    for f in sorted(pred)[::stride]:
        ids, boxes = pred[f]
        clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(len(boxes), bool)
        if clean.any():
            yield f, np.asarray(ids)[clean], np.asarray(boxes)[clean]


def team_labels_features(seq, pred: dict, describe, stride: int = 5, normalise: bool = True,
                         cache: Path | None = None, tag: str = "", **cluster) -> dict[tuple[int, int], int]:
    """020: ``{(frame, id): team}`` with any box descriptor ``describe(frame, boxes) -> (N, D)``.

    Same boxes as 019 (clean, one frame in ``stride``); the vectors become teams with
    ``cluster_team_labels`` (tracklet means, k-means, nearest centre, optional rejection / UMAP).
    ``cache`` (021): the box vectors are saved there, so another run on the same tracker output with the
    same descriptor skips the model (SigLIP: minutes per clip).
    """
    if cache is not None and cache.is_file() and str(np.load(cache)["tag"]) == tag:
        z = np.load(cache)
        keys, feats = [tuple(k) for k in z["keys"].tolist()], z["feats"]
    else:
        keys, chunks = [], []
        for f, ids, boxes in clean_boxes(pred, stride):
            chunks.append(np.asarray(describe(cv2.imread(str(seq.frame_path(f))), boxes), np.float32))
            keys += [(f, int(t)) for t in ids]
        if not keys:
            return {}
        feats = np.concatenate(chunks)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, tag=tag, keys=np.array(keys), feats=feats)
    return cluster_team_labels(keys, feats, normalise=normalise, **cluster)


def piece_teams(labels: dict, seg: dict, min_boxes: int = 3, majority: float = 2 / 3) -> dict[int, int]:
    """021: ``{piece id: team}`` for the pieces whose team is clear.

    A piece's team is the majority label of its labelled boxes, if it has at least ``min_boxes`` of them
    and the majority holds ``majority`` of the votes ("other" never counts as a team). Pieces without a
    clear team stay unknown and the connector treats them as before.
    """
    votes: dict[int, Counter] = {}
    for (f, t), lab in labels.items():
        if lab in (0, 1) and (f, t) in seg:
            votes.setdefault(seg[(f, t)], Counter())[lab] += 1
    out = {}
    for p, v in votes.items():
        lab, n = v.most_common(1)[0]
        if sum(v.values()) >= min_boxes and n >= majority * sum(v.values()):
            out[p] = lab
    return out


def team_labels_gt(seq, pred: dict, stride: int = 5) -> dict[tuple[int, int], int]:
    """ORACLE (ceiling only): each clean box takes the reference team of the GT player it overlaps."""
    gt, team = seq.load_gt(), player_teams(seq)
    out = {}
    for f, ids, boxes in clean_boxes(pred, stride):
        if f not in gt:
            continue
        gids, gb = gt[f]
        m = iou_matrix(np.asarray(boxes, float), np.asarray(gb, float))
        used = set()
        for k in np.argsort(-m, axis=None):           # greedy one-to-one matching, best overlaps first
            i, j = np.unravel_index(k, m.shape)
            if m[i, j] < 0.5:
                break
            if (f, int(ids[i])) in out or j in used:
                continue
            used.add(j)
            if int(gids[j]) in team:
                out[(f, int(ids[i]))] = team[int(gids[j])]
    return out


def piece_numbers(seq, work: dict, reader, n: int, cache: Path) -> dict[int, str]:
    """023: ``{piece id: jersey number}`` for the pieces whose number is clear (``tracklet_number``).

    Each piece is read on its ``n`` tallest clean boxes (the number is largest there). Reads are cached per box
    (frame + rounded box) in ``cache``, so variants on the same tracker output do not run the model again.
    """
    from hooptrack.reid.jersey import tracklet_number

    picks: dict[int, list] = {}
    for f, (ids, boxes) in work.items():
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(len(boxes), bool)
        for t, b, c in zip(ids, boxes, clean, strict=True):
            if c and b[3] - b[1] >= 50:
                picks.setdefault(int(t), []).append((f, b))
    known = {}
    if cache.is_file():
        for row in cache.read_text().splitlines():
            f, x1, y1, x2, y2, text, conf = row.split(",")
            known[(int(f), int(x1), int(y1), int(x2), int(y2))] = (text, float(conf))
    key = lambda f, b: (int(f), *(int(round(v)) for v in b))  # noqa: E731
    todo: dict[int, list] = {}
    chosen = {}
    for t, lst in picks.items():
        chosen[t] = sorted(lst, key=lambda x: -(x[1][3] - x[1][1]))[:n]
        for f, b in chosen[t]:
            if key(f, b) not in known:
                todo.setdefault(f, []).append(b)
    if todo and reader is not None:
        for f in sorted(todo):
            frame = cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2RGB)
            for b, r in zip(todo[f], reader(frame, np.array(todo[f])), strict=True):
                known[key(f, b)] = (r[0].replace(",", " "), r[1])
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("\n".join(f"{','.join(map(str, k))},{v[0]},{v[1]:.5f}" for k, v in known.items()) + "\n")
    out = {}
    for t, lst in chosen.items():
        num = tracklet_number([known[key(f, b)] for f, b in lst if key(f, b) in known])
        if num is not None:
            out[t] = num
    return out


def piece_numbers_classifier(seq, work: dict, reader, n: int, cache: Path) -> tuple[dict[int, str], dict]:
    """033: ``{piece id: jersey number}`` with the digit-by-digit reader (``hooptrack.reid.jersey_classifier``).

    Same crops as ``piece_numbers`` (the ``n`` tallest clean boxes of each piece, >= 50 px). The reader gives 111
    probabilities per crop; a piece's answer is the most probable number of their MEAN (``read_piece``), kept only
    if it is a number (not "no number") with mean probability >= ``reader.min_conf`` (set on the training data).
    Probabilities are cached per box in ``cache`` (.npz). Returns the numbers and, for every read piece, its
    (answer, confidence), for the analysis.
    """
    from hooptrack.reid.jersey_classifier import read_piece

    picks: dict[int, list] = {}
    for f, (ids, boxes) in work.items():
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        clean = coverage_matrix(boxes).max(axis=1) < 0.2 if len(boxes) > 1 else np.ones(len(boxes), bool)
        for t, b, c in zip(ids, boxes, clean, strict=True):
            if c and b[3] - b[1] >= 50:
                picks.setdefault(int(t), []).append((f, b))
    key = lambda f, b: f"{int(f)}_" + "_".join(str(int(round(v))) for v in b)  # noqa: E731
    known = dict(np.load(cache)) if cache.is_file() else {}
    chosen = {t: sorted(lst, key=lambda x: -(x[1][3] - x[1][1]))[:n] for t, lst in picks.items()}
    todo: dict[int, list] = {}
    for lst in chosen.values():
        for f, b in lst:
            if key(f, b) not in known:
                todo.setdefault(f, []).append(b)
    if todo and reader is not None:
        for f in sorted(todo):
            frame = cv2.cvtColor(cv2.imread(str(seq.frame_path(f))), cv2.COLOR_BGR2RGB)
            for b, pr in zip(todo[f], reader.probs(frame, np.array(todo[f])), strict=True):
                known[key(f, b)] = pr
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **known)
    out, reads = {}, {}
    for t, lst in chosen.items():
        probs = [known[key(f, b)] for f, b in lst if key(f, b) in known]
        if not probs:
            continue
        num, conf = read_piece(np.stack(probs))
        reads[t] = (num, round(conf, 4), len(probs))
        if num != "nan" and conf >= reader.min_conf:
            out[t] = num
    return out, reads


def join_pieces(seg: dict, feats: dict, pairs: list[tuple[int, int]]) -> tuple[dict, dict, int]:
    """046: join pieces linked by ``pairs`` (union-find; the smallest id names the group). Returns the remapped
    ``seg``, the embeddings of each group (its pieces' crops together) and the number of links used."""
    parent: dict[int, int] = {}

    def root(x: int) -> int:
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    used = 0
    present = set(seg.values())
    for a, b in pairs:
        if a in present and b in present and root(a) != root(b):
            ra, rb = sorted((root(a), root(b)))
            parent[rb] = ra
            used += 1
    new_seg = {k: root(v) for k, v in seg.items()}
    groups: dict[int, list] = {}
    for p, f in feats.items():
        if len(f):
            groups.setdefault(root(int(p)), []).append(np.asarray(f, np.float32))
    new_feats = {g: np.concatenate(v) for g, v in groups.items()}
    return new_seg, new_feats, used


def rewrite(src: Path, dst: Path, mapping: dict[int, int], seg: dict[tuple[int, int], int] | None = None,
            drop: set[int] | None = None) -> None:
    """Same MOT file with the ids remapped (all other columns untouched).

    ``seg`` (015): ``{(frame, old id): segment id}`` when the tracklets were cut; ``mapping`` then maps
    segment ids to final ids.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        cols = line.split(",")
        f, tid = int(float(cols[0])), int(float(cols[1]))
        if seg is not None:
            tid = seg.get((f, tid), tid)
        new = mapping.get(tid, tid)
        if drop and new in drop:                         # 036b: not one of the 10 players
            continue
        cols[1] = str(new)
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
    ap.add_argument("--mode", choices=["gta", "conservative"], default="gta",
                    help="gta: GTA's connector with its published SportsMOT settings (default); "
                         "conservative: our first rules (doubt margin, thresholds from the train calibration)")
    ap.add_argument("--spatial-factor", type=float, default=1.0, help="gta mode")
    ap.add_argument("--thresh", type=float, default=None,
                    help="gta: merge distance (default 0.4, published); conservative: default from calibration")
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--max-gap", type=int, default=None)
    ap.add_argument("--roster", type=int, default=0,
                    help="036: players per team (5); after stitching, every group is given to one of the 2 x N "
                         "longest groups (needs --split-teams for the team labels); 0 = off")
    ap.add_argument("--closed-set", type=int, default=0,
                    help="037: players on the court (10); instead of the connector, the pieces cut at crossings "
                         "(--split-crossings) are given out to exactly N players; 0 = off")
    ap.add_argument("--continuity", type=float, default=0.0,
                    help="038, with --closed-set: distance bonus for the player who holds the piece's neighbour in "
                         "the online run (same online tracklet, cut at a crossing); 0 = 037b")
    ap.add_argument("--adapt-dim", type=int, default=0,
                    help="040, with --closed-set: adapt the embeddings to the clip (Fisher directions from pieces "
                         "and pieces on screen together, hooptrack.reid.adapt), this many dimensions; 0 = off")
    ap.add_argument("--closed-teams", action="store_true",
                    help="040b, with --closed-set: 5 players per team, a piece never goes to the other team (kit "
                         "from the zero-shot SigLIP torso vectors cached in <out>/<clip>/team_feats_siglip.npz)")
    ap.add_argument("--closed-team-penalty", type=float, default=0.0,
                    help="042, with --closed-set: soft kit term. A piece's distance to a player of the other kit is "
                         "raised by this x P(other kit); P from the Lab torso colour of the piece's clean boxes "
                         "(one frame in 5, 2-means over the clip, cached in <out>/<clip>/team_feats_lab.npz); 0 = off")
    ap.add_argument("--closed-refine", action="store_true",
                    help="043, with --closed-set: re-decide every piece at once (integer program, players described "
                         "by their current pieces, 3 rounds) starting from the closed-set answer")
    ap.add_argument("--closed-team-own", action="store_true",
                    help="042b, with --closed-team-penalty: each seed keeps its own kit instead of a 5/5 split")
    ap.add_argument("--smooth-swaps", action="store_true",
                    help="047, with --closed-set and --continuity: excursions P -> Q -> P inside an online tracklet "
                         "go back to P (swapping back the mirror excursion when there is one)")
    ap.add_argument("--smooth", action="store_true",
                    help="039, with --closed-set and --continuity: a piece between two online neighbours of one "
                         "player goes back to that player")
    ap.add_argument("--roster-drop", action="store_true",
                    help="036b: remove the groups that fit no roster player (at most 2 x N ids per clip)")
    ap.add_argument("--max-speed", type=float, default=None,
                    help="035, gta mode: refuse joins needing more than this speed on the court (m/s); off by default")
    ap.add_argument("--speed-slack", type=float, default=1.0, help="035: metres of tolerance for the box error")
    ap.add_argument("--court-dir", default="outputs/court_complete/c09_pipeline",
                    help="035: <dir>/<clip>/homographies.npz (C09 + C07, completed)")
    ap.add_argument("--split-teams", action="store_true",
                    help="019: cut tracklets where the team changes for good, before stitching")
    ap.add_argument("--team-source", choices=["osnet", "colour", "lab", "osnet_torso", "siglip", "gt"],
                    default="osnet", help="019/020: where the team labels come from (default osnet; gt = oracle)")
    ap.add_argument("--team-k", type=int, default=3, help="020: clusters (3 = two teams + other, as 019)")
    ap.add_argument("--team-reject", type=float, default=0.0,
                    help="020: label a box only if d(nearest) < ratio * d(2nd nearest); 0 = every box (019)")
    ap.add_argument("--team-umap", type=int, default=0, help="020: UMAP dimensions before k-means; 0 = none")
    ap.add_argument("--siglip", default="weights/siglip-base-patch16-224", help="020: local SigLIP folder")
    ap.add_argument("--jersey-rule", action="store_true",
                    help="023: the connector never merges two pieces with different jersey numbers (Florence-2)")
    ap.add_argument("--jersey-substring", action="store_true",
                    help="023b: a number contained in the other (3 / 23) is a partial read, not a conflict")
    ap.add_argument("--jersey-crops", type=int, default=6, help="023: crops read per piece (tallest clean boxes)")
    ap.add_argument("--jersey-reader", choices=["florence", "classifier"], default="florence",
                    help="023: Florence-2 zero-shot OCR | 033: digit-by-digit reader trained on E-BARD")
    ap.add_argument("--jersey-model", default="weights/jersey_reader_siglip", help="033: the trained reader's folder")
    ap.add_argument("--florence", default="weights/florence-2-base", help="023: local Florence-2 folder")
    ap.add_argument("--team-merge-rule", action="store_true",
                    help="021: the connector never merges two pieces of different teams (needs --split-teams)")
    ap.add_argument("--team-min-seconds", type=float, default=1.0,
                    help="019: a cut needs this long of each team on both sides (default 1 s)")
    ap.add_argument("--link-file", type=Path, default=None,
                    help="046, with --split-crossings: CSV clip,a,b of pieces that are the same person (e.g. a "
                         "crossing verified by propagated masks); they are joined into one piece before anything else")
    ap.add_argument("--split-crossings", action="store_true",
                    help="015: cut tracklets at the end of every crossing before stitching")
    ap.add_argument("--crossing-coverage", type=float, default=0.2,
                    help="a box is in a crossing when it covers / is covered by another this much (default 0.2)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    rcfg = cfg["reid"]
    calib = args.calibration or Path("outputs/calibration") / f"{rcfg['type']}.json"
    thresh, margin = args.thresh, args.margin
    if args.mode == "gta":
        scfg = GTAConfig(thresh=0.4 if thresh is None else thresh, spatial_factor=args.spatial_factor,
                         number_substring=args.jersey_substring, max_speed=args.max_speed,
                         speed_slack=args.speed_slack)
    elif thresh is None or margin is None:
        if not calib.is_file():
            raise SystemExit(f"{calib} not found: run scripts/calibrate_reid.py or pass --thresh and --margin")
        hn = json.loads(calib.read_text())["hard_neg_pct"]
        thresh = hn["p5"] if thresh is None else thresh
        margin = round(hn["p5"] - hn["p1"], 3) if margin is None else margin
    if args.mode == "conservative":
        scfg = StitchConfig(thresh=thresh, margin=margin, max_gap=args.max_gap)
    tag = f"{rcfg['type']}|{rcfg.get('weights')}|{args.crops}|{args.min_height}"
    if args.split_crossings:
        tag += f"|split{args.crossing_coverage}"
    if args.split_teams:
        tag += f"|teams-{args.team_source}{args.team_min_seconds}"
        if (args.team_k, args.team_reject, args.team_umap) != (3, 0.0, 0):
            tag += f"|k{args.team_k}r{args.team_reject}u{args.team_umap}"
    if args.team_merge_rule and not (args.split_teams and args.mode == "gta"):
        raise SystemExit("--team-merge-rule needs --split-teams (the team labels) and the gta mode")
    if args.jersey_rule and args.mode != "gta":
        raise SystemExit("--jersey-rule needs the gta mode")
    if args.smooth and not (args.closed_set and args.continuity):
        raise SystemExit("--smooth needs --closed-set and --continuity (the online links)")
    if args.closed_set and not args.split_crossings:
        raise SystemExit("--closed-set needs --split-crossings (pieces that hold one player each)")
    if args.roster and not args.split_teams:
        raise SystemExit("--roster needs --split-teams (the team labels)")
    if args.split_teams and args.split_crossings:
        raise SystemExit("--split-teams and --split-crossings are alternatives: pick one")
    print(f"ReID {rcfg['type']} {rcfg.get('weights') or ''} | mode {args.mode} | {scfg}"
          + (f" | split at crossings (coverage {args.crossing_coverage})" if args.split_crossings else "")
          + (f" | split at team changes ({args.team_source}, {args.team_min_seconds} s)" if args.split_teams else ""))

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
    link_pairs: dict[str, list[tuple[int, int]]] = {}
    if args.link_file is not None:                   # 046
        for r in pd.read_csv(args.link_file).itertuples():
            link_pairs.setdefault(r.clip, []).append((int(r.a), int(r.b)))
    class _LazyReid:                               # built on first use: cached embeddings need no model
        def __init__(self):
            self.m = None

        def __call__(self, *a):
            if self.m is None:
                self.m = build_reid(rcfg)
            return self.m(*a)

    reid = _LazyReid()
    describe, normalise = None, True                 # 020 team descriptors
    if args.split_teams and args.team_source in ("lab", "osnet_torso", "siglip", "osnet"):
        if args.team_source == "lab":
            describe, normalise = lab_torso, False
        elif args.team_source == "osnet_torso":
            describe = lambda img, b: reid(img, torso_boxes(b))  # noqa: E731
        elif args.team_source == "siglip":
            from hooptrack.reid.siglip import SigLIPExtractor

            sig = SigLIPExtractor(args.siglip)
            describe = lambda img, b: sig(img, torso_boxes(b))  # noqa: E731
        elif (args.team_k, args.team_reject, args.team_umap) != (3, 0.0, 0):
            describe = reid                          # osnet with 020 options; plain osnet = 019 below
    reader = None
    if args.jersey_rule and args.jersey_reader == "classifier":     # 033
        from hooptrack.reid.jersey_classifier import JerseyClassifierReader

        reader = JerseyClassifierReader(args.jersey_model)
    elif args.jersey_rule:
        from hooptrack.reid.jersey import JerseyReader

        reader = JerseyReader(args.florence)
    cluster = {"k": args.team_k, "reject_ratio": args.team_reject or None, "umap_dims": args.team_umap or None}
    before, after, rows, all_logs, all_numbers, all_roster = {}, {}, [], {}, {}, {}
    for clip in clips:
        src = source(clip)
        pred = load_mot_txt(src)
        n_orig = len({int(t) for ids, _ in pred.values() for t in ids})
        seg = None
        work = pred
        if args.split_crossings:     # 015: tracklets -> segments; everything below works on segments
            seg = split_at_crossings(pred, args.crossing_coverage)
            work = {f: (np.array([seg[(f, int(t))] for t in ids]), b) for f, (ids, b) in pred.items()}
        if args.split_teams:         # 019: tracklets -> pieces of one team each
            # labels every frame (colour) or every 5th frame (osnet): min_run in labelled boxes = seconds
            if describe is not None:
                tcache = args.out / clip / f"team_feats_{args.team_source}.npz"
                labels = team_labels_features(seqs[clip], pred, describe, 5, normalise, tcache,
                                              f"{args.team_source}|{src}", **cluster)
                stride = 5
            elif args.team_source == "gt":
                labels, stride = team_labels_gt(seqs[clip], pred, stride=5), 5
            elif args.team_source == "osnet":
                labels, stride = team_labels_osnet(seqs[clip], pred, reid, stride=5), 5
            else:
                labels, stride = team_labels(seqs[clip], pred, cfg.get("teams")), 1
            min_run = max(1, round(args.team_min_seconds * seqs[clip].fps / stride))
            seg = split_at_team_changes(pred, labels, min_run)
            work = {f: (np.array([seg[(f, int(t))] for t in ids]), b) for f, (ids, b) in pred.items()}
            n_cuts = len(set(seg.values())) - n_orig
            print(f"{clip}: {len(labels)} boxes with a team label, {n_cuts} team cuts")
        feats = embed_tracklets(seqs[clip], work, reid, args.crops, args.min_height,
                                args.out / clip / "embeddings.npz", tag)
        if args.link_file is not None and seg is not None:   # 046: pieces known to be one person, joined first
            pairs = link_pairs.get(clip, [])
            seg, feats, n_joined = join_pieces(seg, feats, pairs)
            work = {f: (np.array([seg[(f, int(t))] for t in ids]), b) for f, (ids, b) in pred.items()}
            print(f"{clip}: {n_joined} links from {args.link_file.name} -> {len(set(seg.values()))} pieces")
        frames: dict[int, list[tuple[int, np.ndarray]]] = {}
        for f, (ids, boxes) in work.items():
            for t, b in zip(ids, boxes, strict=True):
                frames.setdefault(int(t), []).append((f, b))
        tracklets = []
        for t, fb in frames.items():
            fb.sort(key=lambda x: x[0])
            tracklets.append(Tracklet(t, np.array([f for f, _ in fb]), feats.get(t, np.zeros((0, 1), np.float32)),
                                      np.array([b for _, b in fb], float)))
        if args.max_speed is not None:   # 035: feet on the court for every box (NaN where the clip has no court)
            from dataclasses import replace

            from hooptrack.integrations.topdown import TopDownConfig, feet_on_court, load_homographies

            court = load_homographies(TopDownConfig(dir=args.court_dir), clip)
            for t in tracklets:
                t.ground = (feet_on_court(t.frames, t.boxes, court[1], court[0]) if court is not None
                            else np.full((len(t.frames), 2), np.nan))
            scfg = replace(scfg, fps=seqs[clip].fps)
            print(f"{clip}: court {'found' if court is not None else 'MISSING (speed rule inactive)'}")
        if args.closed_set:              # 037: no connector, the pieces go to exactly N players
            links: dict[int, set[int]] = {}
            if args.continuity:          # 038: pieces next to each other inside one online tracklet
                chain: dict[int, list[tuple[int, int]]] = {}
                for (f, t), p in seg.items():
                    chain.setdefault(t, []).append((f, p))
                for lst in chain.values():
                    order = [p for _, p in sorted(lst)]
                    for a_, b_ in zip(order, order[1:], strict=False):
                        if a_ != b_:
                            links.setdefault(a_, set()).add(b_)
                            links.setdefault(b_, set()).add(a_)
            scale, tfeats = 1.0, None
            if args.adapt_dim:           # 040: embeddings adapted to this clip, no labels
                from hooptrack.reid.adapt import adapt

                adapted, scale = adapt({t.tid: t.feats for t in tracklets}, {t.tid: t.frames for t in tracklets},
                                       args.adapt_dim)
                for t in tracklets:
                    t.feats = adapted[t.tid]
            if args.closed_teams:        # 040b: kit descriptor per piece, from the cached torso vectors
                tc = args.out / clip / "team_feats_siglip.npz"
                if not tc.is_file():
                    raise SystemExit(f"{tc} not found: copy it from the 029 run (same online tracklets)")
                z = np.load(tc)
                acc: dict[int, list[np.ndarray]] = {}
                for (f, t), v in zip(z["keys"].tolist(), z["feats"], strict=True):
                    pc = seg.get((int(f), int(t)))
                    if pc is not None:
                        acc.setdefault(pc, []).append(v / (np.linalg.norm(v) + 1e-12))
                tfeats = {pc: np.mean(v, axis=0) for pc, v in acc.items()}
            tprob = None
            if args.closed_team_penalty:  # 042: P(kit 1) per piece from the Lab torso colour of its clean boxes
                tprob = piece_kit_prob(seqs[clip], pred, seg, args.out / clip / "team_feats_lab.npz", str(src))
            mapping, leftovers, cs_log = assign_closed_set(tracklets, args.closed_set, links=links,
                                                           continuity=args.continuity, dist_scale=scale,
                                                           team_feats=tfeats, team_prob=tprob,
                                                           team_penalty=args.closed_team_penalty,
                                                           team_split=not args.closed_team_own)
            refine_log = []
            if args.closed_refine:       # 043: global re-assignment instead of never revisiting a decision
                mapping, refine_log = refine_closed_set(
                    tracklets, mapping, links, continuity=args.continuity, dist_scale=scale,
                    players=[a["piece"] for a in cs_log if a.get("seed")])
                leftovers = {t.tid for t in tracklets if mapping[t.tid] == t.tid
                             and sum(v == t.tid for v in mapping.values()) == 1}
                print(f"{clip}: refinement {refine_log}")
            drop = leftovers if args.roster_drop else set()     # 040c: at most N ids per clip
            log = []
            smooth_log = []
            if args.smooth_swaps:        # 047: excursions and swapped excursions inside online tracklets
                chains_l = [[p for _, p in sorted(set(lst))] for lst in chain.values()]
                chains_l = [[p for k, p in enumerate(c) if k == 0 or p != c[k - 1]] for c in chains_l]
                mapping, swap_log = smooth_swapped_excursions(tracklets, mapping, chains_l)
                smooth_log = smooth_log + swap_log
                print(f"{clip}: {len(swap_log)} excursions sent back")
            if args.smooth:              # 039: short excursions between two pieces of one player
                mapping, smooth_log = smooth_sandwiches(tracklets, mapping, links)
                print(f"{clip}: smoothing moved {len(smooth_log)} pieces")
            all_roster[clip] = {"assigned": cs_log, "leftovers": sorted(int(p) for p in leftovers),
                                "smoothed": smooth_log, "refined": refine_log}
            print(f"{clip}: closed set: {len(tracklets)} pieces -> {len(set(mapping.values()))} ids "
                  f"({len(leftovers)} pieces fit no player)")
        elif args.team_merge_rule or args.jersey_rule:
            # 021: pieces of two different kits are never one player; 023: nor two different numbers
            teams = piece_teams(labels, seg) if args.team_merge_rule else None
            numbers = None
            if args.jersey_rule and args.jersey_reader == "classifier":
                numbers, reads = piece_numbers_classifier(seqs[clip], work, reader, args.jersey_crops,
                                                          args.out / clip / "jersey_probs.npz")
                all_numbers[clip] = {"numbers": numbers, "reads": reads}
                # 033: the pieces themselves, for the analysis (which ground-truth player each piece is)
                piece_rows = [(f, int(t), b, 1.0) for f, (ids, bs) in work.items()
                              for t, b in zip(ids, bs, strict=True)]
                write_mot_txt(args.out / clip / "pieces.txt", piece_rows)
            elif args.jersey_rule:
                numbers = piece_numbers(seqs[clip], work, reader, args.jersey_crops,
                                        args.out / clip / "jersey_reads.csv")
            if args.jersey_rule:
                print(f"{clip}: {len(numbers)} of {len(tracklets)} pieces with a jersey number")
            mapping, log = stitch_gta(tracklets, scfg, teams, numbers)
        else:
            mapping, log = (stitch_gta if args.mode == "gta" else stitch)(tracklets, scfg)
        if not args.closed_set:
            drop = set()
        roster_log = []
        if args.roster:                  # 036: every group to one of the 10 players
            mapping, leftovers, roster_log = assign_roster(tracklets, mapping, piece_teams(labels, seg), args.roster)
            drop = leftovers if args.roster_drop else set()
            all_roster[clip] = {"assigned": roster_log, "leftovers": sorted(int(g) for g in leftovers)}
            print(f"{clip}: roster: {len(roster_log)} groups given to a player, {len(leftovers)} fit nobody"
                  + (" (dropped)" if args.roster_drop else ""))
        dst = args.out / clip / "results" / f"{clip}.txt"
        rewrite(src, dst, mapping, seg, drop)
        gt = seqs[clip].load_gt()
        before[clip], after[clip] = (gt, pred), (gt, load_mot_txt(dst))
        merged = [e for e in log if e["merged"]]
        rows.append({"clip": clip, "ids_before": n_orig, "segments": len(frames),
                     "ids_after": len(set(mapping.values()) - drop),
                     "merges": len(merged), "refused": len(log) - len(merged),
                     "refused_speed": sum(e.get("why") == "speed" for e in log)})
        all_logs[clip] = log
        print(f"{clip}: {n_orig} ids, {len(frames)} segments -> {len(set(mapping.values()))} "
              f"({len(merged)} merges, "
              f"{len(log) - len(merged)} refused)")

    mb, ma = metrics(before), metrics(after)
    table = mb.add_suffix(" before").join(ma.add_suffix(" after"))
    table = table[[c for m in ("IDF1", "MOTA", "IDSW", "Frag") for c in (f"{m} before", f"{m} after")]]
    ids = pd.DataFrame(rows).set_index("clip")
    print("\n" + table.to_string() + "\n\n" + ids.to_string())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({
        "run": str(args.run), "reid": rcfg["type"], "reid_weights": rcfg.get("weights"),
        "config": vars(scfg), "metrics": table.reset_index().to_dict(orient="records"),
        "ids": rows, "merges": all_logs, "jersey": all_numbers, "roster": all_roster}, indent=2, default=str))
    print(f"\nSaved {args.out / 'summary.json'} and the stitched results in {args.out}/<clip>/results/")


if __name__ == "__main__":
    main()
