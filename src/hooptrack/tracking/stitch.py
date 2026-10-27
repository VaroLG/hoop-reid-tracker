"""Offline tracklet stitching by appearance (experiment 013).

Why
---
The online tracker decides frame by frame, with what it has seen so far. When a
player is lost for a while (occlusion, leaves the frame, a missed detection) and
comes back, it often gets a NEW id: 17–21 identities for 10 players on the
validation clips (012). Its memory compares ONE fresh crop, often blurred or
half hidden, with the lost players, so it must be very cautious and almost
never revives anyone.

After the whole video has been processed we can do better, because we can see
every *tracklet* (all the boxes that share one id) at once:

* each tracklet is described by MANY crops (the cleanest ones, spread over its
  life) instead of one, so one bad crop cannot decide;
* we know where every tracklet starts and ends, so we know which ones *could*
  be the same player: two tracklets visible in the same frame are two people.

This is the idea of GTA (Global Tracklet Association) on top of Deep-EIoU, the
reference tracker on SportsMOT. It only rewrites ids in the output file: the
online tracker is not touched (same spirit as the offline mode of 007).

The rules
---------
Tracklets ``a`` (earlier) and ``b`` (later) are merged when ALL of these hold:

1. **Never together**: no frame where both groups (``a`` with everything already
   merged into it, same for ``b``) have a box. Two boxes at the same time are
   two players, whatever their appearance says.
2. **They look alike**: the appearance distance between them is below
   ``thresh``. The distance between two tracklets is the MEDIAN of the cosine
   distances between all their crop pairs, so a few bad crops do not matter.
3. **No doubt**: the pair must beat its *competitors* by ``margin``. A competitor
   of ``b`` (as successor of ``a``) is another later tracklet that is on screen at
   the same time as ``b``'s group: both cannot be ``a``, so if they look almost
   equally like ``a`` (two team-mates), nothing is merged. A missing merge costs a
   fragment; a wrong merge gives a player someone else's identity, which is worse.
   Tracklets that are NOT on screen together are not competitors: they can all be
   the same player in a row (a player lost three times = three fragments).
4. Optionally, ``max_gap`` frames at most between the end of ``a`` and the
   start of ``b``.

Pairs are taken from the most similar to the least similar (greedy), so the
clearest merges happen first and constrain the doubtful ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tracklet:
    """All the boxes that share one id in the tracker output."""

    tid: int
    frames: np.ndarray                     # sorted frame numbers where the id has a box
    feats: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), np.float32))  # (K, D) crop embeddings
    boxes: np.ndarray | None = None        # (len(frames), 4) TLBR, aligned with frames (GTA's spatial rule)

    @property
    def start(self) -> int:
        return int(self.frames[0])

    @property
    def end(self) -> int:
        return int(self.frames[-1])


@dataclass
class StitchConfig:
    thresh: float = 0.077          # max tracklet distance to merge (OSNet: hard-negative p5 on train)
    margin: float = 0.021          # lead over the second-best option (OSNet: hard-negative p5 - p1)
    max_gap: int | None = None     # frames between the end of a and the start of b (None = any)
    min_crops: int = 3             # tracklets with fewer usable crops are never merged


def tracklet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Median cosine distance between all crop pairs of two tracklets (embeddings L2-normalised)."""
    return float(np.median(1.0 - a @ b.T))


def stitch(tracklets: list[Tracklet], cfg: StitchConfig) -> tuple[dict[int, int], list[dict]]:
    """Return ``{old id: new id}`` and a log of the merges (and of the pairs refused for doubt).

    The new id of a group is the id of its earliest tracklet.
    """
    usable = [t for t in tracklets if len(t.feats) >= cfg.min_crops]
    n = len(usable)
    idx = {t.tid: i for i, t in enumerate(usable)}

    # 1. distance between every pair that COULD be the same player (a ends before b starts)
    dist = np.full((n, n), np.inf)                 # dist[i, j]: i earlier, j later
    for i, a in enumerate(usable):
        for j, b in enumerate(usable):
            if a.end < b.start and (cfg.max_gap is None or b.start - a.end <= cfg.max_gap):
                dist[i, j] = tracklet_distance(a.feats, b.feats)

    # 2. union-find over groups; each group remembers its frames to enforce rule 1
    parent = list(range(n))
    frames = [set(t.frames.tolist()) for t in usable]

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def best_competitor(dists: np.ndarray, chosen: int) -> float:
        # closest alternative to `chosen` that was on screen together with chosen's group
        g = frames[find(chosen)]
        best = np.inf
        for k in np.flatnonzero(np.isfinite(dists)):
            if k != chosen and find(k) != find(chosen) and frames[find(k)] & g:
                best = min(best, float(dists[k]))
        return best

    log = []
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    for i, j in order:
        d = dist[i, j]
        if not np.isfinite(d) or d > cfg.thresh:
            break                                  # sorted: nothing closer is left
        gi, gj = find(i), find(j)
        if gi == gj:
            continue
        if frames[gi] & frames[gj]:                # rule 1: the two groups were on screen together
            continue
        # rule 3: the best successor of i must be j, and the best predecessor of j must be i, with a margin
        alt = min(best_competitor(dist[i], j), best_competitor(dist[:, j], i))
        entry = {"a": usable[i].tid, "b": usable[j].tid, "dist": round(d, 4),
                 "best_competitor": round(alt, 4) if np.isfinite(alt) else None,
                 "gap_frames": usable[j].start - usable[i].end}
        if alt - d < cfg.margin:
            log.append({**entry, "merged": False, "why": "ambiguous"})
            continue
        root, child = (gi, gj) if usable[gi].start <= usable[gj].start else (gj, gi)
        parent[child] = root
        frames[root] |= frames[child]
        log.append({**entry, "merged": True})

    mapping = {t.tid: t.tid for t in tracklets}
    for t in usable:
        mapping[t.tid] = usable[find(idx[t.tid])].tid
    # a group's id is its earliest tracklet's id
    groups: dict[int, list[int]] = {}
    for t in usable:
        groups.setdefault(mapping[t.tid], []).append(t.tid)
    first = {root: min(members, key=lambda m: usable[idx[m]].start) for root, members in groups.items()}
    for t in usable:
        mapping[t.tid] = first[mapping[t.tid]]
    return mapping, log


# ============================================================================ GTA connector
# A second way to merge, reproducing the *connector* of GTA (Sun et al., "GTA: Global Tracklet
# Association for Multi-Object Tracking in Sports", ACCV-W 2024; code: github.com/sjc042/gta-link, MIT).
# Experiment 013 ran the original code on our results: with its published SportsMOT settings it beats the
# conservative rules above (IDF1 80.5 vs 77.2), so the same rules are rebuilt here, explained:
#
#   distance  MEAN cosine distance over all crop pairs of the two groups; 1 (= never) if the two groups
#             have a box in the same frame (two people).
#   order     hierarchical: always merge the globally closest pair first, then recompute the distances
#             of the merged group (its crops are now the union of both), and repeat while some pair is
#             closer than `thresh`.
#   space     a merge is refused if, walking through the merged group in time, the jump between the
#             last box of one fragment and the first box of the next is larger than the range the
#             players' centres cover in the clip (x and y separately, times `spatial_factor`). A refused
#             pair is set to `thresh` so it is never tried again.
#
# There is no "doubt" rule: with a loose threshold the merges are limited by the time rule (each group
# ends up with at most one box per frame) and, in our clips, the result is close to one id per player.
# Differences with the original code, on purpose: groups are kept sorted in time, and short fragments are
# not dropped from the spatial check (the original skips leftover segments under 30 frames).


@dataclass
class GTAConfig:
    thresh: float = 0.4            # GTA's published SportsMOT value (merge_dist_thres)
    spatial_factor: float = 1.0    # GTA's published SportsMOT value
    min_crops: int = 1


def _centres(boxes: np.ndarray) -> np.ndarray:
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]


def _mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(1.0 - a @ b.T))


def _spatially_ok(frames: np.ndarray, boxes: np.ndarray, owner: np.ndarray, max_dx: float, max_dy: float) -> bool:
    """Walk the merged group in time; every change of fragment must be a plausible jump."""
    order = np.argsort(frames, kind="stable")
    c, o = _centres(boxes[order]), owner[order]
    change = np.flatnonzero(o[1:] != o[:-1])            # index i: box i ends a fragment, box i+1 starts another
    if not len(change):
        return True
    jump = np.abs(c[change + 1] - c[change])
    return bool(np.all(jump[:, 0] <= max_dx) and np.all(jump[:, 1] <= max_dy))


def stitch_gta(tracklets: list[Tracklet], cfg: GTAConfig) -> tuple[dict[int, int], list[dict]]:
    """GTA connector. Returns ``{old id: new id}`` (new id = earliest id of the group) and a merge log."""
    usable = [t for t in tracklets if len(t.feats) >= cfg.min_crops and t.boxes is not None]
    mapping = {t.tid: t.tid for t in tracklets}
    if len(usable) < 2:
        return mapping, []
    allc = _centres(np.concatenate([t.boxes for t in usable]))
    max_dx = (allc[:, 0].max() - allc[:, 0].min()) * cfg.spatial_factor
    max_dy = (allc[:, 1].max() - allc[:, 1].min()) * cfg.spatial_factor

    # one group per tracklet: members, frames, boxes, owner of each box, crop embeddings
    groups = [{"members": [t.tid], "frames": t.frames.copy(), "boxes": t.boxes.copy(),
               "owner": np.full(len(t.frames), t.tid), "feats": t.feats} for t in usable]

    def dist(a: dict, b: dict) -> float:
        if np.intersect1d(a["frames"], b["frames"]).size:
            return 1.0                                   # on screen together: two people
        return _mean_distance(a["feats"], b["feats"])

    n = len(groups)
    D = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dist(groups[i], groups[j])
    alive = np.ones(n, bool)
    log = []
    while True:
        Dm = np.where(alive[:, None] & alive[None, :], D, np.inf)
        i, j = np.unravel_index(np.argmin(Dm), Dm.shape)
        d = Dm[i, j]
        if not d < cfg.thresh:
            break
        a, b = groups[i], groups[j]
        frames = np.concatenate([a["frames"], b["frames"]])
        boxes = np.concatenate([a["boxes"], b["boxes"]])
        owner = np.concatenate([a["owner"], b["owner"]])
        entry = {"a": a["members"], "b": b["members"], "dist": round(float(d), 4)}
        if not _spatially_ok(frames, boxes, owner, max_dx, max_dy):
            D[i, j] = D[j, i] = cfg.thresh               # never try this pair again
            log.append({**entry, "merged": False, "why": "spatial"})
            continue
        order = np.argsort(frames, kind="stable")
        groups[i] = {"members": a["members"] + b["members"], "frames": frames[order], "boxes": boxes[order],
                     "owner": owner[order], "feats": np.concatenate([a["feats"], b["feats"]])}
        alive[j] = False
        D[j, :] = D[:, j] = np.inf
        for k in np.flatnonzero(alive):
            if k != i:
                D[i, k] = D[k, i] = dist(groups[i], groups[k])
        log.append({**entry, "merged": True})

    start = {t.tid: t.start for t in usable}
    for g, ok in zip(groups, alive, strict=True):
        if ok:
            new = min(g["members"], key=lambda m: start[m])
            for m in g["members"]:
                mapping[m] = new
    return mapping, log


# ============================================================================ split at crossings (015)
# Why: stitching can only JOIN tracklets. When two players cross and the tracker swaps their ids, each
# tracklet ends up holding two players (before and after the crossing) and no join can fix it.
# Diagnosis on 011b (015 README): every identity change while the player stays visible happens right
# after a crossing, and with a perfect connector, cutting the tracklets at the end of each crossing
# lifts the IDF1 ceiling from 86 to 97.
#
# How: a box is "in a crossing" when it covers, or is covered by, another box by at least
# `max_coverage` (0.2, the same threshold the tracker uses to call a crop clean). Walking each
# tracklet in time, a new segment starts at the first clean box after a crossing: from there on it may
# be another player. The crossing frames stay with the segment before them, so every new segment
# starts with clean crops that the ReID model can describe. A segment that has not had a clean box yet
# is not cut (it would leave a piece with nothing but mixed crops). The connector (`stitch_gta`) then
# joins the segments back, and the time rule ("on screen together = two people") decides most of it.


def split_at_crossings(
    pred: dict[int, tuple[np.ndarray, np.ndarray]], max_coverage: float = 0.2
) -> dict[tuple[int, int], int]:
    """``{(frame, old id): segment id}``. The first segment of a tracklet keeps its id, later ones get new ids."""
    from .matching import coverage_matrix  # local import: the stitching functions above need no geometry

    crossing: dict[tuple[int, int], bool] = {}
    for f, (ids, boxes) in pred.items():
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if len(boxes) > 1:
            cov = coverage_matrix(boxes)
            hit = np.maximum(cov, cov.T).max(axis=1) >= max_coverage   # covers OR is covered
        else:
            hit = np.zeros(len(boxes), bool)
        for t, h in zip(ids, hit, strict=True):
            crossing[(int(f), int(t))] = bool(h)

    by_id: dict[int, list[int]] = {}
    for f, t in crossing:
        by_id.setdefault(t, []).append(f)
    next_id = max(by_id, default=0) + 1
    seg: dict[tuple[int, int], int] = {}
    for t in sorted(by_id):
        cur, had_clean, prev_crossing = t, False, False
        for f in sorted(by_id[t]):
            c = crossing[(f, t)]
            if not c and prev_crossing and had_clean:      # a crossing just ended: maybe someone else now
                cur, next_id, had_clean = next_id, next_id + 1, False
            had_clean |= not c
            prev_crossing = c
            seg[(f, t)] = cur
    return seg


# ============================================================================ split at team changes (019)
# Why: after stitching (013b), 46 identity changes in the 4 evaluation clips are ids STOLEN in a crossing:
# player A's track goes on with player B, and A gets another id. Stitching can never give A his id back,
# because it is still on screen (on B). In 36 of the 46, B plays for the OTHER team. A tracklet that is
# team A for seconds and team B afterwards contains two players, and the jersey colour shows it.
#
# 006 tried the team as an ONLINE rule, frame by frame, and failed: the colour classifier is wrong on
# 3-6 % of boxes, and inside a crossing the boxes are hidden. Offline, the whole tracklet is known. A
# wrong label is a single box; a stolen id is a long run of the other team's labels. So a tracklet is cut
# only where a run of at least `min_run` labelled boxes of one team is followed by a run of at least
# `min_run` labelled boxes of another. The cut goes at the end of the last crossing before the new run
# (same crossings as `split_at_crossings`), or at the first box of the new run if there is no crossing in
# between. A wrong cut is cheap: it leaves two pieces of the same player that never overlap, and
# `stitch_gta` joins them again. Ceiling with ground-truth teams (019 README): IDF1 81.3 -> 89.1.


def split_at_team_changes(
    pred: dict[int, tuple[np.ndarray, np.ndarray]],
    labels: dict[tuple[int, int], int],
    min_run: int = 25,
    max_coverage: float = 0.2,
) -> dict[tuple[int, int], int]:
    """``{(frame, old id): piece id}``, cutting tracklets where their team changes for good.

    ``labels``: ``{(frame, id): label}`` with 0 / 1 (teams), 2 (other) or -1 / missing (unknown).
    The first piece of a tracklet keeps its id; later pieces get new ids.
    """
    crossing = split_at_crossings(pred, max_coverage)       # crossing-end boundaries, to place the cut
    by_id: dict[int, list[int]] = {}
    for f, (ids, _) in pred.items():
        for t in ids:
            by_id.setdefault(int(t), []).append(int(f))
    next_id = max(by_id, default=0) + 1
    out: dict[tuple[int, int], int] = {}
    for t in sorted(by_id):
        frames = sorted(by_id[t])
        # frames where a new crossing segment starts (= right after a crossing ended)
        starts = [f for prev, f in zip(frames, frames[1:], strict=False) if crossing[(f, t)] != crossing[(prev, t)]]
        known = [(f, labels.get((f, t), -1)) for f in frames]
        known = [(f, lab) for f, lab in known if lab >= 0]
        # runs of identical consecutive labels
        runs: list[tuple[int, int, int, int]] = []            # (label, first frame, last frame, length)
        for f, lab in known:
            if runs and runs[-1][0] == lab:
                lab0, f0, _, n = runs[-1]
                runs[-1] = (lab0, f0, f, n + 1)
            else:
                runs.append((lab, f, f, 1))
        cuts, team, team_end = [], None, None
        for lab, f0, f1, n in runs:
            if n < min_run:
                continue                                     # too short: noise, or not yet established
            if team is not None and lab != team:
                # the player changed between team_end (last box of the old team) and f0 (first of the new)
                ends = [f for f in starts if team_end < f <= f0]
                cuts.append(ends[-1] if ends else f0)
            team, team_end = lab, f1
        piece, k = t, 0
        for f in frames:
            if k < len(cuts) and f >= cuts[k]:
                piece, next_id, k = next_id, next_id + 1, k + 1
            out[(f, t)] = piece
    return out
