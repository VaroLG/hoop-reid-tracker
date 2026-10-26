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
