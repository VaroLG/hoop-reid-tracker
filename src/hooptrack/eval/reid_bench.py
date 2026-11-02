"""Team-mate ReID benchmark (experiment 016): can an appearance model tell team-mates apart?

Why a new benchmark
-------------------
012 measured ReID as retrieval among all the players of a clip (rank-1 0.71).
015 showed that this hides the real problem: after a crossing, OSNet cannot
tell whether a player is still the same one or a team-mate (AUC 0.55). The
opponents are easy (different kit). The team-mates are the hard case. Here
the two are measured apart, at the two levels the pipeline uses:

* **crop**: one crop against one crop (what the online tracker does);
* **tracklet**: a 2-second piece of a player (several crops, mean distance
  over all crop pairs) against another piece (what ``stitch_gta`` does).

Metrics
-------
* ``auc_mate``: probability that a pair of the SAME player is closer than a pair
  of TEAM-MATES. 1.0 = always, 0.5 = a coin. This is the number to improve.
* ``auc_opp``: same, against OPPONENTS (expected to be easy).
* ``rank1_mate``: for each query, among the crops (or pieces) of its own team
  only (about 5 players), is the closest one the same player? Chance ~ 1/5.

Every pair is at least ``min_gap`` frames apart: two crops one frame apart are
almost the same image and would make any model look good.

The functions below work on plain arrays (embeddings, player ids, teams,
frames) so they can be tested without images or models.
"""

from __future__ import annotations

import numpy as np


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a positive distance < a negative distance), ties count half (Mann-Whitney U).

    ``pos``: distances that should be SMALL (same player); ``neg``: distances that
    should be LARGE (another player). Computed with ranks: O(n log n), not O(n*m).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    # U statistic of the NEGATIVES: how often a negative is larger than a positive
    r_neg = ranks[len(pos):].sum()
    u = r_neg - len(neg) * (len(neg) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def pair_distances(feats, pid, team, frame, min_gap):
    """All pairs (i < j) at least ``min_gap`` frames apart, split by kind.

    feats are L2-normalised (cosine distance = 1 - dot). Returns a dict with the
    distances of the pairs ``same`` / ``mate`` / ``opp``.
    """
    feats = np.asarray(feats, np.float32)
    d = 1.0 - feats @ feats.T
    i, j = np.triu_indices(len(feats), k=1)
    far = np.abs(frame[i] - frame[j]) >= min_gap
    i, j = i[far], j[far]
    same = pid[i] == pid[j]
    mate = ~same & (team[i] == team[j])
    opp = team[i] != team[j]
    return {"same": d[i[same], j[same]], "mate": d[i[mate], j[mate]], "opp": d[i[opp], j[opp]]}


def rank1_among_teammates(feats, pid, team, frame, min_gap) -> float:
    """Share of queries whose nearest crop of their OWN team (>= min_gap away) is the same player.

    Queries whose player has no crop far enough are skipped (they cannot be right).
    """
    feats = np.asarray(feats, np.float32)
    d = 1.0 - feats @ feats.T
    hits, n = 0, 0
    for q in range(len(feats)):
        ok = (team == team[q]) & (np.abs(frame - frame[q]) >= min_gap)
        ok[q] = False
        if not np.any(ok & (pid == pid[q])) or not np.any(ok & (pid != pid[q])):
            continue
        cand = np.flatnonzero(ok)
        hits += int(pid[cand[np.argmin(d[q, cand])]] == pid[q])
        n += 1
    return hits / n if n else float("nan")


def pieces(pid, frame, length: int, min_crops: int):
    """Cut each player's crops into consecutive pieces of ``length`` frames (pseudo-tracklets).

    Returns a list of index arrays (into the crop arrays). A piece needs ``min_crops`` crops.
    """
    out = []
    for p in np.unique(pid):
        idx = np.flatnonzero(pid == p)
        idx = idx[np.argsort(frame[idx])]
        start = frame[idx[0]]
        cur: list[int] = []
        for k in idx:
            if frame[k] - start >= length:
                if len(cur) >= min_crops:
                    out.append(np.array(cur))
                cur, start = [], frame[k]
            cur.append(k)
        if len(cur) >= min_crops:
            out.append(np.array(cur))
    return out


def piece_level(feats, pid, team, frame, groups, min_gap):
    """Same metrics with pieces: distance = mean cosine distance over all crop pairs (as stitch_gta)."""
    feats = np.asarray(feats, np.float32)
    n = len(groups)
    D = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            D[a, b] = D[b, a] = float(np.mean(1.0 - feats[groups[a]] @ feats[groups[b]].T))
    gp = np.array([pid[g[0]] for g in groups])
    gt = np.array([team[g[0]] for g in groups])
    gs = np.array([frame[g].min() for g in groups])
    ge = np.array([frame[g].max() for g in groups])
    # two pieces are "far" if the gap between them is at least min_gap frames
    a, b = np.triu_indices(n, k=1)
    gap = np.maximum(gs[b] - ge[a], gs[a] - ge[b])
    keep = gap >= min_gap
    a, b = a[keep], b[keep]
    same = gp[a] == gp[b]
    mate = ~same & (gt[a] == gt[b])
    opp = gt[a] != gt[b]
    dist = {"same": D[a[same], b[same]], "mate": D[a[mate], b[mate]], "opp": D[a[opp], b[opp]]}
    hits, nq = 0, 0
    far = np.zeros((n, n), bool)
    far[a, b] = far[b, a] = True
    for q in range(n):
        ok = far[q] & (gt == gt[q])
        if not np.any(ok & (gp == gp[q])) or not np.any(ok & (gp != gp[q])):
            continue
        cand = np.flatnonzero(ok)
        hits += int(gp[cand[np.argmin(D[q, cand])]] == gp[q])
        nq += 1
    return dist, (hits / nq if nq else float("nan"))


# ---------------------------------------------------------------------------- crossings
# The situation of 015, with ground-truth identities: two players overlap for a while (a crossing),
# then separate. Is each player, AFTER the crossing, closer to himself BEFORE the crossing than to
# the other player before it? Same moment, same place, same light, often the same pose: harder than
# random pairs from the whole clip, and exactly the decision a swap check has to take.


def crossing_events(gt: dict, min_coverage: float = 0.2, merge_gap: int = 5) -> list[tuple[int, int, int, int]]:
    """(player a, player b, first frame, last frame) of every crossing in a ground truth.

    ``gt``: frame -> (ids, TLBR boxes). Two players cross while one covers the other by at least
    ``min_coverage`` (in either direction). Frames closer than ``merge_gap`` belong to one crossing.
    """
    from ..tracking.matching import coverage_matrix

    touching: dict[tuple[int, int], list[int]] = {}
    for f, (ids, boxes) in gt.items():
        if len(ids) < 2:
            continue
        cov = coverage_matrix(np.asarray(boxes, float))
        cov = np.maximum(cov, cov.T)
        for i, j in zip(*np.nonzero(np.triu(cov >= min_coverage, k=1)), strict=True):
            a, b = sorted((int(ids[i]), int(ids[j])))
            touching.setdefault((a, b), []).append(int(f))
    events = []
    for (a, b), frames in touching.items():
        frames.sort()
        start = prev = frames[0]
        for f in frames[1:] + [None]:
            if f is None or f - prev > merge_gap:
                events.append((a, b, start, prev))
                if f is not None:
                    start = f
            if f is not None:
                prev = f
    return sorted(events, key=lambda e: e[2])


def crossing_decisions(feats, pid, team, frame, events, window: int, min_crops: int = 2,
                       window_after: int | None = None) -> list[dict]:
    """For every crossing with enough clean crops on both sides: keep vs swap distances.

    before = crops in [start - window, start), after = crops in (end, end + window_after] (default: ``window``;
    030 uses a short one, like the online crossing check that cannot wait 2 s after a crossing).
    ``keep`` = d(a_before, a_after) + d(b_before, b_after); ``swap`` = d(a_before, b_after) + d(b_before, a_after),
    each d = mean cosine distance over crop pairs. The model gets the crossing right if keep < swap.
    """
    feats = np.asarray(feats, np.float32)

    def side(p, lo, hi):
        return np.flatnonzero((pid == p) & (frame >= lo) & (frame <= hi))

    def d(x, y):
        return float(np.mean(1.0 - feats[x] @ feats[y].T))

    wa = window if window_after is None else window_after
    out = []
    for a, b, s, e in events:
        ab, aa = side(a, s - window, s - 1), side(a, e + 1, e + wa)
        bb, ba = side(b, s - window, s - 1), side(b, e + 1, e + wa)
        if min(len(ab), len(aa), len(bb), len(ba)) < min_crops:
            continue
        ta, tb = team[pid == a][0], team[pid == b][0]
        keep_a, keep_b = d(ab, aa), d(bb, ba)
        out.append({"a": int(a), "b": int(b), "start": int(s), "end": int(e),
                    "kind": "mate" if ta == tb else "opp",
                    "same": [keep_a, keep_b], "other": [d(ab, ba), d(bb, aa)],
                    "keep": keep_a + keep_b, "swap": d(ab, ba) + d(bb, aa)})
    return out
