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
    # 035: (len(frames), 2) feet on the court in metres, aligned with frames; NaN where the frame has no court.
    # Filled by the caller (scripts/stitch_tracklets.py, through hooptrack.integrations): this module never sees
    # a homography, so the tracker stays independent of the court code.
    ground: np.ndarray | None = None

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
#             pair is set to FORBIDDEN (+inf) so it is never tried again.
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
    # 023b: a 1-digit read inside a 2-digit one ("3" vs "23") is a partial read, not a conflict
    number_substring: bool = False
    # 035: speed rule on the court. A merge is refused if, at a change of fragment, the player would have had to
    # cover the distance between where one fragment ends and the next begins (feet in metres) faster than
    # ``max_speed`` m/s, with ``speed_slack`` metres of tolerance for the position error of the boxes.
    # None = off (the rule up to 034). Needs Tracklet.ground; frames without a court are not checked.
    max_speed: float | None = None
    speed_slack: float = 1.0
    fps: float = 25.0


def conflicting_numbers(a: str | None, b: str | None, substring_ok: bool = False) -> bool:
    """023: are two jersey numbers different players? Unknown never conflicts.

    ``substring_ok`` (023b): the reader often sees one digit of a two-digit number (a fold, an arm, the edge of
    the crop: "31" read as "3"). Then "3" and "31" are compatible; only numbers that cannot be one another conflict.
    """
    if a is None or b is None or a == b:
        return False
    return not (substring_ok and (a in b or b in a))


def _centres(boxes: np.ndarray) -> np.ndarray:
    return np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]


FORBIDDEN = np.inf     # distance of a pair stitch_gta must never merge, whatever the threshold (029)


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


def _speed_ok(frames: np.ndarray, ground: np.ndarray, owner: np.ndarray, cfg: GTAConfig) -> bool:
    """035: walk the merged group in time; at every change of fragment, could one player cover the gap?

    The pixel rule above compares box centres on the screen, which mean nothing once the camera has panned or
    zoomed between the two fragments (a player who left the view on the left can come back on the right). On the
    court plan a position is a position: going from the last known feet of one fragment to the first known feet
    of the next takes at least distance / max_speed seconds. If the gap between them is shorter (plus the slack
    for the box error), the two fragments are two players.

    Positions may be NaN (frame without a court, feet off the plan): the last valid one of the earlier fragment
    and the first valid one of the later fragment are used, each with its own frame; a fragment with none is
    not checked (no court, no evidence).
    """
    order = np.argsort(frames, kind="stable")
    f, g, o = frames[order], ground[order], owner[order]
    valid = ~np.isnan(g).any(axis=1)
    bounds = np.flatnonzero(o[1:] != o[:-1]) + 1                 # first index of each fragment after the first
    starts = np.r_[0, bounds]
    ends = np.r_[bounds, len(f)]
    for k in range(len(starts) - 1):
        a = np.flatnonzero(valid[starts[k]:ends[k]])             # valid positions of the earlier fragment
        b = np.flatnonzero(valid[starts[k + 1]:ends[k + 1]])     # ... and of the later one
        if not len(a) or not len(b):
            continue
        i, j = starts[k] + a[-1], starts[k + 1] + b[0]            # last known feet, first known feet
        seconds = (f[j] - f[i]) / cfg.fps
        if np.hypot(*(g[j] - g[i])) > cfg.max_speed * seconds + cfg.speed_slack:
            return False
    return True


def stitch_gta(tracklets: list[Tracklet], cfg: GTAConfig, teams: dict[int, int] | None = None,
               numbers: dict[int, str] | None = None) -> tuple[dict[int, int], list[dict]]:
    """GTA connector. Returns ``{old id: new id}`` (new id = earliest id of the group) and a merge log.

    ``teams`` (021, optional): ``{tracklet id: team}`` for the tracklets whose team is known. Two groups
    of DIFFERENT known teams are never merged (like two groups on screen together). A group
    takes the team of its members; tracklets without a team are free to join either. Without ``teams``
    this is exactly GTA.

    ``numbers`` (023, optional): ``{tracklet id: jersey number}``, same rule. Two groups with different known
    numbers are two players, whatever their appearance. Unlike the team, this also separates TEAM-MATES.
    """
    teams, numbers = teams or {}, numbers or {}
    usable = [t for t in tracklets if len(t.feats) >= cfg.min_crops and t.boxes is not None]
    mapping = {t.tid: t.tid for t in tracklets}
    if len(usable) < 2:
        return mapping, []
    allc = _centres(np.concatenate([t.boxes for t in usable]))
    max_dx = (allc[:, 0].max() - allc[:, 0].min()) * cfg.spatial_factor
    max_dy = (allc[:, 1].max() - allc[:, 1].min()) * cfg.spatial_factor

    # one group per tracklet: members, frames, boxes, owner of each box, crop embeddings
    speed = cfg.max_speed is not None
    if speed and any(t.ground is None for t in usable):
        raise ValueError("GTAConfig.max_speed needs Tracklet.ground on every tracklet (NaN where unknown)")
    groups = [{"members": [t.tid], "frames": t.frames.copy(), "boxes": t.boxes.copy(),
               "owner": np.full(len(t.frames), t.tid), "feats": t.feats, "team": teams.get(t.tid),
               "number": numbers.get(t.tid),
               "ground": t.ground.copy() if speed else np.full((len(t.frames), 2), np.nan)}
              for t in usable]

    # 029: a forbidden pair is at distance +inf, never "1.0". The cosine distance goes up to 2, and a ReID model
    # with another scale can need a threshold above 1 (029: fine-tuned SigLIP, 1.04): with 1.0, forbidden pairs
    # then passed the threshold and every piece of a clip was merged into one id. For any threshold below 1 the
    # result is identical to the code up to 028.
    def dist(a: dict, b: dict) -> float:
        if np.intersect1d(a["frames"], b["frames"]).size:
            return FORBIDDEN                             # on screen together: two people
        if a["team"] is not None and b["team"] is not None and a["team"] != b["team"]:
            return FORBIDDEN                             # 021: two different kits: two people
        if conflicting_numbers(a["number"], b["number"], cfg.number_substring):
            return FORBIDDEN                             # 023: two different jersey numbers: two people
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
        ground = np.concatenate([a["ground"], b["ground"]])
        entry = {"a": a["members"], "b": b["members"], "dist": round(float(d), 4)}
        if not _spatially_ok(frames, boxes, owner, max_dx, max_dy):
            D[i, j] = D[j, i] = FORBIDDEN                # never try this pair again
            log.append({**entry, "merged": False, "why": "spatial"})
            continue
        if speed and not _speed_ok(frames, ground, owner, cfg):
            D[i, j] = D[j, i] = FORBIDDEN                # 035: same, for an impossible run on the court
            log.append({**entry, "merged": False, "why": "speed"})
            continue
        order = np.argsort(frames, kind="stable")
        groups[i] = {"members": a["members"] + b["members"], "frames": frames[order], "boxes": boxes[order],
                     "owner": owner[order], "ground": ground[order],
                     "feats": np.concatenate([a["feats"], b["feats"]]),
                     "team": a["team"] if a["team"] is not None else b["team"],
                     "number": a["number"] if a["number"] is not None else b["number"]}
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


# ============================================================================ roster of 10 (036)
# Why: a basketball game has 10 players on the court, 5 per team, and in every SportsMOT clip the ground truth has
# exactly 10 identities. After stitching (029) the clips still end with 11-14 ids: the extra ids hold 6 % of the
# boxes, and 89 % of those boxes are real players whose pieces the threshold left unjoined (the rest: referees and
# other false detections). The threshold stops because it is not sure; the roster rule uses what we KNOW instead:
#
#   roster    the 5 longest groups of each team (team of a group = majority team of its pieces, by boxes) are the
#             10 identities of the clip. Each keeps a gallery: all the crop embeddings of its pieces.
#   assign    every other group is given to one of the 10, in rounds of one Hungarian assignment:
#             - only to a roster player of its own team (unknown team: any),
#             - never to one that is on screen at the same time (two boxes in one frame are two people),
#             - to the closest in appearance (mean cosine distance between the two galleries).
#             After each round the galleries and frames of the roster grow, so two groups on screen together can
#             never end in the same player. No threshold: the number of players is known, so the closest
#             possible player is taken even when the distance is large.
#   leftovers groups that fit nobody (on screen with all 5 players of their team, or without crops): almost surely
#             not a player. Kept with their id, or dropped (``drop`` in the caller): then the clip has at most 10 ids.


def assign_roster(tracklets: list[Tracklet], mapping: dict[int, int], teams: dict[int, int] | None,
                  per_team: int = 5) -> tuple[dict[int, int], set[int], list[dict]]:
    """``mapping`` {piece: group} from the stitching -> ({piece: final id}, leftover group ids, assignment log).

    ``teams``: {piece: team} for the pieces whose team is clear (``piece_teams``). Without any team, the roster is
    the ``2 * per_team`` longest groups and assignments ignore teams.
    """
    from collections import Counter

    from .matching import linear_assignment

    teams = teams or {}
    by_id = {t.tid: t for t in tracklets}
    members: dict[int, list[int]] = {}
    for p, g in mapping.items():
        members.setdefault(g, []).append(p)

    def gallery(ps: list[int]) -> np.ndarray:
        fs = [by_id[p].feats for p in ps if by_id[p].feats.ndim == 2 and len(by_id[p].feats)]
        return np.concatenate(fs) if fs else np.zeros((0, 1), np.float32)

    groups = {}
    for g, ps in members.items():
        votes = Counter()
        for p in ps:
            if p in teams:
                votes[teams[p]] += len(by_id[p].frames)
        groups[g] = {"frames": np.unique(np.concatenate([by_id[p].frames for p in ps])),
                     "feats": gallery(ps), "team": votes.most_common(1)[0][0] if votes else None}

    # the roster: longest groups, per team
    size = lambda g: len(groups[g]["frames"])  # noqa: E731
    known = sorted({v["team"] for v in groups.values() if v["team"] is not None})
    if known:
        roster = [g for t in known for g in sorted((g for g in groups if groups[g]["team"] == t), key=size,
                                                     reverse=True)[:per_team]]
    else:
        roster = sorted(groups, key=size, reverse=True)[:2 * per_team]
    slots = {g: {"frames": groups[g]["frames"], "feats": groups[g]["feats"], "team": groups[g]["team"]}
             for g in roster}
    extras = [g for g in groups if g not in slots and len(groups[g]["feats"])]
    final = {g: g for g in groups}
    log, rnd = [], 0
    while extras:
        rnd += 1
        cost = np.full((len(extras), len(roster)), np.inf)
        for i, e in enumerate(extras):
            ge = groups[e]
            for j, s in enumerate(roster):
                sl = slots[s]
                if ge["team"] is not None and sl["team"] is not None and ge["team"] != sl["team"]:
                    continue                               # another kit
                if np.intersect1d(ge["frames"], sl["frames"]).size or not len(sl["feats"]):
                    continue                               # on screen together: two people
                cost[i, j] = _mean_distance(ge["feats"], sl["feats"])
        matches, _, _ = linear_assignment(np.where(np.isfinite(cost), cost, 1e6), 1e5)
        if not matches:
            break
        for i, j in matches:
            e, s = extras[i], roster[j]
            slots[s]["frames"] = np.union1d(slots[s]["frames"], groups[e]["frames"])
            slots[s]["feats"] = np.concatenate([slots[s]["feats"], groups[e]["feats"]])
            final[e] = s
            log.append({"group": int(e), "player": int(s), "dist": round(float(cost[i, j]), 4), "round": rnd})
        done = {extras[i] for i, _ in matches}
        extras = [e for e in extras if e not in done]
    leftovers = {g for g in groups if g not in slots and final[g] == g}
    return {p: final[g] for p, g in mapping.items()}, leftovers, log


# ============================================================================ closed set of 10 players (037)
# Why: 036 put the "10 players" rule on whole stitched groups and nothing could be assigned, because after the
# online swaps a group holds pieces of several players. The swaps happen at crossings (031-033), so the pieces
# between two crossings (``split_at_crossings``, 015) almost always hold ONE player each. 037 builds the 10
# identities from those pieces directly, as a closed set:
#
#   seeds     a frame where (at least) 10 pieces are on screen together shows 10 DIFFERENT people. The frame whose
#             10 longest pieces are longest in total gives the 10 starting players (no team labels needed: they
#             failed on two clips in 036). Each player keeps a gallery: the embeddings of his pieces.
#   assign    the other pieces are given out one at a time, most confident first: among all pairs (piece, player)
#             where the player has no box in any frame of the piece, the pair with the smallest mean appearance
#             distance to the player's gallery is taken, the gallery grows, and the distances to that player are
#             recomputed. No threshold: a piece of a player must be one of the 10.
#             (First version: rounds of one Hungarian assignment between 10 pieces and the 10 players. Wrong: the
#             Hungarian gives each player at most ONE piece per round, so 10 pieces that are NOT on screen together
#             were still forced onto 10 different players; HOTA 44.6. Kept as ``batch`` for the record.)
#   leftovers pieces that fit nobody (on screen with all 10, i.e. an 11th person: referee, false detection) or have
#             no usable crop keep their own id.


def assign_closed_set(tracklets: list[Tracklet], n_players: int = 10, batch: int | None = None,
                      links: dict[int, set[int]] | None = None, continuity: float = 0.0,
                      dist_scale: float = 1.0, team_feats: dict[int, np.ndarray] | None = None,
                      team_prob: dict[int, float] | None = None, team_penalty: float = 0.0,
                      team_split: bool = True
                      ) -> tuple[dict[int, int], set[int], list[dict]]:
    """Pieces -> ({piece: player id (the seed piece's id)}, leftover pieces, log).

    ``batch`` = None: most-confident-first assignment (037). An integer: the first version, rounds of one
    Hungarian assignment over ``batch`` pieces (wrong, see above; kept to reproduce 037 v1).

    ``links`` + ``continuity`` (038): ``{piece: pieces it touches in the ONLINE run}`` (the piece before and after
    it inside the same online tracklet, cut at a crossing). The online tracker keeps the right identity through
    ~96 % of the crossings (032), which 037 threw away. With ``continuity`` > 0 the distance of a piece to a player
    who already holds one of its linked pieces is lowered by that much: the piece stays with the online identity
    unless its appearance prefers another player by more than ``continuity``. 0 = 037b.

    ``dist_scale`` (040): appearance distances are multiplied by it (adapted embeddings live on another scale;
    ``hooptrack.reid.adapt`` gives the factor), so ``continuity`` keeps its meaning.

    ``team_feats`` (040b): {piece: kit descriptor} (mean zero-shot SigLIP torso vector of the piece). The 10 seeds
    are split into the two teams of 5 that are most compact in kit (all 126 splits tried: the rule "5 per team"
    replaces the k-means of 020, which put almost everything in one cluster on two clips); every piece with a
    descriptor takes the team of the closer centre and can only go to a player of that team.

    ``team_prob`` + ``team_penalty`` (042): {piece: P(kit 1)} from the torso colour of its clean boxes, shrunk
    towards 0.5 when the piece has few of them. A SOFT version of 040b: the 10 seeds are split 5/5 by the split the
    probabilities like most; a piece's distance to a player is raised by ``team_penalty`` x P(piece is NOT in that
    player's team). A long piece clearly of the other kit is pushed away, a short or occluded one barely, so a wrong
    kit label no longer forbids the right player (040b's failure). ``team_split`` = False (042b): no 5/5 split, each
    seed simply takes its own kit (P > 0.5); a referee or a false detection among the seeds made the 5/5 split put a
    real player in the wrong team in 042.
    """
    from .matching import linear_assignment

    with_feats = [t for t in tracklets if t.feats.ndim == 2 and len(t.feats)]
    mapping = {t.tid: t.tid for t in tracklets}
    if len(with_feats) < 2:
        return mapping, set(), []

    # seeds: the frame with >= n_players pieces on screen whose n longest pieces are longest in total
    on_screen: dict[int, list[Tracklet]] = {}
    for t in with_feats:
        for f in t.frames:
            on_screen.setdefault(int(f), []).append(t)
    best, seeds = -1, []
    for f, ts in on_screen.items():
        top = sorted(ts, key=lambda t: len(t.frames), reverse=True)[:n_players]
        score = (len(top), sum(len(t.frames) for t in top))       # most players first, then longest
        if score > (len(seeds), best):
            best, seeds = score[1], top
    players = {t.tid: {"frames": t.frames.copy(), "feats": t.feats} for t in seeds}
    log = [{"piece": int(t.tid), "player": int(t.tid), "seed": True} for t in seeds]

    # 040b: two teams of n/2 among the seeds, each piece's kit by the closer team centre
    team_of: dict[int, int] = {}
    if team_feats and all(t.tid in team_feats for t in seeds) and len(seeds) == n_players and n_players % 2 == 0:
        from itertools import combinations

        ids = [t.tid for t in seeds]
        V = np.array([team_feats[i] / (np.linalg.norm(team_feats[i]) + 1e-12) for i in ids])
        best_split, best_cost = None, np.inf
        for a in combinations(range(len(ids)), len(ids) // 2):
            if 0 not in a:
                continue                                   # each split once
            b = [k for k in range(len(ids)) if k not in a]
            cost = sum(((V[list(g)] - V[list(g)].mean(0)) ** 2).sum() for g in (a, b))
            if cost < best_cost:
                best_split, best_cost = (list(a), b), cost
        centres = [V[g].mean(0) for g in best_split]
        for k, g in enumerate(best_split):
            for i in g:
                team_of[ids[i]] = k
        for p, v in team_feats.items():
            if p not in team_of:
                team_of[p] = int(np.argmax([float(v @ c) for c in centres]))
        log.append({"teams": [[int(ids[i]) for i in g] for g in best_split]})

    # 042: soft kit term; seeds split 5/5 by likelihood
    team_p: dict[int, int] = {}
    if team_prob and team_penalty and not team_split:
        team_p = {t.tid: int(team_prob.get(t.tid, 0.5) > 0.5) for t in seeds}
    elif team_prob and team_penalty and len(seeds) == n_players and n_players % 2 == 0:
        from itertools import combinations

        ids = [t.tid for t in seeds]
        pr = np.clip([team_prob.get(i, 0.5) for i in ids], 1e-3, 1 - 1e-3)
        best_ll, best_a = -np.inf, None
        for a in combinations(range(len(ids)), len(ids) // 2):
            ll = sum(np.log(pr[k]) if k in a else np.log(1 - pr[k]) for k in range(len(ids)))
            if ll > best_ll:
                best_ll, best_a = ll, set(a)
        team_p = {ids[k]: int(k in best_a) for k in range(len(ids))}
        log.append({"teams_soft": [[int(ids[k]) for k in range(len(ids)) if (k in best_a) == side]
                                   for side in (True, False)]})

    queue = sorted((t for t in with_feats if t.tid not in players), key=lambda t: len(t.frames), reverse=True)
    pids = list(players)
    leftovers: set[int] = set()
    links = links or {}
    if batch is None:                                  # 037: most confident pair first
        def pair_cost(t: Tracklet, p: int) -> float:
            if np.intersect1d(t.frames, players[p]["frames"]).size:
                return np.inf                          # never two boxes of one player in one frame
            if team_of and t.tid in team_of and team_of[t.tid] != team_of[p]:
                return np.inf                          # 040b: never to a player of the other team
            d = dist_scale * _mean_distance(t.feats, players[p]["feats"])
            if team_p and t.tid in team_prob:          # 042: P(the piece is not in p's team)
                q = team_prob[t.tid]
                d += team_penalty * ((1 - q) if team_p[p] == 1 else q)
            if continuity and any(mapping.get(n) == p and (n in players or n in assigned) for n in links.get(t.tid, ())):
                d -= continuity                        # 038: the online tracker says it is the same person
            return d

        assigned: set[int] = set()

        def row(t: Tracklet) -> np.ndarray:
            return np.array([pair_cost(t, p) for p in pids])

        cost = np.array([row(t) for t in queue]).reshape(len(queue), len(pids))
        alive = np.ones(len(queue), bool)
        while alive.any():
            masked = np.where(alive[:, None], cost, np.inf)
            i, j = np.unravel_index(np.argmin(masked), masked.shape)
            if not np.isfinite(masked[i, j]):
                break                                  # what is left fits nobody
            t, p = queue[i], pids[j]
            players[p]["frames"] = np.union1d(players[p]["frames"], t.frames)
            players[p]["feats"] = np.concatenate([players[p]["feats"], t.feats])
            mapping[t.tid] = p
            assigned.add(t.tid)
            log.append({"piece": int(t.tid), "player": int(p), "dist": round(float(cost[i, j]), 4)})
            alive[i] = False
            for k in np.flatnonzero(alive):            # only player p changed: refresh his column
                cost[k, j] = pair_cost(queue[k], p)
        leftovers |= {queue[k].tid for k in np.flatnonzero(alive)}
        queue = []
    rnd = 0
    while queue:
        rnd += 1
        chunk, queue = queue[:batch], queue[batch:]
        cost = np.full((len(chunk), len(pids)), np.inf)
        for i, t in enumerate(chunk):
            for j, p in enumerate(pids):
                if not np.intersect1d(t.frames, players[p]["frames"]).size:     # never two boxes in one frame
                    cost[i, j] = _mean_distance(t.feats, players[p]["feats"])
        matches, unmatched, _ = linear_assignment(np.where(np.isfinite(cost), cost, 1e6), 1e5)
        for i, j in matches:
            t, p = chunk[i], pids[j]
            players[p]["frames"] = np.union1d(players[p]["frames"], t.frames)
            players[p]["feats"] = np.concatenate([players[p]["feats"], t.feats])
            mapping[t.tid] = p
            log.append({"piece": int(t.tid), "player": int(p), "dist": round(float(cost[i, j]), 4), "round": rnd})
        # a piece that lost the Hungarian to another piece of the chunk may still fit someone: try it again in
        # the next round; a piece that fits nobody (time conflict with every player) is a leftover
        for i in unmatched:
            (queue.insert(0, chunk[i]) if np.isfinite(cost[i]).any() else leftovers.add(chunk[i].tid))
    usable = {t.tid for t in with_feats}
    leftovers |= {t.tid for t in tracklets if t.tid not in usable}       # no crop: no evidence
    return mapping, leftovers, log


# ============================================================================ global re-assignment (043)
# Why: 042's error breakdown (``scripts/identity_error_breakdown.py``) on 038: 7.2 % of the matched boxes carry a
# wrong identity; 73 % of them are PURE pieces given to the wrong player, and for ~2/3 of those the right player
# was already busy in the piece's frames: 038's most-confident-first assignment never revisits a decision, so an
# early wrong piece blocks the right player for every later piece that overlaps it (a cascade).
# How: start from 038's answer, describe each player by the crops of his current pieces, and re-decide ALL pieces
# at once with an exact integer program (scipy's HiGHS):
#   minimise   sum_p,k x[p,k] * d(p, k)  -  continuity * sum_(p,q linked),k y[p,q,k]  +  extra * sum_p x[p,own]
#   such that  each piece takes exactly one label (a player or its own id),
#              two pieces on screen together never share a player,
#              y[p,q,k] <= x[p,k], y[p,q,k] <= x[q,k]   (the 038 continuity bonus, now symmetric and global).
# d(p, k): mean crop distance of piece p to player k's pieces (p's own crops left out). Players are re-described
# from the new answer and the program solved again, ``n_iter`` times. ``extra`` (the price of an 11th id) is fixed
# above every usual distance, so a piece keeps its own id only when no player is free (as in 038).


def refine_closed_set(tracklets: list[Tracklet], mapping: dict[int, int], links: dict[int, set[int]] | None = None,
                      continuity: float = 0.3, n_iter: int = 3, extra: float = 1.5,
                      dist_scale: float = 1.0, players: list[int] | None = None) -> tuple[dict[int, int], list[dict]]:
    """038's ``{piece: player}`` -> a globally re-decided one, and a log per round. ``players``: the seed ids."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    with_feats = [t for t in tracklets if t.feats.ndim == 2 and len(t.feats)]
    if players is None:
        players = sorted({mapping[t.tid] for t in with_feats if mapping[t.tid] != t.tid})
    if len(players) < 2:
        return dict(mapping), []
    P, K = len(with_feats), len(players)
    idx = {t.tid: i for i, t in enumerate(with_feats)}
    frames = [set(map(int, t.frames)) for t in with_feats]
    overlap = [(i, j) for i in range(P) for j in range(i + 1, P) if frames[i] & frames[j]]
    pairs = sorted({(min(idx[a], idx[b]), max(idx[a], idx[b])) for a, ns in (links or {}).items() if a in idx
                    for b in ns if b in idx and a != b})
    cur = dict(mapping)
    log: list[dict] = []
    for it in range(n_iter):
        members = {k: [i for i, t in enumerate(with_feats) if cur[t.tid] == k] for k in players}
        D = np.full((P, K), 2.0)
        for j, k in enumerate(players):
            if not members[k]:
                continue
            G = np.concatenate([with_feats[i].feats for i in members[k]])
            owner = np.concatenate([np.full(len(with_feats[i].feats), i) for i in members[k]])
            for i, t in enumerate(with_feats):
                S = 1.0 - t.feats @ G.T                    # (crops of p, crops of k)
                keep = owner != i                          # p's own crops left out
                if keep.any():
                    D[i, j] = dist_scale * float(S[:, keep].mean())
        # variables: x[i,j] (P*K), own[i] (P), y[pair,j] (len(pairs)*K)
        nx, ny = P * K, len(pairs) * K
        n = nx + P + ny
        c = np.concatenate([D.ravel(), np.full(P, extra), np.full(ny, -continuity)])
        rows = P + len(overlap) * K + 2 * ny
        A = lil_matrix((rows, n))
        lo, hi = np.zeros(rows), np.zeros(rows)
        r = 0
        for i in range(P):                                 # one label per piece
            A[r, i * K:(i + 1) * K] = 1
            A[r, nx + i] = 1
            lo[r] = hi[r] = 1
            r += 1
        for i, j in overlap:                               # never two boxes of one player in one frame
            for k in range(K):
                A[r, i * K + k] = 1
                A[r, j * K + k] = 1
                lo[r], hi[r] = -np.inf, 1
                r += 1
        for q, (i, j) in enumerate(pairs):                 # continuity: y <= x[i], y <= x[j]
            for k in range(K):
                v = nx + P + q * K + k
                for a in (i, j):
                    A[r, v] = 1
                    A[r, a * K + k] = -1
                    lo[r], hi[r] = -np.inf, 0
                    r += 1
        ub = np.ones(n)
        for i, t in enumerate(with_feats):                 # a seed's own id IS a player's id: no "own" label
            if t.tid in players:
                ub[nx + i] = 0
        res = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=np.ones(n),
                   bounds=Bounds(0, ub), options={"time_limit": 60})
        if res.x is None:
            break
        x = res.x[:nx].reshape(P, K)
        new = dict(cur)
        for i, t in enumerate(with_feats):
            new[t.tid] = players[int(np.argmax(x[i]))] if x[i].max() > 0.5 else t.tid
        moved = [(t.tid, cur[t.tid], new[t.tid]) for t in with_feats if new[t.tid] != cur[t.tid]]
        log.append({"iter": it, "moved": len(moved), "status": int(res.status)})
        cur = new
        if not moved:
            break
    return cur, log


# ============================================================================ smoothing short excursions (039)
# Why: 038 makes more identity switches than 029 but shorter ones: a piece after a crowded crossing goes to the
# wrong player and the identity comes back right after it. In the online run that piece sits BETWEEN two pieces of
# one online tracklet, and both of them were given to the same player B: the tracker followed one person through
# both crossings, and appearance alone moved the middle piece away. Rule, no parameter: if both online neighbours
# of a piece belong to the same player B, and B has no box in that piece's frames, the piece goes to B. Repeated
# until nothing changes (a move can make a new sandwich).


def smooth_sandwiches(tracklets: list[Tracklet], mapping: dict[int, int],
                      links: dict[int, set[int]]) -> tuple[dict[int, int], list[dict]]:
    """``mapping`` {piece: player} + ``links`` {piece: online neighbours} -> (smoothed mapping, log of moves)."""
    frames = {t.tid: t.frames for t in tracklets}
    mapping = dict(mapping)
    log = []
    changed = True
    while changed:
        changed = False
        by_player: dict[int, list[int]] = {}
        for p, g in mapping.items():
            by_player.setdefault(g, []).append(p)
        for p in sorted(mapping, key=lambda q: len(frames[q])):        # shortest pieces first
            nb = links.get(p, set())
            if len(nb) != 2:
                continue                                                # needs a piece before AND after
            owners = {mapping[n] for n in nb}
            if len(owners) != 1:
                continue
            b = owners.pop()
            if b == mapping[p]:
                continue
            others = [q for q in by_player[b] if q != p]
            if others and np.intersect1d(frames[p], np.concatenate([frames[q] for q in others])).size:
                continue                                                # B is on screen elsewhere then
            log.append({"piece": int(p), "from": int(mapping[p]), "to": int(b)})
            by_player[mapping[p]].remove(p)
            by_player[b].append(p)
            mapping[p] = b
            changed = True
    return mapping, log


# ============================================================================ swapped excursions (047)
# Why: in the honest demo video a player's id often leaves him for 1-2 s in a crowd and comes back. Measured on the
# 15 val clips (038): of the 366 identity switches, 136 are a player leaving his id AND getting it back later (89 of
# them within 2 s); the switches back are another ~136. 039's rule could not fix them because they are not single
# pieces between two right ones (039: errors come in RUNS of pieces) and because the right player is not free in
# those frames: two players SWAP labels for a while (X carries Q's label while Y carries X's label P).
# Rule, no parameter: inside each online tracklet, a maximal run of consecutive pieces with label Q whose pieces just
# before and just after both have label P is an excursion (P -> Q -> P). If P is free in the run's frames, the run
# goes back to P (039 at the run level). If P is busy there only because of an excursion the other way (a run with
# label P between two Q's in another online tracklet, overlapping in time), the two runs swap back. Repeated until
# nothing changes. Nothing else is touched.


def _runs_in_chain(chain: list[int], mapping: dict[int, int]) -> list[tuple[int, int, int]]:
    """(start index, end index exclusive, label) of the runs of equal labels along one online tracklet."""
    out, k = [], 0
    while k < len(chain):
        j = k
        while j < len(chain) and mapping[chain[j]] == mapping[chain[k]]:
            j += 1
        out.append((k, j, mapping[chain[k]]))
        k = j
    return out


def smooth_swapped_excursions(tracklets: list[Tracklet], mapping: dict[int, int],
                              chains: list[list[int]], max_rounds: int = 20) -> tuple[dict[int, int], list[dict]]:
    """``mapping`` {piece: player}, ``chains`` = the pieces of each online tracklet in time order ->
    (smoothed mapping, log). See the comment above."""
    frames = {t.tid: set(map(int, t.frames)) for t in tracklets}
    mapping = dict(mapping)
    log: list[dict] = []
    for _ in range(max_rounds):
        exc = []                                     # (pieces of the run, label now, label around)
        for ch in chains:
            ch = [p for p in ch if p in mapping]
            r = _runs_in_chain(ch, mapping)
            for k in range(1, len(r) - 1):
                if r[k - 1][2] == r[k + 1][2] != r[k][2]:
                    exc.append((ch[r[k][0]:r[k][1]], r[k][2], r[k - 1][2]))
        if not exc:
            break
        by_player: dict[int, set[int]] = {}
        for p, g in mapping.items():
            by_player.setdefault(g, set()).add(p)
        moved = False
        used: set[int] = set()
        for i, (run, q, p_) in enumerate(exc):
            if any(x in used for x in run):
                continue
            fr = set().union(*(frames[x] for x in run))
            busy = [x for x in by_player.get(p_, set()) if x not in run and frames[x] & fr]
            if not busy:                             # P free: the run goes back (039 at the run level)
                for x in run:
                    mapping[x] = p_
                used |= set(run)
                log.append({"run": [int(x) for x in run], "from": int(q), "to": int(p_), "swap_with": None})
                moved = True
                continue
            for j, (run2, q2, p2) in enumerate(exc):  # P busy only because of the mirror excursion
                if j == i or q2 != p_ or p2 != q or any(x in used for x in run2):
                    continue
                fr2 = set().union(*(frames[x] for x in run2))
                if not fr & fr2 or not set(busy) <= set(run2):
                    continue
                busy_q = [x for x in by_player.get(q, set()) if x not in run and x not in run2 and frames[x] & fr2]
                if busy_q:
                    continue
                for x in run:
                    mapping[x] = p_
                for x in run2:
                    mapping[x] = q
                used |= set(run) | set(run2)
                log.append({"run": [int(x) for x in run], "from": int(q), "to": int(p_),
                            "swap_with": [int(x) for x in run2]})
                moved = True
                break
        if not moved:
            break
    return mapping, log
