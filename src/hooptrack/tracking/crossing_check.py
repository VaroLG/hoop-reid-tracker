"""Identity check at the exit of every crossing (experiment 030).

The problem (029, ``scripts/switch_report.py``)
-----------------------------------------------
92 % of the identity switches left in the pipeline happen within a second of two players overlapping, two thirds
of them between players of DIFFERENT teams, and most never undo themselves. They are made by the ONLINE tracker:

* while two boxes overlap, their crops mix both players, so the tracker ignores appearance (``_clean_mask``) and
  motion alone decides who is who when they separate;
* its appearance model is the ImageNet ResNet-50, which hardly tells players apart (016: AUC 0.56 vs team-mates);
* after a wrong decision the track keeps learning the look of the wrong player, so nothing corrects it later.

The idea
--------
Remember how both players looked just BEFORE they overlapped, look at them again just AFTER they separate, and
ask which explanation fits better: "each one is still himself" (keep) or "they have exchanged ids" (swap). If
swap is clearly better, give each track the other's identity. A person checking a video does exactly this.

Cost: the crops are only stored (as small images) while nothing happens. The ReID model, the SigLIP fine-tuned
in 028 (AUC 0.86 vs team-mates, right at 98.8 % of team-mate crossings and 100 % of opponent crossings in the
016 benchmark), runs only when a crossing has to be decided: ~22 crops per crossing, not every box of every frame.

Step by step, per frame (``CrossingCheck.step``, after ``ReIDTracker.update``)
------------------------------------------------------------------------------
1. Visible confirmed tracks and their boxes. A box is CLEAN when no other box covers 20 % or more of it (the
   tracker's and the benchmark's rule) and it is tall enough to be useful.
2. BUFFER: each clean track that is not in a crossing stores a crop every ``sample_every`` frames, the last
   ``before_crops`` of them (8 x 5 frames = 1.6 s at 25 fps).
3. A CROSSING starts when two tracks overlap (coverage >= 20 % in either direction). Their buffers are frozen
   as the "before" look. A crossing involving a third track, too few crops before, a track that disappears or a
   crossing longer than ``max_frames`` is NOT decided: when in doubt, do nothing.
4. While they overlap, nothing is collected. Once they are apart and both clean, ``after_crops`` crops are
   collected for each, ``sample_every`` frames apart (3 x 5 frames = 0.6 s after the separation).
5. DECISION, as in the 016 benchmark (``hooptrack.eval.reid_bench.crossing_decisions``)::

       keep = d(A_before, A_after) + d(B_before, B_after)
       swap = d(A_before, B_after) + d(B_before, A_after)        d = mean cosine distance over crop pairs

   If ``keep - swap > margin``, the two tracks exchange identity (id, appearance model, memory gallery, team
   votes): each track object keeps following the player it is on, but now under the right name. The ``margin`` is
   calibrated on the train split (``scripts/calibrate_crossing_margin.py``): a false swap creates the very error
   this module wants to remove, and most crossings end with no swap at all (015: ~4 %).
6. The ids of the current frame's outputs are corrected too. Frames between the separation and the decision
   (0.6 s) keep the wrong ids: an online tracker cannot change the past.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .matching import coverage_matrix


@dataclass
class CrossingCheckConfig:
    """``crossing_check:`` section of the YAML. Disabled by default (behaviour up to 029)."""

    enabled: bool = False
    reid_type: str = "siglip"           # the model that decides (built with hooptrack.reid.build_reid)
    weights: str | None = None          # e.g. weights/siglip_reid_sportsmot (028)
    min_coverage: float = 0.2           # overlap that starts a crossing / makes a crop unclean
    min_height: float = 50.0            # crops smaller than this are not stored (as the 016 benchmark)
    sample_every: int = 5               # frames between two stored crops (the benchmark's stride)
    before_crops: int = 8               # crops remembered before a crossing (8 x 5 = 1.6 s)
    min_before: int = 2                 # fewer crops before = the crossing is not decided
    after_crops: int = 3                # crops collected after the separation (3 x 5 = 0.6 s)
    max_frames: int = 125               # a crossing not decided after 5 s is dropped
    margin: float = 0.0                 # swap if keep - swap > margin (calibrated on train)
    # 031: "pair" = 030 (two-player crossings only); "group" = GroupCrossingCheck (any number of players)
    mode: str = "pair"
    before_max_age: int = 100           # group: buffered crops older than this (frames, 4 s) are forgotten
    after_wait: int = 50                # group: frames to collect the look after a separation, then decide

    @classmethod
    def from_dict(cls, d: dict | None) -> CrossingCheckConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Crossing:
    """Two tracks that overlap; their look before, and (after they separate) their look after."""

    a: object                           # Track objects (not ids: ids are what may be wrong)
    b: object
    start: int
    before_a: list
    before_b: list
    decide: bool = True                 # False = too unsure (third player, few crops): dropped at exit
    after_a: list = field(default_factory=list)
    after_b: list = field(default_factory=list)
    last_a: int = -10**9                # frame of the last crop collected after the separation
    last_b: int = -10**9


IDENTITY = ("track_id", "smooth_feat", "gallery", "team_votes", "start_frame")


def swap_identity(t1, t2) -> None:
    """The two track objects exchange WHO they are; WHERE they are (Kalman state, last box) stays."""
    for attr in IDENTITY:
        v1, v2 = getattr(t1, attr), getattr(t2, attr)
        setattr(t1, attr, v2)
        setattr(t2, attr, v1)


def mean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Mean cosine distance over all pairs (rows L2-normalised), as ``stitch_gta`` and the benchmark."""
    return float(np.mean(1.0 - x @ y.T))


class CrossingCheck:
    def __init__(self, cfg: CrossingCheckConfig, reid) -> None:
        self.cfg, self.reid = cfg, reid
        self.reset()

    def reset(self) -> None:
        """New video."""
        self.buffers: dict[int, deque] = {}          # id(track object) -> deque of crops (BGR images)
        self.last_sample: dict[int, int] = {}
        self.active: list[Crossing] = []
        self.log: list[dict] = []                    # one entry per crossing that ended (decided or not)
        self.ms = 0.0                                # time spent in the ReID model

    # ------------------------------------------------------------------ helpers
    def _crop(self, frame: np.ndarray, box) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        return frame[y1:y2, x1:x2].copy()

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """L2-normalised embedding of each stored crop (each crop is a 'frame' with one full-size box)."""
        import time

        t0 = time.perf_counter()
        out = np.concatenate([self.reid(c, np.array([[0, 0, c.shape[1], c.shape[0]]], float)) for c in crops])
        self.ms += 1000 * (time.perf_counter() - t0)
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)

    def _end(self, c: Crossing, frame_id: int, outcome: str, **extra) -> None:
        self.active.remove(c)
        self.log.append({"start": c.start, "end": frame_id, "outcome": outcome,
                         "ids": (int(c.a.track_id), int(c.b.track_id)), **extra})

    def _decide(self, c: Crossing, frame_id: int) -> tuple[int, int] | None:
        """keep vs swap. Returns the (id, id) pair to exchange in this frame's outputs, or None."""
        ab, bb = self._embed(c.before_a), self._embed(c.before_b)
        aa, ba = self._embed(c.after_a), self._embed(c.after_b)
        keep = mean_distance(ab, aa) + mean_distance(bb, ba)
        swap = mean_distance(ab, ba) + mean_distance(bb, aa)
        ids = (int(c.a.track_id), int(c.b.track_id))
        swapped = keep - swap > self.cfg.margin
        if swapped:
            swap_identity(c.a, c.b)
        # the frozen crops belonged to the crossing; start fresh afterwards
        for t in (c.a, c.b):
            self.buffers.pop(id(t), None)
        self._end(c, frame_id, "swap" if swapped else "keep", keep=round(keep, 4), swap=round(swap, 4))
        return ids if swapped else None

    # ------------------------------------------------------------------ main
    def step(self, frame: np.ndarray, frame_id: int, tracker, outputs: list) -> list:
        cfg = self.cfg
        shown = {o.track_id for o in outputs}
        tracks = [t for t in tracker.tracked if t.track_id in shown and t.frame_id == tracker.frame_id]
        box_of = {o.track_id: np.asarray(o.tlbr, float) for o in outputs}
        boxes = np.array([box_of[t.track_id] for t in tracks]).reshape(-1, 4)
        k_of = {id(t): k for k, t in enumerate(tracks)}
        n = len(tracks)
        cov = coverage_matrix(boxes) if n > 1 else np.zeros((n, n))
        overlap = np.maximum(cov, cov.T) >= cfg.min_coverage
        tall = (boxes[:, 3] - boxes[:, 1]) >= cfg.min_height if n else np.zeros(0, bool)
        clean = ((cov.max(axis=1) < cfg.min_coverage) if n > 1 else np.ones(n, bool)) & tall
        busy = {id(t) for c in self.active for t in (c.a, c.b)}

        # 1. crossings already open: still overlapping, collecting the look after, or ready to decide
        exchanges = []
        for c in list(self.active):
            ka, kb = k_of.get(id(c.a)), k_of.get(id(c.b))
            if frame_id - c.start > cfg.max_frames:
                self._end(c, frame_id, "timeout")
                continue
            if ka is None or kb is None:
                continue                              # one of them not visible now: wait (timeout applies)
            if overlap[ka, kb]:
                c.after_a.clear()
                c.after_b.clear()                     # still crossing: the "after" starts once they are apart
                continue
            if not c.decide:
                self._end(c, frame_id, "not_decided")
                continue
            for k, crops, last in ((ka, c.after_a, "last_a"), (kb, c.after_b, "last_b")):
                if clean[k] and len(crops) < cfg.after_crops and frame_id - getattr(c, last) >= cfg.sample_every:
                    crops.append(self._crop(frame, boxes[k]))
                    setattr(c, last, frame_id)
            if len(c.after_a) >= cfg.after_crops and len(c.after_b) >= cfg.after_crops:
                ids = self._decide(c, frame_id)
                if ids:
                    exchanges.append(ids)

        # 2. new crossings
        busy = {id(t) for c in self.active for t in (c.a, c.b)}
        for i, j in zip(*np.nonzero(np.triu(overlap, k=1)), strict=True):
            ti, tj = tracks[i], tracks[j]
            if id(ti) in busy and id(tj) in busy and any({id(c.a), id(c.b)} == {id(ti), id(tj)} for c in self.active):
                continue                              # this pair's crossing is already open
            if id(ti) in busy or id(tj) in busy:      # a third player joins: too unsure to decide
                for c in self.active:
                    if {id(c.a), id(c.b)} & {id(ti), id(tj)}:
                        c.decide = False
                continue
            before_i = list(self.buffers.get(id(ti), ()))[-cfg.before_crops:]
            before_j = list(self.buffers.get(id(tj), ()))[-cfg.before_crops:]
            enough = min(len(before_i), len(before_j)) >= cfg.min_before
            self.active.append(Crossing(ti, tj, frame_id, before_i, before_j, decide=enough))
            busy |= {id(ti), id(tj)}

        # 3. remember the look of every clean player who is not crossing
        for k, t in enumerate(tracks):
            if clean[k] and id(t) not in busy and frame_id - self.last_sample.get(id(t), -10**9) >= cfg.sample_every:
                self.buffers.setdefault(id(t), deque(maxlen=cfg.before_crops)).append(self._crop(frame, boxes[k]))
                self.last_sample[id(t)] = frame_id

        # 4. this frame's outputs already carry the corrected names
        for a, b in exchanges:
            for o in outputs:
                if o.track_id == a:
                    o.track_id = b
                elif o.track_id == b:
                    o.track_id = a
        return outputs

    def stats(self) -> dict:
        out = {"crossings": len(self.log), "ms_crossing_reid_total": round(self.ms, 1)}
        for k in ("keep", "swap", "not_decided", "timeout"):
            out[f"crossings_{k}"] = sum(e["outcome"] == k for e in self.log)
        return out


# ============================================================================ 031: crossings of any size
# 030 found that the identity switches happen in crowded crossings (a third player, players with few clean crops
# before), exactly the ones the pair check refused to decide. The group check asks the same question for a whole
# GROUP of players who overlapped: which assignment of "looks before" to "looks after" explains the crops best?
#
#   group      tracks linked by overlaps (A overlaps B, B overlaps C: one group). It grows while it lasts and
#              ends when no two of its VISIBLE members overlap any more. Members that disappear stay in it:
#              their identity can still be the right answer for a visible track.
#   before     per IDENTITY (track id), the last ``before_crops`` clean crops of the last ``before_max_age``
#              frames. The buffer is keyed by id, not by track object, so when an identity moves to another
#              track object its look moves with it. Embeddings of buffered crops are computed once and cached.
#   after      each visible member collects ``after_crops`` clean crops, ``sample_every`` frames apart, as soon as
#              it overlaps no other member (it does not wait for the whole group: in basketball some pair is
#              almost always touching); overlapping again resets ITS crops only. The group is decided when no
#              visible members overlap and the looks after are complete, or ``after_wait`` frames after that,
#              or when it is older than ``max_frames``: then with the members that are complete.
#   decision   cost[track, identity] = mean cosine distance between the track's crops after and the identity's
#              crops before. "Nothing happened" = every track keeps its own identity. The Hungarian algorithm
#              finds the cheapest assignment; it is applied only if it is cheaper than "nothing happened" (no
#              margin: train cannot calibrate one, 030). Identities not taken by a visible track go to the members
#              that are not visible, so ids stay unique.


@dataclass
class Group:
    members: dict                       # id(track) -> Track object
    start: int
    before: dict = field(default_factory=dict)        # id(track) -> list of (frame, crop) frozen when it joined
    after: dict = field(default_factory=dict)         # id(track) -> list of crops after the separation
    last: dict = field(default_factory=dict)          # id(track) -> frame of the last crop after
    separated_at: int | None = None


class GroupCrossingCheck(CrossingCheck):
    """031: identity check for crossings of any number of players (see the comment block above)."""

    def reset(self) -> None:
        super().reset()
        self.groups: list[Group] = []
        self.id_buffers: dict[int, deque] = {}       # track id -> deque of (frame, crop)
        self.emb_cache: dict[tuple[int, int], np.ndarray] = {}   # (track id at storage, frame) -> embedding

    # ---------------------------------------------------------------- looks
    def _before_of(self, t, frame_id: int) -> list:
        """The identity's recent clean crops, frozen: [(key, crop)], key = (id when stored, frame)."""
        buf = self.id_buffers.get(int(t.track_id), ())
        fresh = [(k, c) for k, c in buf if frame_id - k[1] <= self.cfg.before_max_age]
        return fresh[-self.cfg.before_crops:]

    def _embed_before(self, items: list) -> np.ndarray:
        missing = [(k, c) for k, c in items if k not in self.emb_cache]
        if missing:
            for (k, _), e in zip(missing, self._embed([c for _, c in missing]), strict=True):
                self.emb_cache[k] = e
        return np.stack([self.emb_cache[k] for k, _ in items])

    # ---------------------------------------------------------------- decision
    def _decide_group(self, g: Group, frame_id: int, tracker) -> dict[int, int]:
        """Returns {old id: new id} for this frame's outputs (empty if nothing changes)."""
        from .matching import linear_assignment

        cols = [k for k in g.members if len(g.before.get(k, ())) >= self.cfg.min_before]
        rows = [k for k in cols if len(g.after.get(k, ())) >= self.cfg.after_crops]
        info = {"members": len(g.members), "with_look_before": len(cols), "decided_tracks": len(rows)}
        if len(cols) < 2 or len(rows) < 1 or (len(rows) < 2 and len(cols) == len(rows)):
            self._log_group(g, frame_id, "not_decided", reason="few_looks", **info)
            return {}
        before = {k: self._embed_before(g.before[k]) for k in cols}
        after = {k: self._embed(g.after[k]) for k in rows}
        cost = np.array([[mean_distance(after[r], before[c]) for c in cols] for r in rows])
        own = np.array([cost[i, cols.index(r)] for i, r in enumerate(rows)])
        matches, _, _ = linear_assignment(cost, np.inf)
        best = {rows[i]: cols[j] for i, j in matches}
        new_cost = sum(cost[rows.index(r), cols.index(c)] for r, c in best.items())
        changed = {r: c for r, c in best.items() if c != r}
        if not changed or not new_cost < own.sum() - 1e-9:
            self._log_group(g, frame_id, "keep", cost_keep=round(float(own.sum()), 4),
                            cost_best=round(float(new_cost), 4), **info)
            return {}
        # identity permutation: track r takes the identity of member c; leftover identities go to the others
        ident = {k: tuple(getattr(g.members[k], a) for a in IDENTITY) for k in g.members}
        taken = set(best.values())
        receivers = [k for k in g.members if k not in best]               # members not decided (not visible...)
        leftovers = [k for k in g.members if k not in taken]              # identities nobody took
        target = dict(best)
        for k in receivers:                                               # keep your own identity if still free
            if k in leftovers:
                target[k] = k
                leftovers.remove(k)
        for k in receivers:
            if k not in target:
                target[k] = leftovers.pop()
        old_ids = {k: int(g.members[k].track_id) for k in g.members}
        for k, src in target.items():
            for a, v in zip(IDENTITY, ident[src], strict=True):
                setattr(g.members[k], a, v)
        self._log_group(g, frame_id, "reassign", cost_keep=round(float(own.sum()), 4),
                        cost_best=round(float(new_cost), 4),
                        moves=";".join(f"{old_ids[r]}->{old_ids[c]}" for r, c in changed.items()), **info)
        return {old_ids[k]: old_ids[src] for k, src in target.items() if k != src}

    def _log_group(self, g: Group, frame_id: int, outcome: str, **extra) -> None:
        self.groups.remove(g)
        self.log.append({"start": g.start, "end": frame_id, "outcome": outcome,
                         "ids": tuple(int(t.track_id) for t in g.members.values()), **extra})

    # ---------------------------------------------------------------- main
    def step(self, frame: np.ndarray, frame_id: int, tracker, outputs: list) -> list:
        cfg = self.cfg
        shown = {o.track_id for o in outputs}
        tracks = [t for t in tracker.tracked if t.track_id in shown and t.frame_id == tracker.frame_id]
        box_of = {o.track_id: np.asarray(o.tlbr, float) for o in outputs}
        boxes = np.array([box_of[t.track_id] for t in tracks]).reshape(-1, 4)
        k_of = {id(t): k for k, t in enumerate(tracks)}
        n = len(tracks)
        cov = coverage_matrix(boxes) if n > 1 else np.zeros((n, n))
        overlap = np.maximum(cov, cov.T) >= cfg.min_coverage
        tall = (boxes[:, 3] - boxes[:, 1]) >= cfg.min_height if n else np.zeros(0, bool)
        clean = ((cov.max(axis=1) < cfg.min_coverage) if n > 1 else np.ones(n, bool)) & tall

        # 1. grow / merge groups with this frame's overlaps
        group_of = {k: g for g in self.groups for k in g.members}
        for i, j in zip(*np.nonzero(np.triu(overlap, k=1)), strict=True):
            ti, tj = tracks[i], tracks[j]
            gi, gj = group_of.get(id(ti)), group_of.get(id(tj))
            if gi is None and gj is None:
                g = Group(members={}, start=frame_id)
                self.groups.append(g)
            elif gi is not None and gj is not None and gi is not gj:  # two groups meet: one group
                gi.members.update(gj.members)
                gi.before.update(gj.before)
                gi.after.update(gj.after)
                gi.last.update(gj.last)
                gi.start = min(gi.start, gj.start)
                self.groups.remove(gj)
                for k in gj.members:
                    group_of[k] = gi
                g = gi
            else:
                g = gi or gj
            for t in (ti, tj):
                if id(t) not in g.members:
                    g.members[id(t)] = t
                    g.before[id(t)] = self._before_of(t, frame_id)   # its look, frozen when it joins
                group_of[id(t)] = g
                g.after.pop(id(t), None)                              # still crossing: its "after" restarts
            g.separated_at = None

        # 2. groups whose visible members no longer overlap: collect the look after, then decide
        remap: dict[int, int] = {}
        for g in list(self.groups):
            vis = [k for k in g.members if k in k_of]
            alone = {k for k in vis if not any(overlap[k_of[k], k_of[o]] for o in vis if o != k)}
            for k in alone:                                          # apart from the others: look after
                kk = k_of[k]
                if clean[kk] and len(g.after.get(k, [])) < cfg.after_crops \
                        and frame_id - g.last.get(k, -10**9) >= cfg.sample_every:
                    g.after.setdefault(k, []).append(self._crop(frame, boxes[kk]))
                    g.last[k] = frame_id
            if frame_id - g.start > cfg.max_frames:                  # long group: decide with what is complete
                remap.update(self._decide_group(g, frame_id, tracker))
                continue
            if len(alone) < len(vis):
                continue                                             # some members still touching
            if g.separated_at is None:
                g.separated_at = frame_id
            looks = [k for k in g.members if len(g.before.get(k, ())) >= cfg.min_before]
            complete = all(len(g.after.get(k, ())) >= cfg.after_crops for k in looks if k in vis)
            if complete or frame_id - g.separated_at >= cfg.after_wait:
                remap.update(self._decide_group(g, frame_id, tracker))

        # 3. remember the look of every clean player who is not in a group (keyed by IDENTITY)
        in_group = {k for g in self.groups for k in g.members}
        for kk, t in enumerate(tracks):
            if clean[kk] and id(t) not in in_group:
                buf = self.id_buffers.setdefault(int(t.track_id), deque(maxlen=self.cfg.before_crops))
                if not buf or frame_id - buf[-1][0][1] >= cfg.sample_every:
                    buf.append(((int(t.track_id), frame_id), self._crop(frame, boxes[kk])))

        # 4. this frame's outputs carry the corrected names
        for o in outputs:
            o.track_id = remap.get(o.track_id, o.track_id)
        return outputs

    def stats(self) -> dict:
        out = {"groups": len(self.log), "ms_crossing_reid_total": round(self.ms, 1)}
        for k in ("keep", "reassign", "not_decided"):
            out[f"groups_{k}"] = sum(e["outcome"] == k for e in self.log)
        return out
