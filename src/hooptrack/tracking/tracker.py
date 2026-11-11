"""ReIDTracker: online multi-object tracker with long-term re-identification.

Per-frame algorithm (each stage only sees what previous stages left over)
------------------------------------------------------------------------

    0. Predict      Kalman-predict every track, then warp it with the
                    camera motion (GMC).
    1. Stage 1      HIGH-score detections  vs  TRACKED + LOST tracks,
                    fused cost = appearance + IoU, both gated.
    2. Stage 2      LOW-score detections   vs  remaining TRACKED tracks,
                    IoU only (ByteTrack idea: a partially occluded player
                    produces a low-confidence box; it is still him).
    3. Stage 3      remaining HIGH detections vs TENTATIVE tracks (IoU):
                    confirms or kills fresh hypotheses.
    4. Re-ID        remaining clean HIGH detections vs ALL absent identities:
                    recently LOST tracks (appearance + "how far could he have
                    run" gate) and DORMANT identities from long-term memory
                    (appearance only), in one joint assignment with an
                    ambiguity margin. Recovers players after long occlusions
                    and after leaving the frame.
    5. Birth        whatever is still unexplained and confident becomes a
                    new TENTATIVE track (whose id is assigned on promotion,
                    after one more memory check).
    6. Bookkeeping  TRACKED->LOST, LOST->DORMANT (memory), prune memory.

Design principle: every stage is stricter than the one before about what it
accepts, because an identity SWITCH is far more expensive (it corrupts two
trajectories) than a FRAGMENTATION (one trajectory split in two).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gmc import SparseOpticalFlowGMC
from .matching import (
    INF,
    center_distance,
    cosine_distance,
    coverage_matrix,
    eiou_matrix,
    fuse_min,
    fuse_motion_appearance,
    linear_assignment,
)
from .memory import IdentityMemory, ambiguous_columns, gallery_distance
from .track import Track, TrackState


@dataclass
class TrackerConfig:
    """All hyper-parameters (mirrors the ``tracker:`` section of the YAML)."""

    high_thresh: float = 0.5
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    match_thresh: float = 0.8
    second_match_thresh: float = 0.5
    tentative_match_thresh: float = 0.7   # 1 - IoU: a fresh track needs IoU > 0.3
    appearance_weight: float = 0.5
    appearance_gate: float = 0.35
    # Experiment 005a: appearance may only VETO a pair whose IoU is below this.
    # inf = veto at any IoU (behaviour up to 004a).
    appearance_veto_max_iou: float = float("inf")
    # Experiment 006: team constraints (need team labels passed to update()).
    team_gate: bool = False          # forbid matching a track to a box of the OTHER team...
    team_gate_max_iou: float = 0.5   # ...when motion is ambiguous (IoU below this), and always in re-id
    team_cost: float = 0.0           # 006c: soft version, added to the stage-1 cost of cross-team pairs
    suppress_other: bool = False     # do not output tracks voted "other" (referees, staff)
    team_min_votes: int = 3
    team_min_share: float = 0.6
    # 007b: "probation" — a new track is not output until it has a team (0/1), or until it
    # is this many frames old. Hides referees entering the frame before they get votes.
    team_probation: int = 0
    iou_gate: float = 0.2
    # 014: Expansion IoU (Deep-EIoU). Stage 1 compares boxes grown by `eiou_expand` (x their size, each
    # side), then retries the leftovers `eiou_iterations - 1` more times, each time `eiou_step` larger.
    # Stages 2 and 3 use `eiou_expand_low`. All 0 / 1 = plain IoU (behaviour up to 013).
    # Deep-EIoU values: 0.7, step 0.1, 2 iterations, 0.5 for the low-score and unconfirmed stages.
    eiou_expand: float = 0.0
    eiou_step: float = 0.1
    eiou_iterations: int = 1
    eiou_expand_low: float = 0.0
    # 026: two pieces of Deep-EIoU that 025 found to matter, ported. Defaults = behaviour up to 025.
    # fusion: how stage 1 combines motion and appearance.
    #   "weighted" -> appearance_weight * app + (1 - appearance_weight) * (1 - IoU), with gates and veto
    #   "min"      -> min(1 - IoU, app), app only trusted when < min_app_gate and 1 - IoU < min_proximity
    #                 (matching.fuse_min; Deep-EIoU values 0.5 / 0.5 in our cosine scale)
    fusion: str = "weighted"
    min_app_gate: float = 0.5
    min_proximity: float = 0.5
    # assoc_box: which box of a track stages 1-4 compare with the detections.
    #   "kalman"   -> the Kalman prediction, warped by the camera motion (behaviour up to 025)
    #   "last"     -> the last detected box, unchanged (Deep-EIoU: no motion prediction at all)
    #   "last_gmc" -> the last detected box, warped by the camera motion only
    # The Kalman filter still runs in every mode: the OUTPUT box is always the filtered one.
    assoc_box: str = "kalman"
    ema_alpha: float = 0.9
    feature_update_min_score: float = 0.6
    min_hits: int = 2
    max_time_lost: int = 60
    memory_enabled: bool = True
    memory_max_age: int = 900
    memory_thresh: float = 0.25
    memory_margin: float = 0.05
    memory_gallery_size: int = 30
    memory_wait: int = 15        # max frames a new track waits for a clean crop

    gmc: str = "sparse"
    gmc_downscale: int = 2
    # Stage-4 gate: a lost player can drift at most this many body-heights
    # per second of absence (plus one body-height of slack).
    reacquire_speed: float = 1.5
    fps: float = 25.0

    @classmethod
    def from_dict(cls, d: dict) -> TrackerConfig:
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)
        if cfg.fusion not in ("weighted", "min"):
            raise ValueError(f"tracker.fusion must be 'weighted' or 'min', not {cfg.fusion!r}")
        if cfg.assoc_box not in ("kalman", "last", "last_gmc"):
            raise ValueError(f"tracker.assoc_box must be 'kalman', 'last' or 'last_gmc', not {cfg.assoc_box!r}")
        return cfg


@dataclass
class TrackOutput:
    """What the tracker returns for one visible player in one frame."""

    track_id: int
    tlbr: np.ndarray
    score: float
    revived: bool = False  # True on the frame an identity came back from memory


@dataclass
class FrameStats:
    """Per-frame counters, handy for debugging and for the README plots."""

    stage_matches: dict[str, int] = field(default_factory=dict)
    births: int = 0
    revivals: int = 0


class ReIDTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self.frame_id = 0
        self._next_id = 0

        self.tracked: list[Track] = []    # TRACKED + TENTATIVE (visible last frame)
        self.lost: list[Track] = []       # recently lost, still Kalman-predicted
        self.memory = IdentityMemory(
            max_age=self.cfg.memory_max_age,
            thresh=self.cfg.memory_thresh,
            margin=self.cfg.memory_margin,
        )
        self.gmc = SparseOpticalFlowGMC(self.cfg.gmc_downscale) if self.cfg.gmc == "sparse" else None
        self.last_stats = FrameStats()
        # 007c: latest team vote of every confirmed id (read by the offline filter).
        self.final_team: dict[int, int] = {}

    # ------------------------------------------------------------------ utils
    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _new_track(self, tlbr, score, feat) -> Track:
        return Track(
            tlbr, score, feat, self.frame_id,
            ema_alpha=self.cfg.ema_alpha, gallery_size=self.cfg.memory_gallery_size,
        )

    @staticmethod
    def _appearance(
        tracks: list[Track], features: np.ndarray | None, det_idx, reliable: np.ndarray
    ) -> np.ndarray | None:
        """(n_tracks, n_dets) cosine distance with NaN where it can't be trusted.

        NaN marks pairs where the track has no appearance model yet or the
        detection is occluded (crop mixes two players). Downstream stages
        treat NaN as "decide on motion only" (stage 1) or "forbidden"
        (appearance-only stages 4 and 5).
        """
        if features is None:
            return None
        det_idx = np.asarray(det_idx, dtype=int)
        out = np.full((len(tracks), len(det_idx)), np.nan)
        rows = [i for i, t in enumerate(tracks) if t.smooth_feat is not None]
        cols = [j for j, d in enumerate(det_idx) if reliable[d]]
        if rows and cols:
            tf = np.stack([tracks[i].smooth_feat for i in rows])
            out[np.ix_(rows, cols)] = cosine_distance(tf, features[det_idx[cols]])
        return out

    @staticmethod
    def _clean_mask(boxes: np.ndarray, max_coverage: float = 0.2) -> np.ndarray:
        """True for detections whose crop is not significantly covered by another box.

        Only clean crops are trusted for appearance: they update the appearance
        models, seed new tracks and may claim identities in stages 4-5.
        """
        if len(boxes) < 2:
            return np.ones(len(boxes), dtype=bool)
        return coverage_matrix(boxes).max(axis=1) < max_coverage

    def _adopt_from_memory(self, track: Track, competitors: list[Track]) -> int | None:
        """If ``track`` looks like a dormant identity, take over that identity.

        Returns the recovered id, or None if the memory has no confident match.
        """
        if track.smooth_feat is None or len(self.memory) == 0:
            return None
        found = self.memory.query(track.smooth_feat[None], competitors=competitors)
        if not found:
            return None
        dormant, _ = found[0]
        track.track_id = dormant.track_id
        # Merge appearance history: old snapshots + what we saw since re-entry.
        merged = list(dormant.gallery) + list(track.gallery)
        track.gallery.clear()
        track.gallery.extend(merged[-track.gallery.maxlen:])
        return track.track_id

    # ------------------------------------------------------------------ main
    def update(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        features: np.ndarray | None = None,
        frame: np.ndarray | None = None,
        teams: np.ndarray | None = None,
    ) -> list[TrackOutput]:
        """Process one frame.

        Args:
            boxes:    (N, 4) TLBR detections in pixels.
            scores:   (N,) detection confidences.
            features: (N, D) ReID embeddings, or None for a motion-only tracker.
            frame:    BGR image, only needed for camera motion compensation.
            teams:    (N,) team label per box (0/1, 2 = other, -1 = unknown), from
                      ``detection.teams.TeamClassifier``. None = no team logic.

        Returns:
            One :class:`TrackOutput` per confirmed player visible in this frame.
        """
        cfg = self.cfg
        self.frame_id += 1
        stats = FrameStats()

        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        scores = np.asarray(scores, dtype=float).reshape(-1)
        use_app = features is not None and len(features) == len(boxes)
        if not use_app:
            features = None
        clean = self._clean_mask(boxes)
        learn = clean & (scores >= cfg.feature_update_min_score)
        teams_active = teams is not None
        if teams is None or len(teams) != len(boxes):
            teams = np.full(len(boxes), -1, dtype=int)
        teams = np.asarray(teams, dtype=int)

        def team_conflict(tracks: list[Track], det_idx) -> np.ndarray:
            """(n_tracks, n_dets) True where a track and a box belong to different teams.

            Only the two TEAM labels (0 / 1) can conflict. Unknown (-1) never does,
            and neither does 'other' (2): ~7 % of player boxes are labelled
            'other' (odd pose, motion blur), so blocking on it breaks real tracks.
            """
            det_idx = np.asarray(det_idx, dtype=int)
            if not (cfg.team_gate or cfg.team_cost > 0) or not tracks or len(det_idx) == 0:
                return np.zeros((len(tracks), len(det_idx)), dtype=bool)
            tt = np.array([t.team(cfg.team_min_votes, cfg.team_min_share) for t in tracks])
            dt = teams[det_idx]
            team_t, team_d = (tt[:, None] == 0) | (tt[:, None] == 1), (dt[None, :] == 0) | (dt[None, :] == 1)
            return team_t & team_d & (tt[:, None] != dt[None, :])

        # Split detections by confidence (ByteTrack).
        idx_high = np.where(scores >= cfg.high_thresh)[0]
        idx_low = np.where((scores >= cfg.low_thresh) & (scores < cfg.high_thresh))[0]

        # ---------------------------------------------------------- 0. predict
        confirmed = [t for t in self.tracked if t.state == TrackState.TRACKED]
        tentative = [t for t in self.tracked if t.state == TrackState.TENTATIVE]
        pool = confirmed + self.lost                # stage-1 candidates
        for t in pool + tentative:
            t.predict()
        if self.gmc is not None and frame is not None:
            affine = self.gmc.apply(frame, boxes[idx_high] if len(idx_high) else None)
            for t in pool + tentative:
                t.apply_camera_motion(affine, warp_last=cfg.assoc_box == "last_gmc")

        def assoc_boxes(tracks: list[Track]) -> np.ndarray:
            """026: the box each track is matched with (see TrackerConfig.assoc_box)."""
            if cfg.assoc_box == "kalman":
                return np.array([t.tlbr for t in tracks]).reshape(-1, 4)
            return np.array([t.last_tlbr for t in tracks]).reshape(-1, 4)

        matched_tracks: list[Track] = []

        def commit(track: Track, det: int) -> None:
            track.update(
                boxes[det], scores[det],
                None if features is None else features[det],
                self.frame_id, learn_appearance=bool(learn[det]),
            )
            track.add_team_vote(int(teams[det]))
            matched_tracks.append(track)

        # ---------------------------------------------------------- 1. stage 1
        # 014: with Expansion IoU the stage runs `eiou_iterations` times; each pass only sees the
        # tracks and boxes the previous pass left unmatched, with a larger expansion.
        left_pool, left_high = list(range(len(pool))), list(idx_high)
        n_m1 = 0
        for it in range(max(1, cfg.eiou_iterations)):
            if not left_pool or not left_high:
                break
            sub_pool = [pool[r] for r in left_pool]
            pool_boxes = assoc_boxes(sub_pool)
            iou = eiou_matrix(pool_boxes, boxes[left_high], cfg.eiou_expand + cfg.eiou_step * it)
            app = self._appearance(sub_pool, features, np.asarray(left_high, dtype=int), clean)
            if cfg.fusion == "min":                     # 026: Deep-EIoU's rule
                cost = fuse_min(iou, app, cfg.min_app_gate, cfg.min_proximity)
            else:
                cost = fuse_motion_appearance(
                    iou, app, cfg.appearance_weight, cfg.iou_gate, cfg.appearance_gate,
                    veto_max_iou=cfg.appearance_veto_max_iou,
                )
            # 006: when motion is ambiguous, a box of the other team cannot continue this track.
            conflict = team_conflict(sub_pool, left_high)
            if cfg.team_gate:
                cost[conflict & (iou < cfg.team_gate_max_iou)] = INF
            # 006c: at ANY IoU, a cross-team pair pays an extra cost. A lone candidate is
            # still accepted (a wrong label cannot cut a track), but when two players
            # separate after an occlusion the same-team box wins the assignment.
            cost[conflict & (cost < INF)] += cfg.team_cost
            m1, u_p, u_h = linear_assignment(cost, cfg.match_thresh)
            for r, c in m1:
                commit(sub_pool[r], left_high[c])
            n_m1 += len(m1)
            left_pool = [left_pool[r] for r in u_p]
            left_high = [left_high[c] for c in u_h]
        u_pool = left_pool
        stats.stage_matches["s1_fused"] = n_m1
        remaining_high = left_high

        # ---------------------------------------------------------- 2. stage 2
        # Only tracks that were visible last frame: a low-score box is too
        # weak evidence to resurrect a player who has been missing.
        rem_tracked = [pool[r] for r in u_pool if pool[r].state == TrackState.TRACKED]
        rem_lost = [pool[r] for r in u_pool if pool[r].state == TrackState.LOST]
        iou2 = eiou_matrix(assoc_boxes(rem_tracked), boxes[idx_low],
                           cfg.eiou_expand_low)
        m2, u_tr2, _ = linear_assignment(1.0 - iou2, cfg.second_match_thresh)
        for r, c in m2:
            commit(rem_tracked[r], idx_low[c])
        stats.stage_matches["s2_low_iou"] = len(m2)
        newly_lost = [rem_tracked[r] for r in u_tr2]

        # ---------------------------------------------------------- 3. stage 3
        iou3 = eiou_matrix(
            assoc_boxes(tentative), boxes[remaining_high], cfg.eiou_expand_low
        )
        m3, u_tent, u_high3 = linear_assignment(1.0 - iou3, cfg.tentative_match_thresh)
        revived_ids: set[int] = set()
        for r, c in m3:
            t = tentative[r]
            commit(t, remaining_high[c])
            # If dormant identities exist but this track has not seen a clean
            # crop yet, wait (up to memory_wait frames) before minting an id:
            # a few frames of missed output are cheaper than a wrong new id.
            waiting = (
                cfg.memory_enabled and len(self.memory) > 0
                and t.smooth_feat is None and t.hits < cfg.memory_wait
            )
            if t.hits >= cfg.min_hits and not waiting:  # promote to a real identity
                t.state = TrackState.TRACKED
                # Delayed re-identification: a player re-entering next to a
                # team-mate is born from an occluded crop that stage 4 refuses
                # to trust. By promotion time the track may have seen a clean
                # crop: give the memory a second chance BEFORE minting a new id.
                old = self._adopt_from_memory(t, rem_lost) if cfg.memory_enabled else None
                if old is not None:
                    revived_ids.add(old)
                else:
                    t.track_id = self._new_id()
        for r in u_tent:                             # a tentative that misses dies
            tentative[r].state = TrackState.REMOVED
        stats.stage_matches["s3_tentative"] = len(m3)
        remaining_high = [remaining_high[c] for c in u_high3]

        # ---------------------------------------------------------- 4. re-identification
        # Every ABSENT identity competes for the unexplained detections:
        #   * recently LOST tracks — gated by "how far could he have run",
        #   * DORMANT identities in long-term memory — no spatial gate.
        # Solving both in ONE assignment matters: a returning player must be
        # given to his own dormant identity, not to a lost team-mate who
        # merely looks similar and happens to be nearby.
        still_lost = rem_lost
        dormant = self.memory.tracks if cfg.memory_enabled else []
        if use_app and remaining_high and (rem_lost or dormant):
            # Appearance is the only evidence here: occluded crops can't vote.
            dets = [d for d in remaining_high if clean[d]]
            if dets:
                cands = rem_lost + dormant
                nl = len(rem_lost)
                raw = gallery_distance(cands, features[dets])      # (n_cand, n_det)
                cost = raw.copy()
                feasible = np.ones_like(raw, dtype=bool)
                if nl:
                    dist = center_distance(assoc_boxes(rem_lost), boxes[dets])
                    secs = np.array([t.time_since_update for t in rem_lost]) / cfg.fps
                    feasible[:nl] = dist <= (1.0 + cfg.reacquire_speed * secs)[:, None]
                    cost[:nl][raw[:nl] > cfg.appearance_gate] = INF
                if dormant:
                    cost[nl:][raw[nl:] > cfg.memory_thresh] = INF
                    weak = scores[dets] < cfg.new_track_thresh  # only confident boxes revive
                    cost[nl:, weak] = INF
                cost[~feasible] = INF
                if cfg.team_gate:
                    cost[team_conflict(cands, dets)] = INF      # 006: never revive across teams
                # Margin test among all identities that COULD be this player.
                cost[:, ambiguous_columns(np.where(feasible, raw, INF), cfg.memory_margin)] = INF

                m4, _, _ = linear_assignment(cost, INF - 1)    # gates already applied
                matched_dets = set()
                for r, c in m4:
                    det = dets[c]
                    matched_dets.add(det)
                    if r < nl:                                  # lost -> tracked
                        commit(rem_lost[r], det)
                    else:                                       # dormant -> tracked
                        track = self.memory.pop(cands[r].track_id)
                        track.reinitialize(boxes[det], scores[det], self.frame_id)
                        if learn[det]:
                            track.update_feature(features[det])
                        matched_tracks.append(track)
                        revived_ids.add(track.track_id)
                reacquired = {id(rem_lost[r]) for r, _ in m4 if r < nl}
                still_lost = [t for t in rem_lost if id(t) not in reacquired]
                remaining_high = [d for d in remaining_high if d not in matched_dets]
                stats.stage_matches["s4_reid"] = len(m4)
        stats.revivals = len(revived_ids)

        # ---------------------------------------------------------- 5. births
        births: list[Track] = []
        for d in remaining_high:
            if scores[d] < cfg.new_track_thresh:
                continue
            # Seed the appearance model only from a clean crop; otherwise the
            # track will learn its look at its first clean observation.
            seed = features[d] if features is not None and learn[d] else None
            t = self._new_track(boxes[d], scores[d], seed)
            if self.frame_id == 1:   # nothing to be confused with yet
                t.state = TrackState.TRACKED
                t.track_id = self._new_id()
            births.append(t)
        stats.births = len(births)

        # ---------------------------------------------------------- 6. bookkeeping
        for t in newly_lost:
            t.state = TrackState.LOST
        lost = []
        for t in still_lost + newly_lost:
            if self.frame_id - t.frame_id > cfg.max_time_lost:
                if not (cfg.memory_enabled and self.memory.add(t)):
                    t.state = TrackState.REMOVED
            else:
                lost.append(t)

        self.tracked = [
            t for t in matched_tracks + births
            if t.state in (TrackState.TRACKED, TrackState.TENTATIVE)
        ]
        self.lost = lost
        self.memory.prune(self.frame_id)
        for t in self.tracked:
            if t.is_confirmed:
                self.final_team[t.track_id] = t.team(cfg.team_min_votes, cfg.team_min_share)
        self.last_stats = stats

        return [
            TrackOutput(t.track_id, t.tlbr, t.score, revived=t.track_id in revived_ids)
            for t in self.tracked
            if t.state == TrackState.TRACKED and t.frame_id == self.frame_id
            and self._visible(t, teams_active)
        ]

    def _visible(self, t: Track, teams_active: bool) -> bool:
        """Output filter driven by the team vote (006d, 007b). Tracking itself is unaffected."""
        cfg = self.cfg
        team = t.team(cfg.team_min_votes, cfg.team_min_share)
        if cfg.suppress_other and team == 2:
            return False                                   # 006d: referee / staff
        if teams_active and cfg.team_probation > 0 and team not in (0, 1):
            return self.frame_id - t.start_frame >= cfg.team_probation   # 007b: still on probation
        return True

    def reset(self) -> None:
        """Start a new video (ids restart from 1)."""
        self.__init__(self.cfg)
