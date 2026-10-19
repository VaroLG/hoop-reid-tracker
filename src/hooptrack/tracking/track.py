"""A single tracked player and its life cycle.

State machine
-------------

    TENTATIVE ──(min_hits consecutive matches)──▶ TRACKED
        │                                          │  ▲
        │ (missed once)                (missed)    ▼  │ (re-matched)
        ▼                                         LOST
     REMOVED                                       │
                          (lost > max_time_lost)   ▼
                                               DORMANT  ──(revived by ReID)──▶ TRACKED
                                          (stored in IdentityMemory)

The DORMANT state is what makes this tracker different from a plain
ByteTrack: an identity is never forgotten just because the player left the
screen for a few seconds.
"""

from __future__ import annotations

from collections import deque
from enum import IntEnum

import numpy as np

from .kalman import KalmanFilterXYWH
from .matching import normalize


class TrackState(IntEnum):
    TENTATIVE = 0
    TRACKED = 1
    LOST = 2
    DORMANT = 3
    REMOVED = 4


def tlbr_to_xywh(tlbr: np.ndarray) -> np.ndarray:
    """[x1, y1, x2, y2] -> [cx, cy, w, h]."""
    x1, y1, x2, y2 = tlbr
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dtype=float)


def xywh_to_tlbr(xywh: np.ndarray) -> np.ndarray:
    """[cx, cy, w, h] -> [x1, y1, x2, y2]."""
    cx, cy, w, h = xywh[:4]
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=float)


class Track:
    """One player hypothesis: Kalman state + appearance model + bookkeeping."""

    # A single filter instance is shared: it holds only constant matrices.
    kalman = KalmanFilterXYWH()

    def __init__(
        self,
        tlbr: np.ndarray,
        score: float,
        feature: np.ndarray | None,
        frame_id: int,
        ema_alpha: float = 0.9,
        gallery_size: int = 30,
    ) -> None:
        self.track_id: int = -1           # assigned when the track is confirmed
        self.state = TrackState.TENTATIVE
        self.score = float(score)

        self.mean, self.covariance = self.kalman.initiate(tlbr_to_xywh(tlbr))

        # --- appearance model ---------------------------------------------------
        # smooth_feat: exponential moving average -> stable "average look".
        # gallery:     a few raw snapshots -> robust to pose changes; used when
        #              comparing against identities in long-term memory.
        self.ema_alpha = ema_alpha
        self.smooth_feat: np.ndarray | None = None
        self.gallery: deque[np.ndarray] = deque(maxlen=gallery_size)
        if feature is not None:
            self.update_feature(feature)

        # --- bookkeeping -------------------------------------------------------
        self.start_frame = frame_id
        self.frame_id = frame_id          # last frame this track was matched
        self.hits = 1                     # consecutive matched frames
        self.time_since_update = 0

        # --- team votes (experiment 006) ----------------------------------------
        # Per-box team labels are noisy (~3 % wrong); a track's team is the
        # majority of its confident labels: [team A, team B, other].
        self.team_votes = np.zeros(3, dtype=int)

    # ---------------------------------------------------------------- props
    @property
    def tlbr(self) -> np.ndarray:
        return xywh_to_tlbr(self.mean)

    @property
    def is_confirmed(self) -> bool:
        return self.track_id >= 0

    def add_team_vote(self, label: int) -> None:
        if 0 <= label < 3:
            self.team_votes[label] += 1

    def team(self, min_votes: int = 3, min_share: float = 0.6) -> int:
        """Majority team (0, 1, 2 = other) or -1 while the evidence is weak."""
        n = int(self.team_votes.sum())
        if n < min_votes:
            return -1
        k = int(self.team_votes.argmax())
        return k if self.team_votes[k] >= min_share * n else -1

    # ------------------------------------------------------------ appearance
    def update_feature(self, feature: np.ndarray) -> None:
        """Blend a new embedding into the appearance model (EMA + gallery)."""
        feature = normalize(feature)
        if self.smooth_feat is None:
            self.smooth_feat = feature
        else:
            self.smooth_feat = normalize(
                self.ema_alpha * self.smooth_feat + (1 - self.ema_alpha) * feature
            )
        self.gallery.append(feature)

    # ---------------------------------------------------------- life cycle
    def predict(self) -> None:
        """Advance the Kalman state by one frame."""
        mean = self.mean.copy()
        if self.state != TrackState.TRACKED:
            # A lost player's velocity estimate gets stale quickly: freezing
            # the size velocity avoids boxes that shrink/grow to nonsense.
            mean[6] = mean[7] = 0.0
        self.mean, self.covariance = self.kalman.predict(mean, self.covariance)
        self.time_since_update += 1

    def apply_camera_motion(self, affine: np.ndarray) -> None:
        self.mean, self.covariance = self.kalman.apply_affine(self.mean, self.covariance, affine)

    def update(
        self,
        tlbr: np.ndarray,
        score: float,
        feature: np.ndarray | None,
        frame_id: int,
        learn_appearance: bool,
    ) -> None:
        """Correct the track with a matched detection."""
        self.mean, self.covariance = self.kalman.update(
            self.mean, self.covariance, tlbr_to_xywh(tlbr)
        )
        self.score = float(score)
        # Only learn from "clean" crops: a box that overlaps another player
        # contains pixels of BOTH players and would poison the embedding.
        if feature is not None and learn_appearance:
            self.update_feature(feature)

        if self.state == TrackState.TRACKED or self.state == TrackState.TENTATIVE:
            self.hits += 1
        else:  # coming back from LOST
            self.hits = 1
        if self.state != TrackState.TENTATIVE:
            self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.time_since_update = 0

    def reinitialize(self, tlbr: np.ndarray, score: float, frame_id: int) -> None:
        """Restart motion from scratch when an identity is revived from memory.

        The old Kalman state is meaningless after seconds off-screen, so we
        keep the IDENTITY (id + appearance) and reset the MOTION.
        """
        self.mean, self.covariance = self.kalman.initiate(tlbr_to_xywh(tlbr))
        self.score = float(score)
        self.state = TrackState.TRACKED
        self.frame_id = frame_id
        self.hits = 1
        self.time_since_update = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"Track(id={self.track_id}, state={self.state.name}, frame={self.frame_id})"
