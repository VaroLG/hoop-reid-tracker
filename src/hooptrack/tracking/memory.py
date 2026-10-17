"""Long-term identity memory (the "anti-substitution" module).

Problem it solves
-----------------
A short-term tracker deletes a track after N frames without detections. In
basketball a player often leaves the broadcast frame (fast break, camera
following the ball) for several seconds. When they come back a plain tracker
creates a NEW id: the same player now has two identities, and in the worst
case their old id is later stolen by someone else. IDF1 and AssA punish this
heavily.

Solution
--------
When a confirmed track has been lost for too long we do not delete it: we
move it into this gallery as a DORMANT identity, together with several
appearance snapshots. Before the tracker spawns a new id for a confident,
unexplained detection, it asks "have I seen this player before?".

The decision is protected by two guards:

* an absolute threshold on the cosine distance, and
* a margin test: the best identity must be clearly better than the second
  best among ALL absent identities (dormant ones and recently lost ones).
  Two team-mates with the same jersey produce near-ties; in that case we
  prefer a new id (a fragmentation — cheap) over reviving the wrong person
  (an identity switch — expensive).
"""

from __future__ import annotations

import numpy as np

from .matching import INF, cosine_distance, linear_assignment
from .track import Track, TrackState


def gallery_distance(tracks: list[Track], det_features: np.ndarray) -> np.ndarray:
    """(n_tracks, n_dets) nearest-neighbour cosine distance over each gallery.

    Using the MIN over a gallery of snapshots means a player seen from behind
    still matches a snapshot taken from behind, even if his average look
    (the EMA feature) is dominated by frontal views. Tracks without any
    snapshot get +inf.
    """
    out = np.full((len(tracks), len(det_features)), INF)
    if len(det_features) == 0:
        return out
    for i, t in enumerate(tracks):
        if len(t.gallery):
            out[i] = cosine_distance(np.stack(t.gallery), det_features).min(axis=0)
    return out


def ambiguous_columns(dist: np.ndarray, margin: float) -> np.ndarray:
    """True for columns (detections) where the best and 2nd-best rows nearly tie."""
    if dist.shape[0] < 2 or dist.shape[1] == 0:
        return np.zeros(dist.shape[1], dtype=bool)
    s = np.sort(dist, axis=0)
    return (s[1] - s[0]) < margin


class IdentityMemory:
    def __init__(self, max_age: int, thresh: float, margin: float) -> None:
        self.max_age = max_age
        self.thresh = thresh
        self.margin = margin
        self._dormant: dict[int, Track] = {}

    def __len__(self) -> int:
        return len(self._dormant)

    @property
    def tracks(self) -> list[Track]:
        return list(self._dormant.values())

    def add(self, track: Track) -> bool:
        """Store a confirmed track that has an appearance model."""
        if not track.is_confirmed or len(track.gallery) == 0:
            return False
        track.state = TrackState.DORMANT
        self._dormant[track.track_id] = track
        return True

    def pop(self, track_id: int) -> Track:
        return self._dormant.pop(track_id)

    def prune(self, frame_id: int) -> None:
        """Forget identities that have been off-screen for too long."""
        expired = [tid for tid, t in self._dormant.items() if frame_id - t.frame_id > self.max_age]
        for tid in expired:
            self._dormant[tid].state = TrackState.REMOVED
            del self._dormant[tid]

    def query(
        self, det_features: np.ndarray, competitors: list[Track] | None = None
    ) -> list[tuple[Track, int]]:
        """Return (revived_track, detection_index) pairs and remove them from memory.

        ``competitors`` are other absent identities (e.g. recently LOST tracks)
        that take part in the margin test but cannot be revived here.
        """
        if len(self._dormant) == 0 or len(det_features) == 0:
            return []
        tracks = self.tracks
        dist = gallery_distance(tracks, det_features)
        pool = dist if not competitors else np.vstack([dist, gallery_distance(competitors, det_features)])
        dist[:, ambiguous_columns(pool, self.margin)] = INF

        matches, _, _ = linear_assignment(dist, self.thresh)
        revived = []
        for i, j in matches:
            revived.append((self.pop(tracks[i].track_id), j))
        return revived
