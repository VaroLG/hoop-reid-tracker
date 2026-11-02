"""Team label of every clean box of a finished run, without any team annotation (019, 020).

Why
---
019 cuts a tracklet where its team changes for good (a stolen id usually ends on an OPPONENT), then lets
the connector give each piece back to its player. With ground-truth teams that is worth +7.8 IDF1 on the
4 evaluation clips; with the labels of 019 (OSNet, whole body, k = 3) it is worth nothing on the 15 val
clips. The cut rule is fine, the labels are not. This module separates the two questions that make a
labeller:

1. **What describes a box** (``features``): the torso colour, an OSNet embedding of the whole body or of
   the torso only, a SigLIP embedding of the torso... Computed by the caller, one vector per clean box.
2. **How the vectors become teams** (:func:`cluster_team_labels`): the same unsupervised recipe for every
   feature, so that features can be compared on equal terms.

The recipe
----------
* **Tracklet means, one vote each.** One box can look like anything (pose, blur, a hand over the
  number), but the mean over a tracklet is dominated by its kit. Clustering tracklets, not boxes, also
  keeps long tracklets from deciding alone. Tracklets with fewer than ``min_boxes`` boxes do not vote
  (they are still labelled).
* **k clusters.** ``k = 3`` is 019's choice: two teams plus "other" (referees, staff), the cluster with
  the fewest boxes, as in 006. With the players-only detector (011) there are almost no referees left,
  so the third cluster may be carved out of real players. ``k = 2`` tests that.
* **Each box takes the nearest centre**, optionally only when it is clearly closer to it than to the
  second nearest (``reject_ratio``, as the 006 classifier does): an ambiguous box gets no label instead
  of a wrong one. Missing labels are harmless for the cut rule (it only looks at labelled boxes); wrong
  labels create false runs, i.e. false cuts.
* Optionally the vectors are first reduced with UMAP (``umap_dims``), the "SigLIP + UMAP + K-means"
  recipe used for team clustering in sports analytics.

Labels: 0 / 1 = the two teams (arbitrary order), 2 = other, missing = unknown.
"""

from __future__ import annotations

import cv2
import numpy as np

OTHER = 2


def torso_boxes(boxes: np.ndarray, top: float = 0.2, bottom: float = 0.5, side: float = 0.25) -> np.ndarray:
    """The torso of each player box: ``top``-``bottom`` of its height, the central part of its width.

    Same region as the 006 colour classifier and ``hooptrack.eval.gt_teams``: below the head, above the
    shorts, without the arms and the background at the sides. This is where the kit is.
    """
    b = np.asarray(boxes, float).reshape(-1, 4)
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    return np.c_[b[:, 0] + side * w, b[:, 1] + top * h, b[:, 2] - side * w, b[:, 1] + bottom * h]


def lab_torso(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(N, 3) median Lab colour of each torso. The simplest kit descriptor (NaN when too small)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    out = np.full((len(boxes), 3), np.nan, np.float32)
    H, W = frame.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(torso_boxes(boxes).astype(int)):
        patch = lab[max(y1, 0): min(y2, H), max(x1, 0): min(x2, W)]
        if patch.size >= 3 * 20:
            out[i] = np.median(patch.reshape(-1, 3), axis=0)
    return out


def cluster_team_labels(
    keys: list[tuple[int, int]],
    feats: np.ndarray,
    k: int = 3,
    min_boxes: int = 5,
    reject_ratio: float | None = None,
    umap_dims: int | None = None,
    normalise: bool = True,
    seed: int = 0,
) -> dict[tuple[int, int], int]:
    """``{(frame, id): team}`` from one feature vector per labelled box.

    ``keys[i]`` = (frame, tracklet id) of ``feats[i]``. With the defaults (k = 3, no rejection, no UMAP)
    this is exactly 019's ``team_labels_osnet``.

    ``normalise``: L2-normalise each vector (embeddings) or standardise each dimension (``False`` is
    used for colour features, whose dimensions have different units).
    """
    X = np.asarray(feats, np.float32)
    ok = np.isfinite(X).all(axis=1)
    keys = [kk for kk, o in zip(keys, ok, strict=True) if o]
    X = X[ok]
    if len(keys) < k:
        return {}
    # embeddings: unit length (cosine geometry); colours: every dimension on the same scale
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12) if normalise else (X - X.mean(0)) / (X.std(0) + 1e-6)
    if umap_dims:
        import umap  # optional dependency (umap-learn), only for this variant

        X = umap.UMAP(n_components=umap_dims, n_neighbors=30, min_dist=0.0, metric="cosine" if normalise
                      else "euclidean", random_state=seed).fit_transform(X).astype(np.float32)
    tid = np.array([t for _, t in keys])
    tracks = [t for t in np.unique(tid) if (tid == t).sum() >= min_boxes]
    if len(tracks) < k:
        return {}
    means = np.array([X[tid == t].mean(0) for t in tracks], np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    cv2.setRNGSeed(seed)                                   # same clusters on every run
    _, _, centres = cv2.kmeans(means, k, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    d = ((X[:, None, :] - centres[None]) ** 2).sum(-1)     # (boxes, k) squared distances
    nearest = d.argmin(1)
    # name the clusters: with k = 3 the one with the fewest boxes is "other" (006 rule)
    if k == 3:
        other = int(np.argmin(np.bincount(nearest, minlength=3)))
        teams = [c for c in range(3) if c != other]
        name = {teams[0]: 0, teams[1]: 1, other: OTHER}
    else:
        name = {c: c for c in range(k)}
    keep = np.ones(len(keys), bool)
    if reject_ratio is not None:
        ds = np.sqrt(np.sort(d, axis=1))
        keep = ds[:, 0] < reject_ratio * ds[:, 1]          # clearly closer to its cluster than to the next
    return {kk: name[int(n)] for kk, n, kp in zip(keys, nearest, keep, strict=True) if kp}
