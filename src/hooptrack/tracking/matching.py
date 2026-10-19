"""Cost matrices and optimal assignment.

Data association = "which detection belongs to which track?". We phrase it as
a bipartite matching problem: build a cost matrix C[i, j] (track i vs
detection j) and solve it with the Hungarian algorithm, which returns the
assignment with the minimum total cost.

All boxes in this module are in TLBR format: [x1, y1, x2, y2].
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

# Any cost >= INF is treated as "forbidden pair".
INF = 1e5


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise Intersection-over-Union, fully vectorised.

    Args:
        boxes_a: (N, 4) TLBR boxes.
        boxes_b: (M, 4) TLBR boxes.

    Returns:
        (N, M) matrix with values in [0, 1].
    """
    boxes_a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    # Broadcasting: (N, 1) against (1, M) gives every pair at once.
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def coverage_matrix(boxes: np.ndarray) -> np.ndarray:
    """Fraction of box i covered by box j: inter(i, j) / area(i). Diagonal = 0.

    Better than IoU for occlusion reasoning: a small far-away player half
    hidden behind a big close-up player has a LOW IoU with him (the union is
    huge) but a HIGH coverage — and his crop is mostly the other player.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if len(boxes) == 0:
        return np.zeros((0, 0))
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    cov = inter / np.maximum(area[:, None], 1e-9)
    np.fill_diagonal(cov, 0.0)
    return cov


def normalize(features: np.ndarray) -> np.ndarray:
    """L2-normalise rows so that a dot product equals cosine similarity."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def cosine_distance(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
    """Pairwise cosine distance in [0, 1]  (0 = identical appearance).

    ReID embeddings live on a hypersphere; the angle between two vectors is a
    much better identity signal than their Euclidean distance.
    """
    if len(feats_a) == 0 or len(feats_b) == 0:
        return np.zeros((len(feats_a), len(feats_b)))
    sim = normalize(feats_a) @ normalize(feats_b).T
    # Map similarity [-1, 1] -> distance [0, 1]; negatives are "very different".
    return np.clip(1.0 - sim, 0.0, 1.0)


def center_distance(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Centre-to-centre distance normalised by the height of box A.

    Expressed in "body heights" it is roughly scale-invariant: 1.0 means the
    detection is one player-height away from where the track was predicted.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ca = np.c_[(boxes_a[:, 0] + boxes_a[:, 2]) / 2, (boxes_a[:, 1] + boxes_a[:, 3]) / 2]
    cb = np.c_[(boxes_b[:, 0] + boxes_b[:, 2]) / 2, (boxes_b[:, 1] + boxes_b[:, 3]) / 2]
    ha = np.maximum(boxes_a[:, 3] - boxes_a[:, 1], 1.0)
    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    return dist / ha[:, None]


def linear_assignment(
    cost: np.ndarray, thresh: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian matching with a rejection threshold.

    The Hungarian algorithm always pairs as many rows/cols as possible, even if
    a pair is terrible. We therefore (1) forbid pairs above ``thresh`` before
    solving and (2) double-check every returned pair afterwards.

    Returns:
        matches:          list of (row, col) pairs
        unmatched_rows:   row indices without a partner
        unmatched_cols:   column indices without a partner
    """
    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    gated = np.where(cost > thresh, INF, cost)
    rows, cols = linear_sum_assignment(gated)

    matches = [(r, c) for r, c in zip(rows, cols, strict=True) if gated[r, c] < INF]
    matched_r = {r for r, _ in matches}
    matched_c = {c for _, c in matches}
    unmatched_rows = [r for r in range(n_rows) if r not in matched_r]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_c]
    return matches, unmatched_rows, unmatched_cols


def fuse_motion_appearance(
    iou: np.ndarray,
    app_dist: np.ndarray | None,
    appearance_weight: float,
    iou_gate: float,
    appearance_gate: float,
    veto_max_iou: float = float("inf"),
) -> np.ndarray:
    """Combine motion (IoU) and appearance (ReID) into one cost.

    cost = lambda * d_app + (1 - lambda) * (1 - IoU)

    Two gates make the fusion safe:
      * spatial gate   — IoU < ``iou_gate``: players cannot teleport, so a
        great appearance match far away is still forbidden here (long jumps
        are handled by the dedicated re-acquisition stages).
      * appearance gate — d_app > ``appearance_gate``: overlapping boxes of
        two DIFFERENT players (screens, post-ups) must not swap identities just
        because the boxes overlap. This is the key anti-ID-switch rule.

    ``app_dist`` may contain NaN for pairs where appearance is UNRELIABLE
    (the detection overlaps another player, so its crop mixes two people, or
    the track has no appearance model yet). Those pairs fall back to the
    motion cost alone instead of being judged on a corrupted embedding.

    Veto vs soft cost (experiment 005a)
    -----------------------------------
    By default the appearance gate is a hard VETO at any IoU. Experiment 004a
    showed the price: when a player's look changes for a few frames (motion
    blur, a new pose, a partial occlusion) the veto rejects a match that motion
    got right, the track is cut, and the player comes back with a new id.

    ``veto_max_iou`` limits the veto to pairs whose IoU is BELOW that value:

        IoU >= veto_max_iou  -> motion is unambiguous: appearance only adds cost
        IoU <  veto_max_iou  -> appearance may still forbid the pair

    The default (``inf``) keeps the veto everywhere = behaviour before 005a.
    The Hungarian step still uses the appearance term, so when two players
    overlap the better-looking match keeps winning; it just can no longer
    turn a clear motion match into "no match at all".
    """
    motion = 1.0 - iou
    if app_dist is None:
        cost = motion.copy()
        cost[iou < iou_gate] = INF
        return cost

    valid = ~np.isnan(app_dist)
    fused = appearance_weight * np.nan_to_num(app_dist) + (1.0 - appearance_weight) * motion
    cost = np.where(valid, fused, motion)
    vetoed = valid & (np.nan_to_num(app_dist) > appearance_gate) & (iou < veto_max_iou)
    cost[(iou < iou_gate) | vetoed] = INF
    return cost
