"""Per-clip adaptation of the ReID embeddings, with no labels (040).

Why
---
After 038 the identity errors left come from team-mates the ReID model cannot tell apart: on FIBA clips one piece
in five goes to the wrong player. The model (028) learned from 3 basketball games and has never seen these players.
But every clip carries its own supervision, for free, from the tracker's output:

* two crops of the SAME piece (a stretch of one tracklet between two crossings, 015) are the same person;
* two pieces ON SCREEN TOGETHER are two different people.

What it does
------------
A linear map of the embeddings, fitted on one clip in closed form (no training loop, no hyper-parameter search):
the directions along which co-existing pieces differ most, relative to how much the crops of one piece vary.
This is Fisher's linear discriminant with the pieces as classes ("chunklets", as in Relevant Component Analysis)
and only the pairs known to be different people as between-class evidence:

    S_w = mean over crops of (x - mean of its piece)(...)^T                  how one person varies (pose, blur)
    S_b = mean over pairs of co-existing pieces of (m_a - m_b)(m_a - m_b)^T  how two people differ
    directions v: S_b v = lambda S_w v, the ``dim`` largest lambda

Each crop becomes its projection on those directions, L2-normalised. Distances in the new space have another scale
than the original cosine distances; ``distance_scale`` gives the factor that maps the median distance between
co-existing pieces back to its original value (no ground truth involved), so that rules expressed in distance units
(the 038 continuity bonus) keep their meaning.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def _mean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(1.0 - a @ b.T))


def cannot_link_pairs(frames: dict[int, np.ndarray]) -> list[tuple[int, int]]:
    """Pieces that share at least one frame: two different people."""
    by_frame: dict[int, list[int]] = {}
    for p, fs in frames.items():
        for f in fs:
            by_frame.setdefault(int(f), []).append(p)
    pairs = set()
    for ps in by_frame.values():
        ps = sorted(ps)
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                pairs.add((ps[i], ps[j]))
    return sorted(pairs)


def fit_projection(feats: dict[int, np.ndarray], pairs: list[tuple[int, int]], dim: int = 64,
                   ridge: float = 1e-3) -> np.ndarray:
    """(D, dim) projection from the crops of each piece (``feats``: {piece: (n, D)}) and the cannot-link pairs."""
    pieces = [p for p, f in feats.items() if f.ndim == 2 and len(f)]
    D = feats[pieces[0]].shape[1]
    means = {p: feats[p].mean(axis=0) for p in pieces}
    within = np.concatenate([feats[p] - means[p] for p in pieces])
    Sw = within.T @ within / len(within)
    Sw += ridge * np.trace(Sw) / D * np.eye(D)            # a little ridge: S_w must be invertible
    diffs = np.array([means[a] - means[b] for a, b in pairs if a in means and b in means])
    Sb = diffs.T @ diffs / len(diffs)
    vals, vecs = eigh(Sb, Sw)                              # ascending eigenvalues
    return vecs[:, ::-1][:, :dim]


def adapt(feats: dict[int, np.ndarray], frames: dict[int, np.ndarray],
          dim: int = 64) -> tuple[dict[int, np.ndarray], float]:
    """-> ({piece: (n, dim) adapted, normalised crops}, distance scale). Pieces without crops are left as they are."""
    pairs = cannot_link_pairs(frames)
    usable = {p: f for p, f in feats.items() if f.ndim == 2 and len(f)}
    if len(usable) < 3 or not pairs:
        return feats, 1.0
    W = fit_projection(usable, pairs, dim)
    out = dict(feats)
    for p, f in usable.items():
        z = f @ W
        out[p] = (z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    sample = [pr for pr in pairs if pr[0] in usable and pr[1] in usable][:2000]
    before = np.median([_mean_dist(usable[a], usable[b]) for a, b in sample])
    after = np.median([_mean_dist(out[a], out[b]) for a, b in sample])
    return out, float(before / after) if after > 0 else 1.0
