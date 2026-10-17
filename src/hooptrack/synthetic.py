"""Synthetic tracking scenes with known ground truth.

Used by the unit tests and by ``scripts/smoke_test.py`` so the whole pipeline
(tracker -> MOT files -> metrics) can be exercised in CI without downloading
the 13 GB dataset or running a GPU.

Each simulated player has:
  * a trajectory (constant velocity + noise, bouncing on the image borders),
  * an "identity vector" — a random unit vector standing in for a ReID
    embedding — plus per-frame noise,
  * optional scripted events: leaving the frame for K frames (re-entry test)
    or being hidden behind another player (occlusion test).

Team-mates share part of their identity vector (same jersey colour), which
makes the appearance problem realistically hard: a naive tracker will swap
team-mates, a good one will not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticFrame:
    gt_ids: np.ndarray            # (G,)
    gt_boxes: np.ndarray          # (G, 4) TLBR
    det_boxes: np.ndarray         # (N, 4) TLBR
    det_scores: np.ndarray        # (N,)
    det_features: np.ndarray      # (N, D)
    det_gt_ids: np.ndarray        # (N,) gt id per detection, -1 for false positives


@dataclass
class SceneConfig:
    n_players: int = 10
    n_frames: int = 300
    width: int = 1280
    height: int = 720
    feat_dim: int = 64
    team_similarity: float = 0.6      # how much team-mates' embeddings overlap
    feat_noise: float = 0.15
    box_noise: float = 2.0            # px std-dev of detection jitter
    miss_rate: float = 0.03           # random missed detections
    fp_rate: float = 0.02             # false positives per frame (Poisson mean * players)
    # Scripted exits: list of (player_index, start_frame, duration)
    exits: list[tuple[int, int, int]] = field(default_factory=list)
    seed: int = 0


def make_scene(cfg: SceneConfig) -> list[SyntheticFrame]:
    rng = np.random.default_rng(cfg.seed)
    P, D = cfg.n_players, cfg.feat_dim

    # --- appearance: team component + individual component -------------------
    team_vecs = rng.normal(size=(2, D))
    indiv = rng.normal(size=(P, D))
    teams = np.arange(P) % 2
    ident = cfg.team_similarity * team_vecs[teams] + (1 - cfg.team_similarity) * indiv
    ident /= np.linalg.norm(ident, axis=1, keepdims=True)

    # --- motion -------------------------------------------------------------
    h = rng.uniform(140, 220, size=P)           # player height in px
    w = h * rng.uniform(0.35, 0.5, size=P)
    pos = np.c_[rng.uniform(100, cfg.width - 100, P), rng.uniform(200, cfg.height - 100, P)]
    vel = rng.normal(0, 4, size=(P, 2))

    hidden = np.zeros((cfg.n_frames, P), dtype=bool)
    for p, start, dur in cfg.exits:
        hidden[start : start + dur, p] = True

    frames = []
    for f in range(cfg.n_frames):
        vel += rng.normal(0, 0.5, size=vel.shape)
        vel = np.clip(vel, -8, 8)
        pos += vel
        # bounce on the borders
        for k, lim in enumerate((cfg.width, cfg.height)):
            low, high = 60, lim - 60
            out = (pos[:, k] < low) | (pos[:, k] > high)
            vel[out, k] *= -1
            pos[:, k] = np.clip(pos[:, k], low, high)

        boxes = np.c_[pos[:, 0] - w / 2, pos[:, 1] - h / 2, pos[:, 0] + w / 2, pos[:, 1] + h / 2]
        visible = ~hidden[f]
        gt_ids = np.arange(1, P + 1)[visible]
        gt_boxes = boxes[visible]

        det_mask = visible & (rng.random(P) > cfg.miss_rate)
        db = boxes[det_mask] + rng.normal(0, cfg.box_noise, size=(det_mask.sum(), 4))
        ds = rng.uniform(0.55, 0.95, size=det_mask.sum())
        df = ident[det_mask] + rng.normal(0, cfg.feat_noise / np.sqrt(D), size=(det_mask.sum(), D))
        dg = np.arange(1, P + 1)[det_mask]

        n_fp = rng.poisson(cfg.fp_rate * P)
        if n_fp:
            c = np.c_[rng.uniform(100, cfg.width - 100, n_fp), rng.uniform(100, cfg.height - 100, n_fp)]
            fb = np.c_[c - [30, 60], c + [30, 60]]
            db = np.r_[db, fb]
            ds = np.r_[ds, rng.uniform(0.1, 0.45, n_fp)]   # FPs are mostly low-score
            df = np.r_[df, rng.normal(size=(n_fp, D))]
            dg = np.r_[dg, -np.ones(n_fp, dtype=int)]

        frames.append(SyntheticFrame(gt_ids, gt_boxes, db, ds, df, dg))
    return frames
