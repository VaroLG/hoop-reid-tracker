"""Held / in the air, frame by frame, with a learned per-frame score and a two-state temporal model (P05).

Why
---
P04's rules get 86 % of the frames right but predict twice as many catches and releases as there are (213 vs 100 on
the test clips): a hard threshold on every frame flickers when the ball passes close to a hand or the ball track is
noisy. Two things are missing: (1) the cues are combined by hand-set thresholds instead of being weighed by data, and
(2) nothing says how often the state really changes (about once every 1.5 s).

How
---
1. **Per-frame features** from the ball track and the player boxes (``frame_features``): is there a ball and was it
   detected or interpolated, distance from the ball to the closest player's "hands" (in his own size), is it inside
   a reach box, how many players are within reach, its speed relative to that player, its own speed, its height on
   the player's box. Frames without a ball get neutral values and the "no ball" flag.
2. **Logistic regression** (scikit-learn, standardised features, default regularisation) on the labelled train
   frames: P(held | features).
3. **Two-state hidden Markov model**, decoded with Viterbi: the logistic posterior divided by the class prior is the
   emission (hybrid HMM), and the probability of changing state per frame is the one counted in the train labels
   (events / frames). No threshold or window is tuned: everything comes from the train labels.
"""

from __future__ import annotations

import numpy as np

FEATURES = ["has_ball", "interp", "hand_dist", "in_reach", "n_reach", "rel_speed", "ball_speed", "height"]


def frame_features(ball: dict[int, tuple], players: dict[int, tuple[np.ndarray, np.ndarray]], n_frames: int,
                   reach_x: float = 0.35, reach_up: float = 0.25, hands_y: float = 0.45) -> np.ndarray:
    """(n_frames, len(FEATURES)). ``ball`` {frame: (cx, cy, d, source)}; ``players`` {frame: (ids, TLBR)}.

    The reach box and the hands height are P01's definitions (geometry, not tuned)."""
    X = np.zeros((n_frames, len(FEATURES)))
    X[:, 2] = 3.0                                       # "far from every hand" for frames without a ball
    near: list[tuple[int, float, float] | None] = [None] * n_frames    # (id, dx, dy) of the closest player
    for f in range(1, n_frames + 1):
        if f not in ball:
            continue
        x, y, d = ball[f][:3]
        src = ball[f][3] if len(ball[f]) > 3 else "det"
        k = f - 1
        X[k, 0], X[k, 1] = 1.0, float(src == "interp")
        ids, boxes = players.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        best, n_in = None, 0
        for i, (x1, y1, x2, y2) in zip(ids, np.asarray(boxes, float), strict=True):
            w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            inside = x1 - reach_x * w <= x <= x2 + reach_x * w and y1 - reach_up * h <= y <= y2
            n_in += inside
            dist = np.hypot((x - (x1 + x2) / 2) / w, (y - (y1 + hands_y * h)) / h)
            if best is None or dist < best[0]:
                best = (dist, inside, int(i), (y - y1) / h, x - (x1 + x2) / 2, y - (y1 + y2) / 2)
        if best is not None:
            X[k, 2], X[k, 3], X[k, 7] = min(best[0], 3.0), float(best[1]), np.clip(best[3], -1, 2)
            near[k] = (best[2], best[4], best[5])
        X[k, 4] = min(n_in, 3)
        if f - 1 in ball:                               # own speed, in ball diameters per frame
            X[k, 6] = min(np.hypot(x - ball[f - 1][0], y - ball[f - 1][1]) / max(d, 1.0), 5.0)
    for k in range(1, n_frames):                        # speed relative to the same closest player
        a, b = near[k - 1], near[k]
        if a is not None and b is not None and a[0] == b[0] and k + 1 in ball:
            X[k, 5] = min(np.hypot(b[1] - a[1], b[2] - a[2]) / max(ball[k + 1][2], 1.0), 5.0)
    return X


def viterbi(p_held: np.ndarray, prior_held: float, p_switch: float) -> np.ndarray:
    """Most likely held (1) / air (0) sequence. Emissions: posterior / prior (hybrid HMM)."""
    eps = 1e-6
    p = np.clip(p_held, eps, 1 - eps)
    em = np.log(np.c_[(1 - p) / (1 - prior_held), p / prior_held])          # (T, 2): air, held
    tr = np.log(np.array([[1 - p_switch, p_switch], [p_switch, 1 - p_switch]]))
    T = len(p)
    score = np.zeros((T, 2))
    back = np.zeros((T, 2), int)
    score[0] = np.log([1 - prior_held, prior_held]) + em[0]
    for t in range(1, T):
        cand = score[t - 1][:, None] + tr                                     # from (row) -> to (column)
        back[t] = np.argmax(cand, axis=0)
        score[t] = cand[back[t], [0, 1]] + em[t]
    out = np.zeros(T, int)
    out[-1] = int(np.argmax(score[-1]))
    for t in range(T - 1, 0, -1):
        out[t - 1] = back[t, out[t]]
    return out
