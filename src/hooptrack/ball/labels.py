"""Ball labels: from one click (the centre) to a box, and the per-frame ball/no-ball ground truth (B02-B04).

A label is one click on the ball's centre, or "not visible". The box size is not clicked: a basketball is 0.24 m
across and an adult player about 2 m tall with arms down, so the ball is ~0.12 of the height of a player standing
at the same depth. The players closest to the click in the image are at about that depth (the ball is almost always
near a player), so the diameter is ``BALL_TO_PLAYER`` times the median height of the ``k`` nearest player boxes,
kept within ``[MIN_D, MAX_D]`` pixels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BALL_TO_PLAYER = 0.12          # 0.24 m ball / ~2 m player box
MIN_D, MAX_D = 6.0, 48.0       # pixels, 1280 x 720 frames


def ball_diameter(x: float, y: float, player_boxes: np.ndarray, k: int = 3) -> float:
    """Estimated ball diameter (pixels) at (x, y) from the TLBR player boxes of the same frame."""
    b = np.asarray(player_boxes, float).reshape(-1, 4)
    if not len(b):
        return 16.0
    dx = np.maximum(0, np.maximum(b[:, 0] - x, x - b[:, 2]))
    dy = np.maximum(0, np.maximum(b[:, 1] - y, y - b[:, 3]))
    near = np.argsort(np.hypot(dx, dy))[:k]
    return float(np.clip(BALL_TO_PLAYER * np.median(b[near, 3] - b[near, 1]), MIN_D, MAX_D))


def load_labels(path) -> pd.DataFrame:
    """CSV of ``scripts/ball/b02_label_ball.py`` -> clip, frame, x, y, visible (bool)."""
    d = pd.read_csv(path)
    d["visible"] = d["visible"].astype(int) == 1
    return d
