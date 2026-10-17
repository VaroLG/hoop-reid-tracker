"""Drawing helpers for qualitative inspection and demo videos."""

from __future__ import annotations

import colorsys

import cv2
import numpy as np


def id_color(track_id: int) -> tuple[int, int, int]:
    """Deterministic, well-spread colour per id (golden-ratio hue stepping)."""
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)  # BGR for OpenCV


def draw_tracks(frame: np.ndarray, outputs, frame_idx: int | None = None) -> np.ndarray:
    """Draw boxes + ids. Identities revived from memory get a thick white ring."""
    img = frame.copy()
    for o in outputs:
        x1, y1, x2, y2 = (int(v) for v in o.tlbr)
        color = id_color(o.track_id)
        if getattr(o, "revived", False):
            cv2.rectangle(img, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (255, 255, 255), 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"#{o.track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    if frame_idx is not None:
        cv2.putText(img, f"frame {frame_idx}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img
