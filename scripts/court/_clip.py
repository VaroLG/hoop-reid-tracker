"""Glue between the court chapter and SportsMOT data, shared by the C04 scripts.

Scripts may import both chapters (they are the glue); ``src/hooptrack/court``
may not. Here: find a sequence, read its frames, and get the cached person
detections (to mask players out of the floor).
"""

from __future__ import annotations

import cv2
import numpy as np

from hooptrack.cli import _sequences
from hooptrack.config import load_config
from hooptrack.detection import build_detector


class Clip:
    def __init__(self, sequence: str, config: str = "configs/default.yaml",
                 cache_dir: str = "outputs/detections/yolo26m", min_score: float = 0.3) -> None:
        for split in ("val", "train", "test"):                     # find the clip in whichever split holds it
            cfg = load_config(config, ["detector.type=cached", f"detector.cache_dir={cache_dir}",
                                       f"data.only=[{sequence}]", f"data.split={split}"])
            try:
                self.seq = _sequences(cfg)[0]
                break
            except SystemExit:
                continue
        else:
            raise FileNotFoundError(f"sequence {sequence} not found in any split")
        self.det = build_detector(cfg["detector"])
        try:
            self.det.bind(self.seq)
            self.has_people = True
        except FileNotFoundError:                                   # no cached detections for this clip
            self.has_people = False
        self.min_score = min_score
        self.length = self.seq.length

    def frame(self, f: int) -> np.ndarray:
        return cv2.imread(str(self.seq.frame_path(f)))

    def people(self, f: int) -> np.ndarray | None:
        if not self.has_people:
            return None
        b, s = self.det(None, frame_idx=f)
        return b[s >= self.min_score]

    def frames(self, start: int, stop: int):
        """(index, image) from ``start`` to ``stop`` inclusive, forwards or backwards."""
        step = 1 if stop >= start else -1
        for f in range(start, stop + step, step):
            yield f, self.frame(f)
