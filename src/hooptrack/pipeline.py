"""End-to-end loop for one video: frames -> detector -> ReID -> tracker -> MOT file.

Also measures the latency of every stage separately. On an edge device
(Jetson) the bottleneck is usually ReID on many crops, not the detector:
reporting the split per stage makes that visible instead of a single FPS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .data.sportsmot import Sequence, write_mot_txt
from .tracking import ReIDTracker, TrackerConfig


@dataclass
class RunTiming:
    detect: float = 0.0
    reid: float = 0.0
    track: float = 0.0
    frames: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        n = max(self.frames, 1)
        total = self.detect + self.reid + self.track
        return {
            "frames": self.frames,
            "ms_detect": 1000 * self.detect / n,
            "ms_reid": 1000 * self.reid / n,
            "ms_track": 1000 * self.track / n,
            "fps_total": n / total if total > 0 else float("nan"),
            **self.extra,
        }


def run_sequence(
    seq: Sequence,
    detector,
    reid,
    tracker_cfg: TrackerConfig,
    out_txt: str | Path,
    video_path: str | Path | None = None,
    show_progress: bool = True,
) -> RunTiming:
    """Track one sequence and write ``out_txt`` in MOT format."""
    # Pass the real frame rate: stage 4 converts "frames lost" into seconds.
    tracker = ReIDTracker(TrackerConfig(**{**tracker_cfg.__dict__, "fps": seq.fps}))
    if hasattr(detector, "bind"):
        detector.bind(seq)  # cached / oracle detectors need to know the sequence

    writer = None
    if video_path is not None:
        from .viz import draw_tracks

        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (seq.width, seq.height)
        )

    timing = RunTiming()
    rows: list[tuple[int, int, np.ndarray, float]] = []
    revivals = 0

    iterator = seq.frames()
    if show_progress:
        iterator = tqdm(iterator, total=seq.length, desc=seq.name, leave=False)

    for frame_idx, path in iterator:
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)

        t0 = time.perf_counter()
        boxes, scores = detector(frame, frame_idx=frame_idx)
        t1 = time.perf_counter()
        feats = reid(frame, boxes) if reid is not None else None
        t2 = time.perf_counter()
        outputs = tracker.update(boxes, scores, feats, frame)
        t3 = time.perf_counter()

        timing.detect += t1 - t0
        timing.reid += t2 - t1
        timing.track += t3 - t2
        timing.frames += 1

        for o in outputs:
            rows.append((frame_idx, o.track_id, o.tlbr, o.score))
            revivals += int(o.revived)
        if writer is not None:
            writer.write(draw_tracks(frame, outputs, frame_idx))

    if writer is not None:
        writer.release()
    write_mot_txt(out_txt, rows)
    timing.extra = {"ids_created": tracker._next_id, "memory_revivals": revivals}
    return timing
