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
    filter: float = 0.0
    reid: float = 0.0
    track: float = 0.0
    frames: int = 0
    extra: dict = field(default_factory=dict)

    def summary(self) -> dict:
        n = max(self.frames, 1)
        total = self.detect + self.filter + self.reid + self.track
        return {
            "frames": self.frames,
            "ms_detect": 1000 * self.detect / n,
            "ms_filter": 1000 * self.filter / n,
            "ms_reid": 1000 * self.reid / n,
            "ms_track": 1000 * self.track / n,
            "fps_total": n / total if total > 0 else float("nan"),
            **self.extra,
        }


def embed_needed(reid, frame, boxes, scores, cfg: TrackerConfig) -> np.ndarray:
    """Run ReID only on the detections whose embedding the tracker can use.

    Experiment 002 (``reid.only_usable_boxes: true``).

    Low-score boxes (below ``high_thresh``) are only ever matched by IoU in
    stage 2, and they are too weak to update an appearance model (below
    ``feature_update_min_score``). Embedding them is wasted work: with a COCO
    detector that keeps everything above conf=0.1, they are ~70 % of the boxes
    and ReID is by far the slowest stage on CPU.

    Rows for skipped boxes are zeros; the tracker never reads them.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    cutoff = min(cfg.high_thresh, cfg.feature_update_min_score)
    need = np.where(scores >= cutoff)[0]
    sub = reid(frame, boxes[need])
    feats = np.zeros((len(boxes), sub.shape[1] if sub.ndim == 2 else 0), dtype=np.float32)
    if len(need) and feats.shape[1]:
        feats[need] = sub
    return feats


def _retro_label_other(seq, rows, tracker, team_classifier, samples: int = 10) -> set[int]:
    """007c: ids that never got a team vote (e.g. they left before the colour model was
    fitted) are labelled now with the final model, on a few of their frames."""
    import collections

    voted = {i for i, t in tracker.final_team.items() if t >= 0}
    by_id: dict[int, list] = collections.defaultdict(list)
    by_frame: dict[int, list] = collections.defaultdict(list)
    for f, i, box, _ in rows:
        by_frame[f].append(box)
        if i not in voted:
            by_id[i].append((f, box))
    paths = dict(seq.frames())
    found = set()
    for i, seen in by_id.items():
        pick = [seen[k] for k in np.linspace(0, len(seen) - 1, min(samples, len(seen))).astype(int)]
        votes = np.zeros(3, dtype=int)
        for f, box in pick:
            others = [b for b in by_frame[f] if not np.allclose(b, box)]
            lab = team_classifier.classify(cv2.imread(str(paths[f])), np.asarray(box)[None], np.asarray(others))[0]
            if lab >= 0:
                votes[lab] += 1
        n = votes.sum()
        if n >= tracker.cfg.team_min_votes and votes[2] >= tracker.cfg.team_min_share * n:
            found.add(i)
    return found


def _render_offline_video(seq, video_path, drawn: dict, hidden_ids: set) -> None:
    """Second pass over the frames: draw the stored outputs minus the hidden ids."""
    from .viz import draw_tracks

    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, (seq.width, seq.height))
    for frame_idx, path in seq.frames():
        outputs, hulls = drawn.get(frame_idx, ([], []))
        shown = [o for o in outputs if o.track_id not in hidden_ids]
        writer.write(draw_tracks(cv2.imread(str(path)), shown, frame_idx, polygons=hulls))
    writer.release()


def run_sequence(
    seq: Sequence,
    detector,
    reid,
    tracker_cfg: TrackerConfig,
    out_txt: str | Path,
    video_path: str | Path | None = None,
    show_progress: bool = True,
    filters: list | None = None,
    only_usable_boxes: bool = False,
    team_classifier=None,
    offline_suppress_other: bool = False,
) -> RunTiming:
    """Track one sequence and write ``out_txt`` in MOT format.

    Args:
        filters: post-detection filters (experiment 003, e.g. ``CourtMask``),
            applied in order before ReID. ``None``/empty = no filtering.
        only_usable_boxes: experiment 002 — embed only the boxes the tracker
            can use. False reproduces experiment 001 exactly.
        team_classifier: experiment 006 — labels each box team A / B / other.
            None = no team logic (behaviour up to 005).
        offline_suppress_other: experiment 007c — after the whole video, drop EVERY
            row of an id whose final vote is "other", including the frames before
            the vote was known (warm-up, first frames of a referee). Needs the whole
            video, so it is an offline mode; the video overlay stays online.
    """
    # Pass the real frame rate: stage 4 converts "frames lost" into seconds.
    tracker = ReIDTracker(TrackerConfig(**{**tracker_cfg.__dict__, "fps": seq.fps}))
    if hasattr(detector, "bind"):
        detector.bind(seq)  # cached / oracle detectors need to know the sequence
    filters = list(filters or [])
    if team_classifier is not None:
        team_classifier.reset()
    for flt in filters:
        if hasattr(flt, "reset"):
            flt.reset()  # per-video state (e.g. last good court hull)

    writer = None
    # 007c offline mode: the video is rendered AFTER the run, so it shows the same
    # filtered tracks as the results file. Online mode draws frame by frame.
    deferred_video = video_path is not None and offline_suppress_other and team_classifier is not None
    drawn: dict[int, tuple[list, list]] = {}   # frame -> (outputs, hulls), deferred mode only
    if video_path is not None and not deferred_video:
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
        tf = time.perf_counter()
        for flt in filters:
            boxes, scores = flt(frame, boxes, scores)
        t1 = time.perf_counter()
        if reid is None:
            feats = None
        elif only_usable_boxes:
            feats = embed_needed(reid, frame, boxes, scores, tracker.cfg)
        else:
            feats = reid(frame, boxes)
        t2 = time.perf_counter()
        teams = team_classifier(frame, boxes, scores) if team_classifier is not None else None
        if teams is not None:                      # 008: players teach the court mask where they stand
            players = boxes[(teams == 0) | (teams == 1)]
            for flt in filters:
                if hasattr(flt, "observe_players"):
                    flt.observe_players(players)
        outputs = tracker.update(boxes, scores, feats, frame, teams=teams)
        t3 = time.perf_counter()

        timing.detect += tf - t0
        timing.filter += t1 - tf
        timing.reid += t2 - t1
        timing.track += t3 - t2
        timing.frames += 1

        for o in outputs:
            rows.append((frame_idx, o.track_id, o.tlbr, o.score))
            revivals += int(o.revived)
        if writer is not None or deferred_video:
            hulls = [h for h in (getattr(f, "last_hull", None) for f in filters) if h is not None]
            if deferred_video:
                drawn[frame_idx] = (outputs, hulls)
            else:
                writer.write(draw_tracks(frame, outputs, frame_idx, polygons=hulls))

    if writer is not None:
        writer.release()
    dropped = 0
    if offline_suppress_other and team_classifier is not None:
        other_ids = {i for i, team in tracker.final_team.items() if team == 2}
        other_ids |= _retro_label_other(seq, rows, tracker, team_classifier)
        kept = [r for r in rows if r[1] not in other_ids]
        dropped, rows = len(rows) - len(kept), kept
        if deferred_video:
            _render_offline_video(seq, video_path, drawn, other_ids)
    write_mot_txt(out_txt, rows)
    timing.extra = {"ids_created": tracker._next_id, "memory_revivals": revivals}
    if offline_suppress_other:
        timing.extra["offline_dropped_rows"] = dropped
    if team_classifier is not None:
        timing.extra.update(team_classifier.stats())
    if filters:
        for flt in filters:
            if hasattr(flt, "stats"):
                timing.extra.update(flt.stats())
    return timing
