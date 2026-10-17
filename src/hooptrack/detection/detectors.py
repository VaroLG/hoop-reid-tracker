"""Detectors: anything that maps a frame to (boxes TLBR, scores).

Three interchangeable implementations behind one tiny interface:

* :class:`YoloDetector`   — Ultralytics YOLO, the real thing.
* :class:`CachedDetector` — replays detections saved to disk. Detection is the
  expensive step; caching it once lets you iterate on the tracker in seconds.
* :class:`OracleDetector` — ground-truth boxes (optionally degraded). Used in
  the ablation to answer "how good is my ASSOCIATION if detection were
  perfect?" — it separates tracker errors from detector errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.sportsmot import Sequence


class YoloDetector:
    def __init__(
        self,
        weights: str = "yolo11m.pt",
        conf: float = 0.1,
        iou: float = 0.7,
        imgsz: int = 1280,
        device: str | None = None,
        classes: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO  # lazy: keeps the core importable without torch

        self.model = YOLO(weights)
        # COCO-pretrained models: class 0 = person. A model fine-tuned on
        # SportsMOT has a single "player" class, also index 0.
        self.classes = [0] if classes is None else classes
        self.kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

    def __call__(self, frame: np.ndarray, **_) -> tuple[np.ndarray, np.ndarray]:
        res = self.model.predict(frame, classes=self.classes, **self.kwargs)[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(float)
        scores = res.boxes.conf.cpu().numpy().astype(float)
        return boxes, scores


class CachedDetector:
    """Reads ``<cache_dir>/<sequence>.txt`` in MOT-det format (frame,-1,x,y,w,h,score)."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._seq: str | None = None

    def bind(self, seq: Sequence) -> None:
        path = self.cache_dir / f"{seq.name}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"No cached detections at {path}. Run `hooptrack detect` first.")
        data = np.loadtxt(path, delimiter=",", ndmin=2)
        self._cache = {}
        for f in np.unique(data[:, 0]).astype(int):
            rows = data[data[:, 0] == f]
            tlbr = np.c_[rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]]
            self._cache[int(f)] = (tlbr, rows[:, 6])
        self._seq = seq.name

    def __call__(self, frame: np.ndarray | None = None, frame_idx: int = 0, **_):
        return self._cache.get(frame_idx, (np.zeros((0, 4)), np.zeros(0)))


class OracleDetector:
    """Ground-truth boxes with optional jitter and dropout."""

    def __init__(self, box_noise: float = 0.0, drop_rate: float = 0.0, seed: int = 0) -> None:
        self.box_noise, self.drop_rate = box_noise, drop_rate
        self.rng = np.random.default_rng(seed)
        self._gt: dict = {}

    def bind(self, seq: Sequence) -> None:
        self._gt = seq.load_gt()

    def __call__(self, frame: np.ndarray | None = None, frame_idx: int = 0, **_):
        _, boxes = self._gt.get(frame_idx, (None, np.zeros((0, 4))))
        keep = self.rng.random(len(boxes)) >= self.drop_rate
        boxes = boxes[keep] + self.rng.normal(0, self.box_noise, size=(keep.sum(), 4))
        return boxes, np.full(len(boxes), 0.9)


def save_detections(path: str | Path, rows: list[tuple[int, np.ndarray, np.ndarray]]) -> None:
    """Write (frame, boxes, scores) tuples in MOT-det format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for frame, boxes, scores in rows:
            for (x1, y1, x2, y2), s in zip(boxes, scores, strict=True):
                f.write(f"{frame},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{s:.4f}\n")


def build_detector(cfg: dict):
    kind = cfg.get("type", "yolo")
    if kind == "yolo":
        return YoloDetector(
            weights=cfg.get("weights", "yolo11m.pt"), conf=cfg.get("conf", 0.1),
            iou=cfg.get("iou", 0.7), imgsz=cfg.get("imgsz", 1280), device=cfg.get("device"),
        )
    if kind == "cached":
        return CachedDetector(cfg.get("cache_dir", "outputs/detections"))
    if kind == "gt":
        return OracleDetector(cfg.get("box_noise", 0.0), cfg.get("drop_rate", 0.0))
    raise ValueError(f"Unknown detector type: {kind}")
