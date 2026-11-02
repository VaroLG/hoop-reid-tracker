"""Reuse ReID embeddings computed in an earlier run (experiment 026).

Why: with cached detections, ReID is by far the slowest part of a tracking run on CPU (OSNet on every box of
every frame). Experiments that only change the TRACKER re-compute exactly the same embeddings each time.
``scripts/run_deep_eiou.py`` (024) already saved the OSNet embedding of every cached detection, so a tracker
ablation can read them instead of running the network again.

How a box is found again: the pipeline gives the extractor only ``(frame, boxes)``, without the clip name or the
frame number. But the boxes come from the same detection cache file, parsed the same way (``np.loadtxt``, then
``x + w``, ``y + h``), so their coordinates are bit-identical to the ones the embeddings were saved for. The key is
the box rounded to 0.01 px. A box that is not in the cache (or whose key is ambiguous: two different embeddings
for the same coordinates) is sent to the real extractor, so the result never depends on the cache being complete.

Config (``reid:`` section)::

    embedding_cache: outputs/runs/024_deep_eiou/embeddings   # <clip>.npz, one array per frame (file order)
    embedding_cache_dets: outputs/detections/yolo26m_player  # the detection files those arrays follow

The cache must come from the SAME model and weights as ``reid.type`` / ``reid.weights``: it is not checked here
(024's embeddings are OSNet ``weights/osnet_x1_0_sportsmot.pth``, L2-normalised like the extractor's output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _key(box) -> tuple:
    return tuple(np.round(np.asarray(box, dtype=float), 2).tolist())


class CachedReID:
    """Wrap an extractor: answer from saved embeddings when the box is known, else compute."""

    def __init__(self, base, emb_dir: str | Path, det_dir: str | Path) -> None:
        self.base = base
        self.hits = 0
        self.misses = 0
        self._table: dict[tuple, np.ndarray] = {}
        ambiguous: set[tuple] = set()
        for npz in sorted(Path(emb_dir).glob("*.npz")):
            det_file = Path(det_dir) / f"{npz.stem}.txt"
            if not det_file.is_file():
                continue
            raw = np.loadtxt(det_file, delimiter=",", ndmin=2)      # same parsing as CachedDetector
            z = np.load(npz)
            for f in z.files:
                rows = raw[raw[:, 0] == int(f)]
                tlbr = np.c_[rows[:, 2:4], rows[:, 2:4] + rows[:, 4:6]]
                emb = z[f]
                if len(emb) != len(tlbr):                              # not the same detections: skip frame
                    continue
                for box, e in zip(tlbr, emb, strict=True):
                    k = _key(box)
                    if k in self._table and not np.allclose(self._table[k], e, atol=1e-4):
                        ambiguous.add(k)
                    self._table[k] = e
        for k in ambiguous:                                            # never guess: compute those
            del self._table[k]
        if not self._table:
            raise SystemExit(f"reid.embedding_cache: no usable embeddings in {emb_dir} (detections {det_dir})")

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        found = [self._table.get(_key(b)) for b in boxes]
        missing = [i for i, e in enumerate(found) if e is None]
        self.hits += len(boxes) - len(missing)
        self.misses += len(missing)
        if missing:                                                    # the real network, only for these
            computed = self.base(frame, boxes[missing])
            for i, e in zip(missing, computed, strict=True):
                found[i] = e
        if not len(boxes):
            return self.base(frame, boxes)
        return np.stack(found).astype(np.float32)
