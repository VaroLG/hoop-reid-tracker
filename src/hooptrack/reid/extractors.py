"""Appearance (ReID) feature extractors.

A ReID model maps a person crop to a vector such that crops of the SAME person
are close (small cosine distance) and crops of different people are far.

Implemented, from weakest to strongest:

* :class:`HistogramExtractor` — HSV colour histograms of the upper and lower
  body. No deep learning; it mostly encodes the jersey colour, so it separates
  TEAMS well but team-mates poorly. A useful, honest baseline.
* :class:`ResNetExtractor`    — ImageNet ResNet-50 global features. Generic,
  no person-specific training; better than colour, worse than a ReID model.
* :class:`OSNetExtractor`     — OSNet trained for person re-identification
  (torchreid). Learns omni-scale cues (number, shoes, hair, body shape) and
  is the default for serious runs. Fine-tune it on SportsMOT crops with
  ``scripts/export_reid_crops.py`` for the best results.

All extractors share ``__call__(frame, boxes) -> (N, D) float32``.
"""

from __future__ import annotations

import cv2
import numpy as np

# Standard ReID input resolution (height x width): tall and narrow like a person.
REID_H, REID_W = 256, 128


def crop_boxes(frame: np.ndarray, boxes: np.ndarray, size=(REID_W, REID_H)) -> list[np.ndarray]:
    """Cut and resize every box. Boxes are clipped to the image first."""
    h, w = frame.shape[:2]
    crops = []
    for x1, y1, x2, y2 in np.asarray(boxes, dtype=float):
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w, x2)), int(min(h, y2))
        if x2 <= x1 or y2 <= y1:  # degenerate box (fully outside the image)
            crops.append(np.zeros((size[1], size[0], 3), dtype=np.uint8))
            continue
        crops.append(cv2.resize(frame[y1:y2, x1:x2], size))
    return crops


class HistogramExtractor:
    """Concatenated HSV histograms of the torso (jersey) and legs (shorts)."""

    def __init__(self, bins=(16, 8, 4)) -> None:
        self.bins = bins

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 2 * int(np.prod(self.bins))), dtype=np.float32)
        feats = []
        for crop in crop_boxes(frame, boxes):
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Central 60 % horizontally: drop background pixels at the sides.
            hsv = hsv[:, int(0.2 * REID_W) : int(0.8 * REID_W)]
            torso = hsv[int(0.15 * REID_H) : int(0.55 * REID_H)]
            legs = hsv[int(0.55 * REID_H) : int(0.85 * REID_H)]
            parts = []
            for part in (torso, legs):
                hist = cv2.calcHist([part], [0, 1, 2], None, list(self.bins), [0, 180, 0, 256, 0, 256])
                parts.append(hist.flatten())
            v = np.concatenate(parts)
            feats.append(v / (np.linalg.norm(v) + 1e-12))
        return np.asarray(feats, dtype=np.float32)


class _TorchExtractor:
    """Shared batching / normalisation logic for the deep extractors."""

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = None  # set by subclasses
        self.dim = 0

    def _to_tensor(self, crops: list[np.ndarray]):
        batch = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32) / 255.0  # BGR->RGB
        batch = (batch - self.mean) / self.std
        return self.torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        crops = crop_boxes(frame, boxes)
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                feats = self.model(self._to_tensor(crops[i : i + self.batch_size]))
                feats = self.torch.nn.functional.normalize(feats.flatten(1), dim=1)
                out.append(feats.cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class ResNetExtractor(_TorchExtractor):
    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        super().__init__(device, batch_size)
        import torchvision

        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        model.fc = self.torch.nn.Identity()  # drop the classifier: keep the 2048-D embedding
        self.model = model.eval().to(self.device)
        self.dim = 2048


class OSNetExtractor(_TorchExtractor):
    def __init__(self, weights: str | None = None, device: str | None = None, batch_size: int = 64) -> None:
        super().__init__(device, batch_size)
        try:  # the import path moved between torchreid releases
            from torchreid.reid.utils import FeatureExtractor
        except ImportError:
            try:
                from torchreid.utils import FeatureExtractor
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "OSNet needs torchreid: "
                    "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
                ) from e
        # FeatureExtractor downloads ImageNet/Market-1501 weights when model_path is empty.
        self._fe = FeatureExtractor(model_name="osnet_x1_0", model_path=weights or "", device=self.device)
        self.model = self._fe.model
        self.dim = 512


def build_reid(cfg: dict):
    kind = cfg.get("type", "resnet")
    if kind in (None, "none"):
        return None
    if kind == "histogram":
        return HistogramExtractor()
    if kind == "resnet":
        return ResNetExtractor(cfg.get("device"), cfg.get("batch_size", 64))
    if kind == "osnet":
        return OSNetExtractor(cfg.get("weights"), cfg.get("device"), cfg.get("batch_size", 64))
    raise ValueError(f"Unknown reid type: {kind}")
