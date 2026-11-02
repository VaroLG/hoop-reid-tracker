"""Jersey-number reader trained on player crops, digit by digit (experiment 033).

Why digit by digit
------------------
The training data (E-BARD, NBA 2024-25 broadcast) has only 52 different numbers, and our val clips (NCAA,
European leagues) have others. A classifier with one output per number can never answer a number it did not see
in training. Jersey-number papers therefore read the number the way a person does, one digit at a time:

* ``length``: how many digits are visible: 0 (no number readable: back turned, arm in front, too small),
  1 or 2;
* ``tens``: the first digit of a two-digit number (0-9; "00" is tens 0, units 0);
* ``units``: the last digit (0-9), shared by one- and two-digit numbers.

The probability of a number is the product of its parts::

    P(no number) = P(length = 0)
    P("7")       = P(length = 1) * P(units = 7)
    P("23")      = P(length = 2) * P(tens = 2) * P(units = 3)

so the 111 answers (no number, 0-9, 00-99) are scored from 3 small heads, and every digit is learned from every
number that contains it.

The model
---------
The image tower of a pretrained model (default: SigLIP base, web image-text pretraining, so it has seen a lot of
printed text and digits; the 028 ReID model is the same tower) on the player crop at 256 x 128 (height x width,
not deformed, position vectors interpolated as in ``hooptrack.reid.siglip``), then the three linear heads.

Reading a piece
---------------
One frame is often not enough (the number turns away, motion blur). ``read_piece`` averages the 111-way
distributions of the crops of a piece and answers the most probable number with its mean probability as
confidence; the caller decides the confidence needed (calibrated on the training data's val split).

This file is used by the Colab notebook (training: it writes an exact copy) and by the laptop (inference), so the
two cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 (PyTorch convention)
from torch import nn

INPUT_SIZE = (256, 128)                      # (height, width), as the ReID crops
NUMBERS = ["nan"] + [str(d) for d in range(10)] + [f"{t}{d}" for t in range(10) for d in range(10)]   # 111 answers


def encode(label: str) -> tuple[int, int, int]:
    """'nan' / '7' / '23' / '00' -> (length, tens, units); -100 = no target for that head (ignored by the loss)."""
    label = str(label).strip()
    if label.lower() == "nan" or not label.isdigit() or len(label) > 2:
        return 0, -100, -100
    if len(label) == 1:
        return 1, -100, int(label)
    return 2, int(label[0]), int(label[1])


def number_log_probs(length_logits: torch.Tensor, tens_logits: torch.Tensor,
                     units_logits: torch.Tensor) -> torch.Tensor:
    """(B, 3), (B, 10), (B, 10) logits -> (B, 111) log-probabilities over ``NUMBERS``."""
    ll, lt, lu = (F.log_softmax(x.float(), dim=1) for x in (length_logits, tens_logits, units_logits))
    none = ll[:, :1]
    one = ll[:, 1:2] + lu                                            # (B, 10)
    two = (ll[:, 2:3, None] + lt[:, :, None] + lu[:, None, :]).flatten(1)   # (B, 100), tens-major
    return torch.cat([none, one, two], dim=1)


class JerseyNumberModel(nn.Module):
    """Pretrained image tower + 3 digit heads."""

    def __init__(self, tower, dim: int, kind: str = "siglip") -> None:
        super().__init__()
        self.tower, self.kind = tower, kind
        self.length = nn.Linear(dim, 3)
        self.tens = nn.Linear(dim, 10)
        self.units = nn.Linear(dim, 10)

    def features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.tower(pixel_values=pixel_values, interpolate_pos_encoding=True)
        return out.pooler_output if self.kind == "siglip" else out.image_embeds

    def forward(self, pixel_values: torch.Tensor):
        f = self.features(pixel_values)
        return self.length(f), self.tens(f), self.units(f)


def loss_fn(logits, targets) -> torch.Tensor:
    """Cross-entropy of each head; a head with no target for a sample (-100) does not count."""
    (ll, lt, lu), (yl, yt, yu) = logits, targets
    loss = F.cross_entropy(ll.float(), yl)
    if (yt >= 0).any():
        loss = loss + F.cross_entropy(lt.float(), yt, ignore_index=-100)
    if (yu >= 0).any():
        loss = loss + F.cross_entropy(lu.float(), yu, ignore_index=-100)
    return loss


def load_tower(path: str, kind: str):
    from transformers import CLIPVisionModelWithProjection, SiglipVisionModel

    if kind == "siglip":
        tower = SiglipVisionModel.from_pretrained(path)
        return tower, tower.config.hidden_size
    tower = CLIPVisionModelWithProjection.from_pretrained(path)
    return tower, tower.config.projection_dim


def save(model: JerseyNumberModel, processor, out_dir: str | Path, meta: dict) -> None:
    """Folder = the tower (save_pretrained) + its processor (mean / std) + heads.pt + jersey_reader.json."""
    out_dir = Path(out_dir)
    model.tower.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("tower.")}, out_dir / "heads.pt")
    (out_dir / "jersey_reader.json").write_text(json.dumps({"kind": model.kind, "input_size": list(INPUT_SIZE),
                                                            **meta}, indent=2))


def test_transform(input_size: tuple[int, int], mean, std):
    """Resize to (h, w), to tensor, normalise: identical to ``hooptrack.reid.siglip.reid_transform`` (copied so that
    this file also works on its own, in the Colab notebook)."""
    import torchvision.transforms as T  # noqa: N812 (torchvision convention)

    return T.Compose([T.Resize(tuple(input_size)), T.ToTensor(), T.Normalize(mean, std)])


def load(path: str | Path, device: str = "cpu"):
    """-> (model in eval mode, test transform, meta)."""
    from transformers import AutoImageProcessor

    path = Path(path)
    meta = json.loads((path / "jersey_reader.json").read_text())
    tower, dim = load_tower(str(path), meta["kind"])
    model = JerseyNumberModel(tower, dim, meta["kind"])
    missing, unexpected = model.load_state_dict(torch.load(path / "heads.pt", map_location="cpu"), strict=False)
    if [k for k in missing if not k.startswith("tower.")] or unexpected:
        raise ValueError(f"heads.pt does not match the model: {missing} {unexpected}")
    proc = AutoImageProcessor.from_pretrained(str(path))
    return model.eval().to(device), test_transform(tuple(meta["input_size"]), proc.image_mean, proc.image_std), meta


def read_piece(probs: np.ndarray) -> tuple[str, float]:
    """(n_crops, 111) probabilities of the crops of one piece -> (number or 'nan', confidence)."""
    mean = np.asarray(probs, float).mean(axis=0)
    k = int(mean.argmax())
    return NUMBERS[k], float(mean[k])


class JerseyClassifierReader:
    """Laptop side: ``probs(frame_rgb, boxes) -> (N, 111)`` for the player crops in ``boxes`` (TLBR pixels).

    Same input as ``hooptrack.reid.jersey.JerseyReader`` (an RGB frame), so the stitching can use either. The crop is
    the whole player box, as in training (E-BARD stubs are whole-player crops). ``min_conf`` comes with the model:
    the confidence at which 95 % of the answers were right on the training data's val split.
    """

    def __init__(self, path: str | Path, device: str = "cpu", batch_size: int = 32) -> None:
        self.model, self.transform, self.meta = load(path, device)
        self.device, self.batch_size = device, batch_size
        self.min_conf = float(self.meta.get("min_conf", 0.5))

    def probs(self, frame_rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        from PIL import Image

        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros((0, len(NUMBERS)), np.float32)
        H, W = frame_rgb.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2))
            c = frame_rgb[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((8, 4, 3), np.uint8)
            crops.append(self.transform(Image.fromarray(np.ascontiguousarray(c))))
        out = []
        with torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                x = torch.stack(crops[i:i + self.batch_size]).to(self.device)
                out.append(number_log_probs(*self.model(x)).exp().cpu().numpy())
        return np.concatenate(out).astype(np.float32)
