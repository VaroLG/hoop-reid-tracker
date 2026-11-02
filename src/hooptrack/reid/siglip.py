"""SigLIP / CLIP image embeddings of crops (020: team labels; 027-028: as a ReID model).

SigLIP (Zhai et al. 2023) is a CLIP-like image-text model trained on billions of web images. Its image
tower describes a crop with general visual concepts (colours, patterns, logos, "a white jersey with blue
trim"), with no person re-identification training at all. That is the point here: to tell the two KITS
apart we want a descriptor of the kit, not of the player, and one that was never fitted to our clips.
The "SigLIP + UMAP + K-means" recipe for team clustering comes from the sports-analytics community
(e.g. Roboflow's ``sports`` examples), where the crop is the central part of the torso.

The weights are read from a local folder (``hf download google/siglip-base-patch16-224 --local-dir
weights/siglip-base-patch16-224``); nothing is downloaded at run time. Needs ``transformers``.

``__call__(frame, boxes) -> (N, D)`` L2-normalised, like the ReID extractors.

Two ways to prepare a crop (``input_size``)
-------------------------------------------
* ``None`` (020, team labels): the model's own image processor. SigLIP's resizes the crop to a 224 x 224
  square (the crop is squeezed). Kept as it was, so 020 does not change.
* ``(height, width)``, default ``(256, 128)`` for ReID (``build_reid``, 027-028): our own preprocessing,
  identical in the Colab notebook (028) and here, see :func:`reid_transform`. The crop keeps its proportions.

Why not the models' processors for ReID: CLIP's resizes the SHORT side to 224 and keeps the central
224 x 224 square, so a 1:2 player loses the top and bottom quarters (head and feet: the cues that tell
team-mates apart); SigLIP's squeezes the player. A ViT-B/16 cuts the image into 16 x 16 patches and adds a
learned vector per patch POSITION, learned on a 14 x 14 grid (224 x 224). ``interpolate_pos_encoding=True``
resizes those position vectors to the 16 x 8 grid of a 256 x 128 crop. ViT ReID models do the same (e.g.
CLIP-ReID, Li et al. 2023, at 256 x 128). It is also cheaper: 128 patches instead of 196.

A fine-tuned folder (028) contains ``hooptrack_reid.json`` with the input size it was trained at, which then
overrides the default.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

REID_INPUT_SIZE = (256, 128)          # (height, width), like the OSNet crops of 012
META_FILE = "hooptrack_reid.json"     # written by the 028 notebook next to the weights


def reid_transform(input_size: tuple[int, int], mean, std):
    """THE test-time preprocessing for ReID: resize to ``input_size`` (h, w), to tensor, normalise.

    The 028 notebook uses exactly this composition for its test and its saved model, so training and the
    laptop see the same pixels. PIL input, bilinear resize with antialiasing (torchvision's default).
    """
    import torchvision.transforms as T  # noqa: N812 (torchvision convention)

    return T.Compose([T.Resize(tuple(input_size)), T.ToTensor(), T.Normalize(mean, std)])


class SigLIPExtractor:
    def __init__(self, path: str, device: str = "cpu", batch_size: int = 32,
                 input_size: tuple[int, int] | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor

        self.torch, self.device, self.batch_size = torch, device, batch_size
        self.processor = AutoImageProcessor.from_pretrained(path)
        self.model = self._load(path)
        meta = Path(path) / META_FILE
        if input_size is not None and meta.is_file():     # a fine-tuned model knows its own input size
            input_size = tuple(json.loads(meta.read_text())["input_size"])
        self.input_size = None if input_size is None else tuple(input_size)
        self.transform = (None if self.input_size is None else
                          reid_transform(self.input_size, self.processor.image_mean, self.processor.image_std))

    def _load(self, path: str):
        from transformers import SiglipVisionModel

        model = SiglipVisionModel.from_pretrained(path).eval().to(self.device)
        self.dim = model.config.hidden_size
        return model

    def _embed(self, pixel_values):
        """One batch -> (B, D) image embedding (SigLIP: the pooled output of the vision tower)."""
        return self.model(pixel_values=pixel_values, interpolate_pos_encoding=True).pooler_output

    def _pixels(self, crops: list[np.ndarray]):
        """RGB crops -> (B, 3, H, W) tensor, with the processor (020) or the ReID transform (027-028)."""
        if self.transform is None:
            return self.processor(images=crops, return_tensors="pt")["pixel_values"]
        from PIL import Image

        return self.torch.stack([self.transform(Image.fromarray(c)) for c in crops])

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, float).reshape(-1, 4)
        if not len(boxes):
            return np.zeros((0, self.dim), np.float32)
        H, W = frame.shape[:2]
        crops = []
        for x1, y1, x2, y2 in boxes:
            x1, y1 = int(max(0, x1)), int(max(0, y1))
            x2, y2 = int(min(W, x2)), int(min(H, y2))
            crop = frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else np.zeros((8, 8, 3), np.uint8)
            crops.append(np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(crops), self.batch_size):
                x = self._pixels(crops[i: i + self.batch_size])
                e = self._embed(x.to(self.device))
                out.append(self.torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


class CLIPExtractor(SigLIPExtractor):
    """027: OpenAI CLIP image tower (e.g. ``openai/clip-vit-base-patch16``), projected embedding (512-d).

    Weights from a local folder: ``hf download openai/clip-vit-base-patch16 --local-dir weights/clip-vit-base-patch16``.
    """

    def _load(self, path: str):
        from transformers import CLIPVisionModelWithProjection

        model = CLIPVisionModelWithProjection.from_pretrained(path).eval().to(self.device)
        self.dim = model.config.projection_dim
        return model

    def _embed(self, pixel_values):
        return self.model(pixel_values=pixel_values, interpolate_pos_encoding=True).image_embeds
