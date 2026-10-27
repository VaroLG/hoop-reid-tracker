"""Part-based OSNet: one appearance vector per body region (experiment 018).

Why parts
---------
The ReID models so far (012, 017) squeeze the whole crop into ONE vector by
averaging the last feature map over every position (global average pooling).
Team-mates wear the same kit, so the kit dominates that average. What tells them
apart is small and local: shoes and socks at the bottom, hair and skin at the top,
knee pads, sleeves. In a global average these details are a few positions out of 128,
and they get diluted. The team-mate benchmark (016/017) shows it: AUC same player vs
team-mate stays at 0.71 whatever the training data.

PCB (Part-based Convolutional Baseline, Sun et al., ECCV 2018) keeps the regions
apart. The feature map is cut into horizontal stripes (here 4: head and shoulders,
torso, hips and thighs, legs and feet), each stripe is averaged on its own, and
each stripe vector has to recognise the player by itself during training. A stripe
cannot lean on the jersey if it does not contain it, so the legs stripe learns
shoes and socks, and the head stripe learns hair and skin.

The model
---------
::

    crop (H x W)
      -> OSNet backbone (ImageNet weights) -> feature map 512 x H/16 x W/16
      -> global average pool ......................... g   (512)   the whole player, as before
      -> average pool per horizontal stripe (P of them) p_1 .. p_P (512 each)
      -> one BatchNorm "neck" per branch (BNNeck, Luo et al. 2019)
      -> training:  one identity classifier per branch (cross-entropy on each)
      -> inference: concatenate the P + 1 vectors, each L2-normalised, and
                    scale so the result has norm 1

With every branch normalised and the whole vector scaled by 1/sqrt(P + 1), the cosine
similarity of two embeddings is exactly the MEAN of the per-branch cosine
similarities: every body region gets one vote, and the jersey gets no more than
one. So the rest of the pipeline (tracker, stitching, benchmark) can use the
vector exactly like the old one: same cosine distance, only a larger dimension.

The same file is used by the Colab notebook (training) and by
``hooptrack.reid.extractors.OSNetPartsExtractor`` (inference), so the two
cannot drift apart.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 (PyTorch convention)
from torch import nn


class OSNetParts(nn.Module):
    """OSNet-x1.0 backbone + global branch + ``parts`` horizontal stripe branches."""

    def __init__(self, num_classes: int = 1, parts: int = 4, pretrained: bool = True) -> None:
        super().__init__()
        from torchreid.reid import models as reid_models  # OSNet architecture (+ ImageNet weights)

        self.parts = parts
        self.backbone = reid_models.build_model("osnet_x1_0", num_classes=1, pretrained=pretrained)
        dim = 512                                           # OSNet's last feature map has 512 channels
        n = parts + 1                                       # global branch + stripes
        # BNNeck: the triplet loss works on the vector BEFORE the BatchNorm, the classifier and the
        # inference embedding AFTER it. The BN bias is frozen (standard BNNeck), so it only rescales.
        self.necks = nn.ModuleList(nn.BatchNorm1d(dim) for _ in range(n))
        for bn in self.necks:
            bn.bias.requires_grad_(False)
        self.classifiers = nn.ModuleList(nn.Linear(dim, num_classes, bias=False) for _ in range(n))

    def branches(self, x: torch.Tensor) -> list[torch.Tensor]:
        """[global, stripe_1 (top), ..., stripe_P (bottom)], each (B, 512), before the BN neck."""
        fmap = self.backbone.featuremaps(x)                 # (B, 512, h, w)
        g = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        stripes = F.adaptive_avg_pool2d(fmap, (self.parts, 1)).squeeze(-1)   # (B, 512, P)
        return [g] + [stripes[:, :, k] for k in range(self.parts)]

    def forward(self, x: torch.Tensor):
        feats = self.branches(x)
        normed = [bn(f) for bn, f in zip(self.necks, feats, strict=True)]
        if self.training:
            logits = [clf(f) for clf, f in zip(self.classifiers, normed, strict=True)]
            return logits, feats
        return embed_from_branches(normed)


def embed_from_branches(branches: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate L2-normalised branch vectors, scaled to unit norm overall.

    cos(a, b) of two such embeddings = mean over branches of cos(a_k, b_k).
    """
    n = len(branches)
    return torch.cat([F.normalize(b, dim=1) for b in branches], dim=1) / n ** 0.5


def save_checkpoint(model: OSNetParts, path, input_size: tuple[int, int]) -> None:
    """The file the tracker loads: weights + everything needed to rebuild the model (no classifier)."""
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("classifiers.")}
    torch.save({"arch": "osnet_parts", "parts": model.parts, "input_size": list(input_size),
                "state_dict": state}, path)


def load_checkpoint(path, device: str = "cpu") -> tuple[OSNetParts, tuple[int, int]]:
    """Rebuild the model for inference. Returns (model in eval mode, (height, width) of the input)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("arch") != "osnet_parts":
        raise ValueError(f"{path} is not an osnet_parts checkpoint")
    model = OSNetParts(num_classes=1, parts=int(ck["parts"]), pretrained=False)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    missing = [k for k in missing if not k.startswith("classifiers.")]
    if missing or unexpected:
        raise ValueError(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    return model.eval().to(device), tuple(ck["input_size"])
