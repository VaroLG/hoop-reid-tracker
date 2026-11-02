"""Jersey numbers read zero-shot by Florence-2 (experiment 023).

Why
---
After 020, what is left is the team-mate problem: the connector joins a piece to the wrong TEAM-MATE, and appearance
models cannot tell team-mates apart (016-018, AUC ~0.71). The jersey number is the one cue that is different for every
player of a team. 022 showed that a generic scene-text recogniser (PARSeq) cannot read it. Florence-2 (Xiao et al.
2024, a small vision-language model trained on a huge set of image tasks, OCR among them) can, with no training on
our data. Measured on the train clips (023 README): with confidence >= 0.7, 15 % of upper-body crops are read, 44 %
of the players get a number, reads along one player agree 96 % of the time, and 36 of 40 reads checked by eye are right.

How
---
* Crop: upper body (5-55 % of the box height, full width), where the number is, front or back.
* Task ``<OCR>``, greedy decoding, at most 6 new tokens. A read counts only when the whole answer is 1-2 digits.
* Confidence = product of the probabilities of the generated tokens.

It is slow on CPU (~3 s per crop on 2 cores), so it runs offline only, on a few crops per tracklet.

``__call__(frame_rgb, boxes) -> list[(text, confidence)]``. Weights from a local folder:
``hf download florence-community/Florence-2-base --local-dir weights/florence-2-base`` (needs ``transformers`` >= 5).
"""

from __future__ import annotations

import re

import numpy as np

DIGITS = re.compile(r"\d{1,2}")


class JerseyReader:
    def __init__(self, path: str = "weights/florence-2-base", device: str = "cpu",
                 top: float = 0.05, bottom: float = 0.55) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device, self.top, self.bottom = torch, device, top, bottom
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = AutoModelForImageTextToText.from_pretrained(path).eval().to(device)

    def __call__(self, frame_rgb: np.ndarray, boxes: np.ndarray) -> list[tuple[str, float]]:
        from PIL import Image

        H, W = frame_rgb.shape[:2]
        ims, ok = [], []
        for x1, y1, x2, y2 in np.asarray(boxes, float).reshape(-1, 4):
            h = y2 - y1
            c = frame_rgb[int(max(0, y1 + self.top * h)): int(min(H, y1 + self.bottom * h)),
                          int(max(0, x1)): int(min(W, x2))]
            ok.append(c.size > 0)
            if c.size:
                ims.append(Image.fromarray(np.ascontiguousarray(c)))
        out: list[tuple[str, float]] = [("", 0.0)] * len(ok)
        if not ims:
            return out
        x = self.processor(text=["<OCR>"] * len(ims), images=ims, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            gen = self.model.generate(**x, max_new_tokens=6, num_beams=1, do_sample=False,
                                      return_dict_in_generate=True, output_scores=True)
        logp = self.model.compute_transition_scores(gen.sequences, gen.scores, normalize_logits=True)
        texts = self.processor.batch_decode(gen.sequences, skip_special_tokens=True)
        reads = []
        for t, lp in zip(texts, logp, strict=True):
            lp = lp[self.torch.isfinite(lp)]
            reads.append((t.strip(), float(lp.sum().exp())))
        it = iter(reads)
        return [next(it) if o else ("", 0.0) for o in ok]


def tracklet_number(reads: list[tuple[str, float]], min_conf: float = 0.7, min_reads: int = 2,
                    majority: float = 2 / 3) -> str | None:
    """The number of a tracklet from its reads, or None when it is not clear.

    Only reads that are 1-2 digits with confidence >= ``min_conf`` vote. At least ``min_reads`` of them, and the most
    common number must hold ``majority`` of the votes (train: reads of one player agree 96 % of the time).
    """
    votes = [t for t, c in reads if c >= min_conf and DIGITS.fullmatch(t)]
    if len(votes) < min_reads:
        return None
    best = max(set(votes), key=votes.count)
    return best if votes.count(best) >= majority * len(votes) else None
