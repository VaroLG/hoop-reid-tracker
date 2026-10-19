"""Reading and cleaning the public court-keypoint dataset (court chapter, C01-C02).

Shared by ``scripts/court/audit_dataset.py`` (C01: measure the problems) and
``scripts/court/convert_to_yolo.py`` (C02: fix them), so both use exactly the
same definitions of "game", "clamped label" and "split".
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .template import CourtTemplate

SPLITS = ("train", "valid", "test")


@dataclass
class Record:
    """One labelled image of the Roboflow COCO export."""

    file: str                 # file name inside <root>/<split>/
    split: str                # original Roboflow split
    game: str                 # e.g. "denver-nuggets-los-angeles-clippers-game-1"
    size: tuple[int, int]     # (width, height) as stored (576 x 576, stretched)
    bbox: np.ndarray          # COCO [x, y, w, h] of the "court" object
    kp: np.ndarray            # (K, 3) x, y, visibility (0 = not labelled, 2 = visible)

    def path(self, root: Path) -> Path:
        return root / self.split / self.file


def game_of(file_name: str) -> str:
    """'denver-nuggets-...-game-1-q1-06_17-06_12_mp4-000...jpg' -> 'denver-nuggets-...-game-1'."""
    return re.split(r"-q\d", file_name)[0]


def load_coco(root: Path) -> tuple[list[str], list[Record], list[list[int]]]:
    """Keypoint names, all records of the three splits, and the dataset's flip pairs (1-based)."""
    names, flip_pairs, recs = None, [], []
    for split in SPLITS:
        d = json.loads((root / split / "_annotations.coco.json").read_text())
        cat = next(c for c in d["categories"] if c.get("keypoints"))
        names, flip_pairs = cat["keypoints"], cat.get("flip_pairs_h", [])
        images = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            im = images[a["image_id"]]
            recs.append(Record(
                file=im["file_name"], split=split, game=game_of(im["file_name"]),
                size=(im["width"], im["height"]), bbox=np.array(a["bbox"], dtype=float),
                kp=np.array(a["keypoints"], dtype=float).reshape(-1, 3),
            ))
    return names, recs, flip_pairs


def fit_homography(rec: Record, names: list[str], template: CourtTemplate, ransac_px: float = 8.0):
    """Court -> image homography from the floor labels of one image (None if < 4 points)."""
    labelled = [n for n, v in zip(names, rec.kp[:, 2] > 0, strict=True) if v]
    floor, court_xy = template.floor_points(labelled)
    if len(floor) < 4:
        return None
    img_xy = np.array([rec.kp[names.index(n), :2] for n in floor])
    H, _ = cv2.findHomography(court_xy, img_xy, cv2.RANSAC, ransac_px)
    return H


def label_errors(rec: Record, names: list[str], template: CourtTemplate, H: np.ndarray) -> dict[str, float]:
    """Pixel distance between each label and where the homography says it should be."""
    labelled = [n for n, v in zip(names, rec.kp[:, 2] > 0, strict=True) if v]
    court_xy = np.array([template.points[n] for n in labelled])
    proj = cv2.perspectiveTransform(court_xy[None], H)[0]
    lab = np.array([rec.kp[names.index(n), :2] for n in labelled])
    return dict(zip(labelled, np.linalg.norm(proj - lab, axis=1).tolist(), strict=True))


def clamped_labels(rec: Record, names: list[str], template: CourtTemplate,
                   outlier_px: float = 15.0, ransac_px: float = 8.0) -> list[str]:
    """Floor labels that disagree with the image's own homography by > ``outlier_px``.

    C01 showed these are points hidden below the frame or behind the blacked-out
    score bar, placed on the image border instead of their true position.
    """
    H = fit_homography(rec, names, template, ransac_px)
    if H is None:
        return []
    return [n for n, e in label_errors(rec, names, template, H).items()
            if e > outlier_px and n not in template.non_floor]


def split_by_game(games, fractions=(0.7, 0.15, 0.15)) -> dict[str, list[str]]:
    """Deterministic game-level split: order games by a hash of their name, then cut.

    A hash (not random.shuffle) gives the same split on every machine and Python
    version, and adding a new game never moves the existing ones.
    """
    order = sorted(set(games), key=lambda g: hashlib.md5(g.encode()).hexdigest())
    n = len(order)
    n_tr, n_va = round(fractions[0] * n), round(fractions[1] * n)
    return {"train": order[:n_tr], "valid": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}


def flip_index(names: list[str], template: CourtTemplate) -> list[int]:
    """For each keypoint, the index of its mirror image under a horizontal image flip.

    Flipping a broadcast frame left <-> right swaps the two halves of the court
    (x -> length - x) and keeps near/far (y). Points on the halfway line map to
    themselves. Ultralytics needs this list (``flip_idx``) to flip labels correctly
    during augmentation; without it a flipped left baseline would be labelled as
    the left baseline again.
    """
    out = []
    for n in names:
        x, y = template.points[n]
        mirror = (template.length - x, y)
        j = min(range(len(names)), key=lambda k: np.hypot(*(np.subtract(template.points[names[k]], mirror))))
        if np.hypot(*(np.subtract(template.points[names[j]], mirror))) > 1e-3:
            raise ValueError(f"keypoint {n} has no mirror in the template")
        out.append(j)
    return out
