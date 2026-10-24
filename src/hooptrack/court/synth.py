"""Synthetic broadcast frames of a basketball court, with an exact court mask (court chapter, C06).

Why synthetic data
------------------
Every method so far learned (or assumed) what ONE kind of court looks like:
the colour mask assumes "the court is the colour of the floor", C02 learned the
2025 NBA broadcast style, C05 assumed thin dark lines. New arenas break all three
(experiment 009). Real labelled frames from many arenas are scarce, so the model
also sees thousands of *invented* courts where everything that should NOT matter
is random — floor colour, painted lane and apron, logos, line colour, camera,
crowd, people, score bar — and the only constant is the geometry: the playing
area is the rectangle bounded by the lines, with the official markings inside.
This is *domain randomisation*: the model cannot memorise colours, so it has to
learn the shape.

The court is painted top-down on a canvas (``PX_PER_M`` pixels per metre) and
warped into the image with a random broadcast camera (``camera.Camera``), so the
label (``masks.court_mask``) is exact by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .camera import Camera
from .leagues import SPECS, CourtSpec, lines
from .masks import court_mask

PX_PER_M = 40          # canvas resolution: 4 cm per pixel, finer than any broadcast pixel on the floor
APRON_M = 4.0          # floor painted around the court (out-of-bounds area), metres


@dataclass
class SynthSample:
    image: np.ndarray          # BGR, (h, w, 3)
    mask: np.ndarray           # uint8 (h, w), 1 = court area
    H: np.ndarray              # court (m) -> image (px)
    spec: str
    camera: Camera


def _hsv(rng, h, s, v) -> tuple[int, int, int]:
    """A BGR colour from HSV ranges (OpenCV hue 0-179)."""
    c = np.uint8([[[rng.uniform(*h) % 180, rng.uniform(*s), rng.uniform(*v)]]])
    return tuple(int(x) for x in cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0, 0])


def _random_colour(rng) -> tuple[int, int, int]:
    return _hsv(rng, (0, 180), (40, 255), (40, 240))


def _wood(rng) -> tuple[int, int, int]:
    return _hsv(rng, (8, 28), (50, 170), (120, 235))


def paint_court(rng: np.random.Generator, spec: CourtSpec) -> np.ndarray:
    """Top-down canvas of the court + apron. Court (0, 0) is at canvas pixel (APRON_M, APRON_M) * PX_PER_M."""
    s = PX_PER_M
    W = int((spec.length + 2 * APRON_M) * s)
    Hc = int((spec.width + 2 * APRON_M) * s)

    def px(p):
        return ((np.asarray(p, float) + APRON_M) * s).astype(np.int32)

    floor = _wood(rng) if rng.random() < 0.75 else _random_colour(rng)
    canvas = np.empty((Hc, W, 3), np.uint8)
    canvas[:] = floor
    # wood planks along the court length + fine grain noise
    if rng.random() < 0.8:
        plank = max(int(rng.uniform(0.06, 0.12) * s), 2)
        shade = rng.normal(0, rng.uniform(3, 10), size=(Hc // plank + 1, 1, 1))
        canvas = np.clip(canvas + np.repeat(shade, plank, axis=0)[:Hc], 0, 255).astype(np.uint8)
    noise = rng.normal(0, rng.uniform(2, 8), size=(Hc, W, 1))
    canvas = np.clip(canvas + noise, 0, 255).astype(np.uint8)

    court = np.array([[0, 0], [spec.length, 0], [spec.length, spec.width], [0, spec.width]])
    # apron (out of bounds): often painted, sometimes the same wood (hardest case)
    if rng.random() < 0.6:
        apron = canvas.copy()
        apron[:] = _random_colour(rng) if rng.random() < 0.7 else _hsv(rng, (0, 180), (0, 80), (20, 90))
        inside = np.zeros(canvas.shape[:2], np.uint8)
        cv2.fillPoly(inside, [px(court)], 1)
        canvas = np.where(inside[..., None] > 0, canvas, apron)

    c = spec.width / 2
    fills = []
    if rng.random() < 0.6:                                        # painted lanes (key)
        lane = spec.lane_polygon()
        fills.append([lane, np.c_[spec.length - lane[:, 0], lane[:, 1]]])
    if rng.random() < 0.25:                                       # painted 3-point areas
        arcs = [ln for ln in lines(spec) if len(ln) > 10 and np.ptp(ln[:, 1]) > spec.width / 2]
        fills.append([np.vstack([a, a[:1]]) for a in arcs])
    for group in fills:
        col = _random_colour(rng)
        for poly in group:
            cv2.fillPoly(canvas, [px(poly)], col)
    # logos: blobs at centre court and random spots
    for _ in range(rng.integers(0, 5)):
        cx = spec.length / 2 if rng.random() < 0.5 else rng.uniform(2, spec.length - 2)
        cy = c if rng.random() < 0.6 else rng.uniform(1, spec.width - 1)
        ax = (int(rng.uniform(0.8, 3.5) * s), int(rng.uniform(0.5, 2.5) * s))
        cv2.ellipse(canvas, tuple(px([cx, cy])), ax, rng.uniform(0, 180), 0, 360, _random_colour(rng), -1)
        if rng.random() < 0.5:                                    # "text" inside the logo
            for k in range(rng.integers(2, 6)):
                x0 = px([cx - 1.5 + 0.6 * k, cy - 0.3])
                cv2.rectangle(canvas, tuple(x0), tuple(x0 + [int(0.4 * s), int(0.6 * s)]), _random_colour(rng), -1)

    # painted lines: white most of the time, sometimes dark or coloured
    r = rng.random()
    lc = (245, 245, 245) if r < 0.55 else (25, 25, 25) if r < 0.75 else _random_colour(rng)
    thick = max(int(rng.uniform(0.04, 0.07) * s), 1)
    for ln in lines(spec):
        cv2.polylines(canvas, [px(ln)], False, lc, thick, cv2.LINE_AA)
    return canvas


def random_camera(rng: np.random.Generator, spec: CourtSpec, image_size) -> Camera:
    """Broadcast-like cameras, plus some corner and low ones (where earlier methods failed)."""
    L, W = spec.length, spec.width
    kind = rng.random()
    if kind < 0.7:                                                # main camera, behind the near sideline
        pos = (L / 2 + rng.uniform(-6, 6), W + rng.uniform(4, 35), rng.uniform(4, 20))
    elif kind < 0.9:                                              # corner / baseline side camera
        end = rng.choice([0.0, L])
        pos = (end + rng.uniform(-3, 3), W + rng.uniform(2, 15), rng.uniform(3, 12))
    else:                                                         # low courtside camera
        pos = (rng.uniform(0, L), W + rng.uniform(1.5, 6), rng.uniform(1.5, 4))
    target = (rng.uniform(-1, L + 1), rng.uniform(2, W - 2))
    return Camera(pos, target, float(np.exp(rng.uniform(np.log(500), np.log(3500)))), image_size)


def _background(rng, image_size) -> np.ndarray:
    """Crowd-like clutter: noisy blobs in a few dominant colours, plus ad-board bands."""
    w, h = image_size
    small = rng.integers(0, 255, size=(h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)
    base = np.array(_random_colour(rng), float)
    small = (0.5 * small + 0.5 * base).astype(np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 2.0))
    for _ in range(rng.integers(1, 4)):                           # ad boards
        y = int(rng.uniform(0.1, 0.6) * h)
        bh = int(rng.uniform(0.02, 0.08) * h)
        cv2.rectangle(img, (0, y), (w, y + bh), _random_colour(rng), -1)
        for x in range(0, w, int(rng.uniform(60, 250))):
            cv2.rectangle(img, (x, y + bh // 4), (x + int(rng.uniform(20, 120)), y + 3 * bh // 4),
                          _random_colour(rng), -1)
    return img


def _people(rng, img, cam: Camera, spec: CourtSpec, n: int) -> None:
    """Upright 'people' (coloured capsules) standing on the floor, inside and outside the court."""
    R = cam.rotation()
    C = np.asarray(cam.position, float)
    w, h = cam.image_size
    f = cam.focal

    def proj(X):
        Xc = R @ (np.asarray(X, float) - C)
        if Xc[2] <= 0.1:
            return None
        return np.array([f * Xc[0] / Xc[2] + w / 2, f * Xc[1] / Xc[2] + h / 2])

    for _ in range(n):
        x = rng.uniform(-3, spec.length + 3)
        y = rng.uniform(-3, spec.width + 3)
        height = rng.uniform(1.2, 2.1)                            # some are seated (bench, press row)
        a, b = proj([x, y, 0]), proj([x, y, height])
        if a is None or b is None:
            continue
        hpx = np.linalg.norm(a - b)
        if hpx < 8 or hpx > h:
            continue
        col = _random_colour(rng)
        wpx = max(int(hpx * 0.28), 2)
        cv2.rectangle(img, (int(a[0] - wpx / 2), int(b[1] + hpx * 0.15)), (int(a[0] + wpx / 2), int(a[1])), col, -1)
        head = _hsv(rng, (5, 20), (40, 150), (60, 220))
        cv2.circle(img, (int(b[0]), int(b[1] + hpx * 0.08)), max(int(hpx * 0.08), 1), head, -1)


def render(rng: np.random.Generator, image_size=(1280, 720), spec: CourtSpec | None = None,
           min_cover: float = 0.15, max_tries: int = 50) -> SynthSample:
    """One synthetic frame. The court covers at least ``min_cover`` of the image."""
    spec = spec or SPECS[rng.choice(["NBA", "FIBA", "NCAA"])]     # fixed list: same samples per seed
    w, h = image_size
    for _ in range(max_tries):
        cam = random_camera(rng, spec, image_size)
        H = cam.homography()
        mask = court_mask(H, spec, image_size)
        if mask.mean() >= min_cover:
            break
    canvas = paint_court(rng, spec)
    # canvas pixel -> court metres -> image
    A = np.array([[1 / PX_PER_M, 0, -APRON_M], [0, 1 / PX_PER_M, -APRON_M], [0, 0, 1.0]])
    img = _background(rng, image_size)
    floor = cv2.warpPerspective(canvas, H @ A, (w, h), flags=cv2.INTER_LINEAR)
    floor_valid = cv2.warpPerspective(np.full(canvas.shape[:2], 255, np.uint8), H @ A, (w, h)) > 0
    img[floor_valid] = floor[floor_valid]
    _people(rng, img, cam, spec, int(rng.integers(4, 25)))
    if rng.random() < 0.5:                                        # score bar
        y0 = int(rng.uniform(0.8, 0.92) * h)
        cv2.rectangle(img, (int(rng.uniform(0, 0.3) * w), y0), (int(rng.uniform(0.6, 1.0) * w), y0 + int(0.06 * h)),
                      _hsv(rng, (0, 180), (0, 120), (10, 90)), -1)
    # camera effects
    img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.5))
    img = np.clip(img.astype(np.float32) * rng.uniform(0.75, 1.25) + rng.uniform(-25, 25), 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(55, 95))])
    img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return SynthSample(img, mask, H, spec.name, cam)
