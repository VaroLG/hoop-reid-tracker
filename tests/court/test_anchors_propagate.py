"""C04: anchor homographies and propagation through camera motion (synthetic scenes)."""

import cv2
import numpy as np
import pytest

from hooptrack.court.anchors import Anchor, homography_from_points, line_intersection
from hooptrack.court.leagues import NCAA_PRE2019, keypoints, lines
from hooptrack.court.propagate import Chain, ChainStep, disagreement, merge_chains, track_homography, trusted

SPEC = NCAA_PRE2019
COURT = np.array([[0, 0], [SPEC.length, 0], [SPEC.length, SPEC.width], [0, SPEC.width]], np.float32)


def _view(dx=0.0):
    """A broadcast-like view of the court, panned ``dx`` pixels."""
    dst = np.float32([[200, 150], [1180, 140], [1270, 650], [60, 680]]) + np.float32([dx, 0])
    return cv2.getPerspectiveTransform(COURT, dst)


def _render(H, seed=0):
    """Wood-like textured floor (for optical flow) with dark painted lines."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.integers(120, 220, (int(SPEC.width * 20), int(SPEC.length * 20)), np.uint8), (5, 5), 0)
    scale = np.diag([20.0, 20.0, 1.0])                                      # metres -> texture px
    img = cv2.warpPerspective(tex, H @ np.linalg.inv(scale), (1280, 720), borderValue=90)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for pl in lines(SPEC):
        p = cv2.perspectiveTransform(pl[None].astype(np.float64), H)[0].astype(np.int32)
        cv2.polylines(img, [p], False, (30, 30, 30), 3)
    return img


def test_homography_from_points_is_exact_on_true_points():
    H = _view()
    kp = keypoints(SPEC)
    names = ["04", "05", "12", "14", "19", "23"]
    pts = {n: tuple(cv2.perspectiveTransform(np.array([[kp[n]]], float), H)[0, 0]) for n in names}
    H2 = homography_from_points(pts, SPEC)
    assert disagreement(H, H2, SPEC)[0] < 1e-3


def test_homography_from_points_rejects_bad_input():
    with pytest.raises(KeyError):
        homography_from_points({"99": (0, 0), "04": (1, 1), "05": (2, 2), "12": (3, 3)}, SPEC)
    with pytest.raises(ValueError):
        homography_from_points({"04": (1, 1), "05": (2, 2), "12": (3, 3)}, SPEC)


def test_line_intersection():
    p = line_intersection((0, 0), (10, 10), (0, 10), (10, 0))
    assert np.allclose(p, (5, 5))


def test_track_homography_follows_a_pan():
    """Camera pans 6 px per frame for 10 frames: the chain must end on the true court."""
    frames = [(i, _render(_view(-6 * i))) for i in range(11)]
    steps = track_homography(frames, _view(0), SPEC)
    assert [s.frame for s in steps] == list(range(1, 11))
    px, m = disagreement(steps[-1].H, _view(-60), SPEC)
    assert px < 2.0 and m < 0.1
    assert min(s.agreement for s in steps) > 0.7


def test_track_homography_survives_a_flash():
    """One over-exposed frame in the middle of a pan must not derail the chain."""
    frames = [(i, _render(_view(-6 * i))) for i in range(9)]
    frames[4] = (4, cv2.convertScaleAbs(frames[4][1], alpha=1.6, beta=60))
    steps = track_homography(frames, _view(0), SPEC)
    px, _ = disagreement(steps[-1].H, _view(-48), SPEC)
    assert px < 2.0


def test_merge_prefers_agreement_then_nearest_anchor():
    eye = np.eye(3)
    a = Chain("a", 0, [ChainStep(0, eye, 0.9, 0), ChainStep(5, eye * 1, 0.4, 0), ChainStep(9, eye, 0.5, 0)])
    b = Chain("b", 10, [ChainStep(10, eye, 0.9, 0), ChainStep(5, eye * 2, 0.6, 0), ChainStep(9, eye * 3, 0.8, 0)])
    best = merge_chains([a, b])
    assert best[5] is b.steps[1] and best[9] is b.steps[2]
    best = merge_chains([a, b], nearest_ranges=[(4, 6)])
    assert best[5] is a.steps[1]                    # 5 frames from both anchors: a comes first (tie -> min keeps first)
    assert best[9] is b.steps[2]                    # outside the range: agreement decides


def test_anchor_defaults():
    a = Anchor(1, {"04": (0, 0)})
    assert a.taus == (12.0, 5.0)


def test_trusted_cuts_a_chain_after_it_gets_lost():
    eye = np.eye(3)
    agree = [0.8, 0.7, 0.2, 0.6, 0.2, 0.2, 0.2, 0.2, 0.2, 0.99, 0.99]   # brief dip survives, 5 lows = lost
    ch = Chain("c", 0, [ChainStep(i, eye, a, 0) for i, a in enumerate(agree)])
    cut = trusted(ch, lost_below=0.3, patience=5)
    assert [s.frame for s in cut.steps] == [0, 1, 2, 3]          # the late 0.99 is not trusted
    assert trusted(Chain("d", 0, ch.steps[:4])).steps == ch.steps[:4]
