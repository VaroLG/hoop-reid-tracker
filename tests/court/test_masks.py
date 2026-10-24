"""C06: court-area masks from homographies (training labels)."""

import cv2
import numpy as np

from hooptrack.court.camera import Camera
from hooptrack.court.leagues import FIBA, NBA
from hooptrack.court.masks import court_mask, court_polygon, mask_iou, yolo_seg_line


def test_mask_matches_a_dense_projection():
    H = Camera((NBA.length / 2, NBA.width + 15, 10.0), (NBA.length / 2, 7.0), 900.0).homography()
    m = court_mask(H, NBA, (1280, 720))
    # every pixel whose back-projection falls inside the court must be 1, and vice versa
    ys, xs = np.mgrid[0:720:7, 0:1280:7]
    pts = cv2.perspectiveTransform(np.c_[xs.ravel(), ys.ravel()][None].astype(np.float64), np.linalg.inv(H))[0]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] <= NBA.length) & (pts[:, 1] >= 0) & (pts[:, 1] <= NBA.width)
    assert np.mean(m[ys.ravel(), xs.ravel()].astype(bool) == inside) > 0.995


def test_court_partly_behind_the_camera_is_clipped_not_flipped():
    """A low camera on the sideline looking along the court: the far baseline behind it must not reappear."""
    cam = Camera((2.0, FIBA.width + 1.0, 4.0), (20.0, 5.0), 700.0)       # near one baseline, looking down the court
    H = cam.homography()
    poly = court_polygon(H, FIBA, (1280, 720))
    assert len(poly) >= 3
    m = court_mask(H, FIBA, (1280, 720))
    ys, xs = np.nonzero(m)
    pts = cv2.perspectiveTransform(np.c_[xs, ys][None].astype(np.float64), np.linalg.inv(H))[0]
    ok = (pts[:, 0] >= -0.2) & (pts[:, 0] <= FIBA.length + 0.2) & (pts[:, 1] >= -0.2) & (pts[:, 1] <= FIBA.width + 0.2)
    assert ok.mean() > 0.99


def test_yolo_line_and_iou():
    poly = np.array([[0, 0], [640, 0], [640, 360], [0, 360]], float)
    line = yolo_seg_line(poly, (1280, 720))
    assert line.startswith("0 0.000000 0.000000 0.500000")
    a = np.zeros((10, 10))
    a[:5] = 1
    assert mask_iou(a, a) == 1.0 and abs(mask_iou(a, np.ones((10, 10))) - 0.5) < 1e-9


def test_synthetic_frames_have_exact_masks():
    """The label of a synthetic frame is the court of its own homography, and the court is visible."""
    from hooptrack.court.leagues import SPECS
    from hooptrack.court.synth import render
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = render(rng, (640, 360))
        assert s.image.shape == (360, 640, 3) and s.mask.shape == (360, 640)
        assert s.mask.mean() >= 0.15
        assert mask_iou(s.mask, court_mask(s.H, SPECS[s.spec], (640, 360))) == 1.0


def test_convex_cleanup_and_feet_test():
    from hooptrack.court.segment import convex_largest, feet_inside
    m = np.zeros((100, 200), np.uint8)
    m[20:80, 20:180] = 1
    m[60:80, 90:110] = 0                     # a player-shaped notch in the court
    m[0:5, 0:5] = 1                          # a stray blob
    c = convex_largest(m)
    assert c[70, 100] == 1 and c[2, 2] == 0
    boxes = np.array([[90, 30, 110, 79], [90, 30, 110, 95], [90, 0, 110, 15]], float)
    assert feet_inside(c, boxes, margin=0.05).tolist() == [True, False, False]
