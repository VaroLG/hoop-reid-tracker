"""009: the court template must be geometrically consistent with official NBA dimensions."""

import cv2
import numpy as np

from hooptrack.court import FT_TO_M, nba_template


def test_template_is_mirror_symmetric():
    tpl = nba_template()
    pairs = [("01", "34"), ("04", "37"), ("12", "28"), ("16", "26"), ("15", "25"), ("10", "31"), ("09", "33")]
    for a, b in pairs:
        (xa, ya), (xb, yb) = tpl.points[a], tpl.points[b]
        assert abs((xa + xb) - tpl.length) < 1e-6 and abs(ya - yb) < 1e-6


def test_rims_excluded_from_floor_homography():
    tpl = nba_template()
    names, xy = tpl.floor_points(["01", "09", "21", "33", "41"])
    assert names == ["01", "21", "41"] and xy.shape == (3, 2)


def test_homography_round_trip():
    """A synthetic camera: project the template, fit H back from 6 points, recover the rest."""
    tpl = nba_template()
    H_true = np.array([[40.0, 8.0, 100.0], [0.0, 25.0, 150.0], [0.0, 0.01, 1.0]])
    fit = ["01", "34", "41", "08", "21", "13"]          # not all on two lines (degenerate)
    src = np.array([tpl.points[n] for n in fit])
    dst = cv2.perspectiveTransform(src[None], H_true)[0]
    H, _ = cv2.findHomography(src, dst)
    test = np.array([tpl.points[n] for n in ("19", "29", "16")])
    err = np.abs(cv2.perspectiveTransform(test[None], H)[0] - cv2.perspectiveTransform(test[None], H_true)[0])
    assert err.max() < 1e-3                               # pixels (float32 inside OpenCV)
    assert abs(tpl.length - 94 * FT_TO_M) < 1e-9
