"""C03: league templates and line-based refinement."""

import cv2
import numpy as np

from hooptrack.court.leagues import FIBA, NBA, NCAA_PRE2019, keypoints, lines
from hooptrack.court.refine import line_evidence, refine, sample_lines, score
from hooptrack.court.template import NBA_KEYPOINTS


def test_nba_spec_matches_verified_template():
    k = keypoints(NBA)
    assert set(k) == set(NBA_KEYPOINTS)
    # only the 3-pt straight ends differ (14.2 ft from the arc geometry vs 14 ft nominal)
    assert max(np.hypot(*np.subtract(k[n], NBA_KEYPOINTS[n])) for n in k) < 0.07


def test_leagues_differ_where_they_should():
    nba, fiba, ncaa = keypoints(NBA), keypoints(FIBA), keypoints(NCAA_PRE2019)
    assert abs((nba["05"][1] - nba["04"][1]) - 16 * 0.3048) < 1e-6        # NBA lane 16 ft
    assert abs((ncaa["05"][1] - ncaa["04"][1]) - 12 * 0.3048) < 1e-6      # NCAA lane 12 ft
    assert abs(fiba["34"][0] - 28.0) < 1e-9                                 # FIBA court 28 m


def _synthetic(H):
    img = np.full((720, 1280, 3), 170, np.uint8)
    for pl in lines(NBA):
        p = cv2.perspectiveTransform(pl[None].astype(np.float64), H)[0].astype(np.int32)
        cv2.polylines(img, [p], False, (40, 40, 40), 3)
    return img


def test_refine_recovers_a_perturbed_homography():
    court = np.array([[0, 0], [28.65, 0], [28.65, 15.24], [0, 15.24]], np.float32)
    true = cv2.getPerspectiveTransform(court, np.float32([[200, 150], [1180, 140], [1270, 650], [60, 680]]))
    rough = cv2.getPerspectiveTransform(court, np.float32([[215, 160], [1165, 132], [1282, 640], [48, 690]]))
    ev = line_evidence(_synthetic(true))
    S, T = sample_lines(lines(NBA))
    a_rough, _ = score(rough, S, ev, tangents=T)
    a_fit, _ = score(refine(rough, S, ev, T), S, ev, tangents=T)
    a_true, _ = score(true, S, ev, tangents=T)
    assert a_fit > a_rough + 0.3 and a_fit > 0.9 * a_true
