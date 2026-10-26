"""014: Expansion IoU (Deep-EIoU) — grown boxes, plain IoU when e = 0, ranking kept."""

import numpy as np

from hooptrack.tracking.matching import eiou_matrix, expand_boxes, iou_matrix


def test_expand_grows_each_side_by_e_times_the_size():
    out = expand_boxes(np.array([[100, 200, 140, 300]]), 0.5)   # w 40, h 100
    np.testing.assert_allclose(out, [[80, 150, 160, 350]])      # width (1 + 2*0.5) * 40 = 80


def test_zero_expansion_is_plain_iou():
    a, b = np.array([[0, 0, 10, 10]]), np.array([[5, 0, 15, 10]])
    np.testing.assert_allclose(eiou_matrix(a, b, 0.0), iou_matrix(a, b))


def test_close_but_not_overlapping_boxes_get_an_overlap():
    a = np.array([[0, 0, 40, 100]])
    b = np.array([[50, 0, 90, 100]])              # 10 px apart: IoU 0
    assert iou_matrix(a, b)[0, 0] == 0
    assert eiou_matrix(a, b, 0.7)[0, 0] > 0.3


def test_nearer_candidate_still_wins():
    track = np.array([[0, 0, 40, 100]])
    dets = np.array([[50, 0, 90, 100], [120, 0, 160, 100]])
    e = eiou_matrix(track, dets, 0.7)[0]
    assert e[0] > e[1]
