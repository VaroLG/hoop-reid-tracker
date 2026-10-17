import numpy as np

from hooptrack.tracking.matching import (
    INF,
    cosine_distance,
    coverage_matrix,
    fuse_motion_appearance,
    iou_matrix,
    linear_assignment,
)


def test_iou_known_values():
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(iou_matrix(a, b), [[1.0, 50 / 150, 0.0]])


def test_iou_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


def test_coverage_detects_hidden_small_box():
    big = [0, 0, 100, 200]
    small = [10, 10, 30, 50]              # fully inside the big box
    cov = coverage_matrix(np.array([big, small]))
    assert cov[1, 0] == 1.0               # small is 100 % covered
    assert cov[0, 1] < 0.05               # big is barely covered
    assert iou_matrix(np.array([small]), np.array([big]))[0, 0] < 0.05  # IoU misses it


def test_cosine_distance_range():
    f = np.array([[1, 0], [0, 1], [-1, 0]], dtype=float)
    d = cosine_distance(f, f)
    np.testing.assert_allclose(np.diag(d), 0, atol=1e-6)
    assert d[0, 1] == 1.0 - 0.0           # orthogonal
    assert d[0, 2] == 1.0                 # opposite is clipped to 1


def test_linear_assignment_respects_threshold():
    cost = np.array([[0.1, 0.9], [0.8, 0.95]])
    m, ur, uc = linear_assignment(cost, thresh=0.5)
    assert m == [(0, 0)]
    assert ur == [1] and uc == [1]


def test_appearance_gate_blocks_overlapping_strangers():
    """Two overlapping boxes with very different appearance must not match."""
    iou = np.array([[0.9]])
    app = np.array([[0.8]])
    cost = fuse_motion_appearance(iou, app, 0.5, iou_gate=0.2, appearance_gate=0.35)
    assert cost[0, 0] >= INF


def test_nan_appearance_falls_back_to_motion():
    iou = np.array([[0.9]])
    app = np.array([[np.nan]])
    cost = fuse_motion_appearance(iou, app, 0.5, iou_gate=0.2, appearance_gate=0.35)
    np.testing.assert_allclose(cost, [[0.1]])
