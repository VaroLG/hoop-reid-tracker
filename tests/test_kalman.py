import numpy as np

from hooptrack.tracking.kalman import KalmanFilterXYWH


def test_learns_constant_velocity():
    """After a few observations the filter should predict the next position."""
    kf = KalmanFilterXYWH()
    mean, cov = kf.initiate(np.array([100.0, 200.0, 40.0, 100.0]))
    for t in range(1, 15):
        mean, cov = kf.predict(mean, cov)
        mean, cov = kf.update(mean, cov, np.array([100.0 + 5 * t, 200.0, 40.0, 100.0]))
    mean, _ = kf.predict(mean, cov)
    assert abs(mean[0] - (100 + 5 * 15)) < 2.0   # x predicted within 2 px
    assert abs(mean[4] - 5.0) < 0.5              # velocity estimated


def test_uncertainty_grows_without_measurements():
    kf = KalmanFilterXYWH()
    mean, cov = kf.initiate(np.array([0.0, 0.0, 50.0, 120.0]))
    var0 = cov[0, 0]
    for _ in range(10):
        mean, cov = kf.predict(mean, cov)
    assert cov[0, 0] > var0


def test_camera_translation_moves_state():
    kf = KalmanFilterXYWH()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 40.0, 80.0]))
    affine = np.array([[1.0, 0.0, -30.0], [0.0, 1.0, 10.0]])  # camera panned right & up
    m2, _ = kf.apply_affine(mean, cov, affine)
    np.testing.assert_allclose(m2[:4], [70.0, 110.0, 40.0, 80.0])


def test_camera_zoom_scales_box():
    kf = KalmanFilterXYWH()
    mean, cov = kf.initiate(np.array([100.0, 100.0, 40.0, 80.0]))
    affine = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])   # 2x zoom
    m2, _ = kf.apply_affine(mean, cov, affine)
    np.testing.assert_allclose(m2[:4], [200.0, 200.0, 80.0, 160.0])
