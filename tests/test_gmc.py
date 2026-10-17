import cv2
import numpy as np

from hooptrack.tracking.gmc import SparseOpticalFlowGMC


def _texture(h=360, w=640, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h + 100, w + 100, 3), np.uint8)
    for _ in range(300):
        c = (int(rng.integers(0, w + 100)), int(rng.integers(0, h + 100)))
        color = tuple(int(v) for v in rng.integers(0, 255, 3))
        cv2.circle(img, c, int(rng.integers(2, 15)), color, -1)
    return img


def test_recovers_camera_pan():
    """Camera moves right by 12 px and up by 5 px -> scene shifts (-12, +5)."""
    big = _texture()
    f0 = big[50:410, 50:690]
    f1 = big[45:405, 62:702]
    gmc = SparseOpticalFlowGMC(downscale=1)
    gmc.apply(f0)
    affine = gmc.apply(f1)
    np.testing.assert_allclose(affine[:, 2], [-12, 5], atol=0.5)
    np.testing.assert_allclose(affine[:, :2], np.eye(2), atol=0.01)


def test_masks_players():
    """Features inside detection boxes are ignored (players move on their own)."""
    big = _texture()
    f0 = big[50:410, 50:690].copy()
    gmc = SparseOpticalFlowGMC(downscale=1)
    gmc.apply(f0, boxes=np.array([[0, 0, 640, 360]]))   # whole frame masked
    pts = gmc._prev_pts
    assert pts is None or len(pts) == 0
