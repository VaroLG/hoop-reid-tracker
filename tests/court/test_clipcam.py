"""C07: one camera per clip — PTZ fit, clip camera, rejection of homographies from another camera."""

import numpy as np

from hooptrack.court.camera import Camera
from hooptrack.court.clipcam import ClipCameraConfig, fit_clip_camera, fit_ptz, plausible, ptz_homography
from hooptrack.court.leagues import FIBA, NBA
from hooptrack.court.propagate import disagreement

SPEC = NBA
POS = (14.5, 44.0, 12.0)          # behind the near sideline, 12 m up


def _pan(pos=POS, n=40, spec=SPEC):
    """A broadcast pan: fixed centre, target sweeps the court, zoom breathes."""
    xs = np.linspace(6, spec.length - 6, n)
    return [Camera(pos, (x, 7.5), 2800 + 400 * np.sin(i / 6)).homography() for i, x in enumerate(xs)]


def test_ptz_reproduces_a_camera_homography():
    cam = Camera(POS, (20.0, 6.0), 3100.0)
    H = cam.homography()
    p = fit_ptz(H, np.array(POS), SPEC)
    assert p.residual_m < 0.01 and abs(p.focal - 3100) < 1
    assert disagreement(H, ptz_homography(np.array(POS), p), SPEC)[1] < 0.01


def test_wrong_centre_costs_metres_only_if_far():
    """Pan / tilt / zoom absorb a small error in the centre; a different camera does not fit."""
    H = Camera(POS, (20.0, 6.0), 3100.0).homography()
    assert fit_ptz(H, np.array(POS) + [0.3, 0.3, 0.3], SPEC).residual_m < 0.2
    assert fit_ptz(Camera((5.0, 18.0, 2.5), (8.0, 6.0), 600.0).homography(), np.array(POS), SPEC).residual_m > 1.0


def test_clip_camera_recovers_centre_and_accepts_all():
    Hs = np.array(_pan())
    cam = fit_clip_camera(np.arange(1, len(Hs) + 1), Hs, SPEC)
    assert cam.verdict == "ok" and cam.share == 1.0
    assert np.linalg.norm(cam.position - POS) < 1.0


def test_frames_from_another_camera_are_rejected():
    """A minority of homographies from a low corner camera (the 009 failure) are dropped."""
    good = _pan(n=30)
    bad = [Camera((4.0, 17.0, 2.0), (x, 6.0), 500.0).homography() for x in np.linspace(4, 10, 10)]
    Hs = np.array(good + bad)
    cam = fit_clip_camera(np.arange(1, 41), Hs, SPEC)
    assert cam.verdict == "ok"
    assert set(cam.accepted) == set(range(1, 31))


def test_implausible_majority_does_not_win():
    """Wrong frames outnumber the right ones (Tenerife, 009): consensus alone would pick them."""
    good = _pan(n=12)
    bad = [Camera((4.0, 17.0, 2.0), (x, 6.0), 500.0).homography() for x in np.linspace(4, 10, 28)]
    cam = fit_clip_camera(np.arange(1, 41), np.array(good + bad), SPEC)
    assert set(cam.accepted) <= set(range(1, 13))
    assert cam.verdict == "review"                 # 12 / 40 explained: not enough to trust the clip


def test_only_implausible_cameras_go_to_review():
    Hs = np.array([Camera((5.0, 17.5, 3.0), (x, 6.0), 400.0).homography() for x in np.linspace(4, 12, 20)])
    cam = fit_clip_camera(np.arange(1, 21), Hs, FIBA)
    assert cam.verdict == "review"


def test_plausible_camera_box():
    cfg = ClipCameraConfig()
    assert plausible(np.array(POS), SPEC, cfg)
    assert not plausible(np.array([14.0, 17.0, 2.0]), FIBA, cfg)       # 2 m up, 2 m behind the sideline
    assert not plausible(np.array([14.0, 18.0, 9.0]), FIBA, cfg)       # high, but on the sideline
