"""C07: one camera per clip — PTZ fit, clip camera, rejection of homographies from another camera."""

import numpy as np

from hooptrack.court.camera import Camera
from hooptrack.court.clipcam import (ClipCameraConfig, fit_clip_camera, fit_ptz, plausible, ptz_homography,
                                     smooth_ptz)
from hooptrack.court.leagues import FIBA, NBA
from hooptrack.court.propagate import blend, disagreement, flow_smooth

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


def test_smoothed_camera_fills_gaps_and_drops_a_jump():
    """C09: a frame fitted far off is replaced, short gaps are filled, long ones are not."""
    C = np.array(POS)
    Hs = [Camera(POS, (x, 7.5), 2800 + 400 * np.sin(i / 30)).homography()       # ~4 s pan at 30 fps
          for i, x in enumerate(np.linspace(8, 20, 120))][:60]
    ptz = {i + 1: fit_ptz(H, C, SPEC) for i, H in enumerate(Hs)}
    ptz[20] = fit_ptz(Camera(POS, (25.0, 3.0), 2000.0).homography(), C, SPEC)     # one wild frame
    frames = [f for f in ptz if not 30 <= f <= 34 and not 41 <= f <= 59]          # 5-frame gap, 19-frame gap
    sm = smooth_ptz(ptz, frames, window=9, max_fill=10)
    assert all(f in sm for f in range(30, 35)) and not any(f in sm for f in range(41, 60))
    for f in (20, 32):                                                             # back on the true camera
        assert disagreement(ptz_homography(C, sm[f]), Hs[f - 1], SPEC)[1] < 0.2
    assert np.isnan(sm[32].residual_m)


def test_flow_smooth_follows_a_fast_pan_and_drops_a_wrong_frame():
    """C09: neighbours carried by the image motion: no lag however fast the camera moves."""
    cams = [Camera(POS, (x, 7.5), 2800 + 30 * i) for i, x in enumerate(np.linspace(6, 22, 20))]   # fast pan + zoom
    Hs = [c.homography() for c in cams]
    motion = {f: Hs[f - 1] @ np.linalg.inv(Hs[f - 2]) for f in range(2, 21)}     # exact image motion t-1 -> t
    noisy = list(Hs)
    noisy[9] = Camera(POS, (13.0, 3.0), 2500).homography()                       # one wrong frame
    out = flow_smooth(list(range(1, 21)), noisy, motion, k=3)
    assert max(disagreement(o, h, SPEC)[1] for o, h in zip(out, Hs, strict=True)) < 0.05


def test_blend_goes_from_one_court_to_the_other():
    Ha, Hb = Camera(POS, (10.0, 7.5), 2800).homography(), Camera(POS, (12.0, 7.5), 2800).homography()
    assert disagreement(blend(Ha, Hb, 0.0), Ha, SPEC)[1] < 1e-3
    assert disagreement(blend(Ha, Hb, 1.0), Hb, SPEC)[1] < 1e-3
    mid = disagreement(blend(Ha, Hb, 0.5), Ha, SPEC)[1]
    assert 0.3 * disagreement(Ha, Hb, SPEC)[1] < mid < 0.7 * disagreement(Ha, Hb, SPEC)[1]


def test_plausible_camera_box():
    cfg = ClipCameraConfig()
    assert plausible(np.array(POS), SPEC, cfg)
    assert not plausible(np.array([14.0, 17.0, 2.0]), FIBA, cfg)       # 2 m up, 2 m behind the sideline
    assert not plausible(np.array([14.0, 18.0, 9.0]), FIBA, cfg)       # high, but on the sideline
