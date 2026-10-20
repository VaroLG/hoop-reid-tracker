"""009: court mask from precomputed court homographies (integration of chapter 2 into the tracker)."""

from types import SimpleNamespace

import numpy as np

from hooptrack.court.camera import Camera
from hooptrack.court.leagues import NBA
from hooptrack.integrations.court_filter import HomographyCourtConfig, HomographyCourtFilter, build_court_filters


def _setup(tmp_path, frames=(1, 2)):
    H = Camera((NBA.length / 2, NBA.width + 18, 12.0), (NBA.length / 2, 7.0), 1300.0).homography()
    d = tmp_path / "v_test"
    d.mkdir()
    np.savez(d / "homographies.npz", frames=np.array(frames), H=np.array([H] * len(frames)), league="NBA")
    return H


def _feet_box(H, x, y, h=150):
    """A person whose feet stand at court point (x, y) metres."""
    p = H @ np.array([x, y, 1.0])
    u, v = p[:2] / p[2]
    return [u - 20, v - h, u + 20, v]


def test_keeps_feet_on_court_and_drops_the_rest(tmp_path):
    H = _setup(tmp_path)
    flt = HomographyCourtFilter(HomographyCourtConfig(dir=str(tmp_path), margin_m=0.25), colour=None)
    flt.bind(SimpleNamespace(name="v_test"))
    boxes = np.array([_feet_box(H, 14, 7),            # midcourt: keep
                      _feet_box(H, 3, 15.2),          # on the near sideline: keep
                      _feet_box(H, 14, 16.5)])        # 1.3 m beyond the near sideline (bench): drop
    kept, _ = flt(np.zeros((720, 1280, 3), np.uint8), boxes, np.ones(3))
    assert len(kept) == 2 and np.allclose(kept, boxes[:2])
    assert flt.stats()["court_frames_homography"] == 1


def test_frames_without_homography_fall_back(tmp_path):
    H = _setup(tmp_path, frames=(1,))
    flt = HomographyCourtFilter(HomographyCourtConfig(dir=str(tmp_path)), colour=None)
    flt.bind(SimpleNamespace(name="v_test"))
    img = np.zeros((720, 1280, 3), np.uint8)
    boxes = np.array([_feet_box(H, 14, 20)])          # off court
    assert len(flt(img, boxes, np.ones(1))[0]) == 0    # frame 1: homography -> dropped
    assert len(flt(img, boxes, np.ones(1))[0]) == 1    # frame 2: none, no fallback -> unfiltered
    s = flt.stats()
    assert s["court_frames_homography"] == 1 and s["court_frames_unfiltered"] == 1


def test_unknown_video_and_switch_off(tmp_path):
    flt = HomographyCourtFilter(HomographyCourtConfig(dir=str(tmp_path)), colour=None)
    flt.bind(SimpleNamespace(name="v_missing"))       # no file: every frame unfiltered (or colour fallback)
    assert len(flt(np.zeros((720, 1280, 3), np.uint8), np.array([[0, 0, 10, 10]]), np.ones(1))[0]) == 1
    assert build_court_filters({"court_mask": True}) is None
    assert len(build_court_filters({"court_source": "homography"})) == 1


def test_guard_distrusts_a_homography_that_drops_someone_deep_inside_the_floor(tmp_path):
    """009b: a wrong homography (court shifted 10 m) would drop a player standing mid-floor."""
    from hooptrack.detection.filters import CourtMask
    H = _setup(tmp_path)
    shift = H @ np.array([[1, 0, 10.0], [0, 1, 0], [0, 0, 1]])           # court drawn 10 m to the side
    d = tmp_path / "v_test"
    np.savez(d / "homographies.npz", frames=np.array([1, 2]), H=np.array([shift, shift]), league="NBA")
    img = np.zeros((720, 1280, 3), np.uint8)
    img[300:700, 100:1200] = (60, 140, 200)                               # a big uniform "floor"
    box = np.array([[620, 400, 660, 600]], float)                          # feet deep inside it
    for guard, expect in ((False, 0), (True, 1)):
        flt = HomographyCourtFilter(HomographyCourtConfig(dir=str(tmp_path), guard_colour=guard), colour=CourtMask())
        flt.bind(SimpleNamespace(name="v_test"))
        assert not flt._on_court(box, shift)[0]                            # the wrong court excludes them
        kept, _ = flt(img, box, np.ones(1))
        assert len(kept) == expect, guard
