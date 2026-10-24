"""C08: tracks -> court plan in metres (integrations/topdown.py), on a synthetic camera."""

import cv2
import numpy as np

from hooptrack.court.camera import Camera
from hooptrack.court.leagues import NBA
from hooptrack.integrations.topdown import TopDownConfig, project_tracks, track_stats

SPEC = NBA
H = Camera((14.3, 44.0, 12.0), (14.3, 7.6), 1500.0).homography()


def _box(x, y, h=180.0):
    """A box whose bottom centre is the image of court point (x, y)."""
    u, v = cv2.perspectiveTransform(np.array([[[x, y]]], float), H)[0, 0]
    return [u - 30, v - h, u + 30, v]


def _walk(n=50, speed=5.0, fps=25.0, noise=0.0, seed=0):
    """Player 7 runs along the court at ``speed`` m/s; feet jitter by ``noise`` px."""
    rng = np.random.default_rng(seed)
    tracks = {}
    for f in range(1, n + 1):
        b = np.array([_box(5 + speed * (f - 1) / fps, 7.0)])
        b[:, [1, 3]] += rng.normal(0, noise, (1, 1))
        tracks[f] = (np.array([7]), b)
    return tracks


def test_feet_land_where_the_player_stands():
    df = project_tracks({1: (np.array([3, 4]), np.array([_box(10.0, 4.0), _box(25.0, 12.0)]))}, {1: H}, SPEC)
    assert np.allclose(df[["x", "y"]].to_numpy(), [[10, 4], [25, 12]], atol=1e-6)


def test_frames_without_verified_court_get_no_position():
    tracks = _walk(10)
    df = project_tracks(tracks, {f: H for f in range(1, 11) if f != 5}, SPEC)
    assert 5 not in set(df["frame"]) and len(df) == 9


def test_people_off_the_court_are_dropped():
    df = project_tracks({1: (np.array([1, 2]), np.array([_box(10.0, 4.0), _box(10.0, -4.0)]))}, {1: H}, SPEC)
    assert list(df["id"]) == [1]


def test_speed_and_distance():
    df = project_tracks(_walk(51, speed=5.0), {f: H for f in range(1, 52)}, SPEC)
    s = track_stats(df, fps=25.0).iloc[0]
    assert abs(s["distance_m"] - 10.0) < 0.01 and abs(s["speed_mean_ms"] - 5.0) < 0.01


def test_smoothing_removes_jitter_from_distance():
    """Pixel noise at the feet inflates the distance run; the per-track smoothing takes most of it out."""
    tracks, Hs = _walk(51, noise=2.0), {f: H for f in range(1, 52)}
    raw = track_stats(project_tracks(tracks, Hs, SPEC, TopDownConfig(smooth=0)), 25.0).iloc[0]["distance_m"]
    smooth = track_stats(project_tracks(tracks, Hs, SPEC, TopDownConfig(smooth=9)), 25.0).iloc[0]["distance_m"]
    assert raw - 10.0 > 2 * abs(smooth - 10.0)
