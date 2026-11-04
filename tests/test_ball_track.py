"""B05: offline ball tracking over the detector's candidates."""

import numpy as np

from hooptrack.ball.track import track_ball


def det(x, y, conf, d=12.0):
    return np.array([[x, y, d, conf, 0.0]])


def test_isolated_false_alarm_is_dropped_and_gap_is_filled():
    # a ball moving right 5 px per frame, detected every other frame; a confident-looking false alarm far away
    dets = {f: det(100 + 5 * f, 300, 0.4) for f in range(0, 40, 2)}
    dets[21] = det(900, 100, 0.45)
    out = track_ball(dets)
    assert 21 in out and abs(out[21][0] - (100 + 5 * 21)) < 1e-6 and out[21][3] == "interp"   # filled, not the FA
    assert all(abs(v[0] - (100 + 5 * f)) < 1e-6 for f, v in out.items())


def test_weak_detections_are_kept_between_confident_ones():
    # confident at both ends, weak (worth < 0 alone) in between, all on one trajectory: the weak ones are kept
    # (with weak detections worth < 0 the path may skip them, the interpolation covers those frames anyway)
    dets = {f: det(200 + 3 * f, 200, 0.07) for f in range(1, 14)}
    dets[0], dets[14] = det(200, 200, 0.5), det(200 + 3 * 14, 200, 0.5)
    out = track_ball(dets)
    assert sorted(out) == list(range(15)) and all(abs(v[0] - (200 + 3 * f)) < 1e-6 for f, v in out.items())


def test_only_weak_detections_give_no_track():
    dets = {f: det(200 + 3 * f, 200, 0.07) for f in range(0, 30)}     # noise level everywhere
    assert len(track_ball(dets)) <= 1


def test_two_candidates_in_a_frame_the_consistent_one_wins():
    dets = {f: det(100 + 4 * f, 300, 0.3) for f in range(0, 20)}
    dets[10] = np.vstack([det(140, 300, 0.2), det(600, 50, 0.5)])
    out = track_ball(dets)
    assert abs(out[10][0] - 140) < 1e-6


def test_weak_segment_is_dropped_when_asked():
    from hooptrack.ball.track import BallTrackConfig

    dets = {f: det(100 + 5 * f, 300, 0.5) for f in range(0, 30)}
    dets.update({f: det(800, 100 + 2 * f, 0.2) for f in range(60, 66)})       # short, weak run far away (~4.2)
    keep = track_ball(dets, BallTrackConfig())
    drop = track_ball(dets, BallTrackConfig(min_segment_score=5.0))
    assert 62 in keep and 62 not in drop and 10 in drop


def test_static_candidates_are_removed_on_the_court_plan():
    from hooptrack.integrations.ball_static import remove_static

    H = {f: np.eye(3) for f in range(100)}                     # image = plan (metres) for the test
    dets = {f: np.array([[5.0, 5.0, 0.2, 0.4, 0], [0.1 * f, 3.0, 0.2, 0.4, 0]]) for f in range(100)}
    out, removed = remove_static(dets, H, min_frames=50)
    assert removed == 100 and all(len(v) == 1 and v[0, 1] == 3.0 for v in out.values())
