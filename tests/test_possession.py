"""P01: who has the ball, from the tracked ball and players."""

import numpy as np

from hooptrack.ball.possession import AIR, events, possession


def test_held_then_passed_then_caught():
    # player 1 at x 100-140, player 2 at x 600-640 (boxes 100 px tall); the ball is held by 1 (frames 1-20), flies
    # to 2 (21-35, 30 px per frame: fast), held by 2 (36-60)
    boxes = np.array([[100, 300, 140, 400], [600, 300, 640, 400]], float)
    players = {f: (np.array([1, 2]), boxes) for f in range(1, 61)}
    ball = {}
    for f in range(1, 61):
        x = 130 if f <= 20 else (610 if f >= 36 else 130 + (f - 20) * 32)
        ball[f] = (x, 345, 12)
    h = possession(ball, players, 60)
    assert (h[:19] == 1).all() and (h[38:] == 2).all() and (h[24:30] == AIR).all()
    kinds = [e[1] for e in events(h)]
    assert kinds == ["release", "catch"]


def test_a_ball_flying_past_a_player_is_not_his():
    players = {f: (np.array([1]), np.array([[300, 300, 340, 400]], float)) for f in range(1, 31)}
    ball = {f: (50 + 20 * f, 340, 12) for f in range(1, 31)}         # crosses his reach box at 20 px / frame
    h = possession(ball, players, 30)
    assert (h[h != -2] == AIR).all()


def test_motion_possession_sees_a_pass_and_one_holder_per_stretch():
    from hooptrack.ball.possession import possession_motion

    players = {f: (np.array([1, 2, 3]), np.array([[100, 300, 140, 400], [110, 300, 150, 400], [600, 300, 640, 400]],
                                                 float)) for f in range(1, 61)}
    ball = {}
    for f in range(1, 61):                       # held near players 1 and 2 (overlapping), passed fast to 3
        x = 125 + (f % 2) if f <= 20 else (615 if f >= 36 else 125 + (f - 20) * 32)
        ball[f] = (x, 345, 12)
    h = possession_motion(ball, players, 60)
    assert len(set(h[:19])) == 1 and h[0] in (1, 2)          # one holder for the whole stretch, no flipping
    assert (h[24:32] == AIR).all() and (h[40:] == 3).all()


def test_keep_state_through_frames_without_a_ball():
    from hooptrack.ball.possession import PossessionConfig

    players = {f: (np.array([1]), np.array([[100, 300, 140, 400]], float)) for f in range(1, 61)}
    ball = {f: (125, 345, 12) for f in list(range(5, 15)) + list(range(45, 61))}      # unseen 15-44
    off = possession(ball, players, 60)
    on = possession(ball, players, 60, PossessionConfig(keep_state=True))
    assert (off[30:40] == -2).all() and (on == 1).all()


def test_viterbi_removes_single_frame_flicker():
    """P05: one isolated 'air' frame inside a long possession is smoothed away; a long air stretch is kept."""
    from hooptrack.ball.possession_temporal import viterbi

    p = np.r_[np.full(30, 0.9), 0.3, np.full(30, 0.9), np.full(30, 0.05)]
    s = viterbi(p, prior_held=0.7, p_switch=0.02)
    assert s[:61].all() and not s[70:].any()
