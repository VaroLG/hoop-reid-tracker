"""C02: dataset helpers (split by game, flip index, clamped labels)."""

import numpy as np

from hooptrack.court import nba_template
from hooptrack.court.dataset import Record, clamped_labels, flip_index, game_of, split_by_game


def test_game_of_strips_quarter_and_clock():
    name = "denver-nuggets-la-clippers-game-1-q1-06_17-06_12_mp4-000.jpg"
    assert game_of(name) == "denver-nuggets-la-clippers-game-1"


def test_split_by_game_is_deterministic_and_disjoint():
    games = [f"team-{i}-game-1" for i in range(20)]
    a, b = split_by_game(games), split_by_game(list(reversed(games)))
    assert a == b                                                     # order of input does not matter
    s = [set(a[k]) for k in ("train", "valid", "test")]
    assert not (s[0] & s[1] or s[0] & s[2] or s[1] & s[2])
    assert sum(map(len, s)) == 20


def test_flip_index_is_an_involution():
    tpl = nba_template()
    names = sorted(tpl.points)
    f = flip_index(names, tpl)
    assert all(f[f[i]] == i for i in range(len(names)))              # flipping twice = identity
    assert names[f[names.index("01")]] == "34" and names[f[names.index("21")]] == "21"


def test_clamped_label_detected():
    """Project the template with a synthetic camera, then move one label to the image border."""
    import cv2

    tpl = nba_template()
    names = sorted(tpl.points)
    H = np.array([[18.0, 4.0, 40.0], [0.0, 22.0, 60.0], [0.0, 0.004, 1.0]])
    xy = cv2.perspectiveTransform(np.array([[tpl.points[n] for n in names]]), H)[0]
    kp = np.c_[xy, np.full(len(names), 2.0)]
    kp[names.index("41"), 1] = 575.0                                 # clamped to the bottom border
    rec = Record("x.jpg", "train", "g", (1024, 576), np.zeros(4), kp)
    assert clamped_labels(rec, names, tpl) == ["41"]
