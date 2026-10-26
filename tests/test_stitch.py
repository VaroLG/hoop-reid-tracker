"""013: offline tracklet stitching — merge fragments of one player, never two players."""

import numpy as np

from hooptrack.tracking.stitch import StitchConfig, Tracklet, stitch

RNG = np.random.default_rng(0)
D = 32
PLAYERS = {p: v / np.linalg.norm(v) for p, v in enumerate(RNG.normal(size=(4, D)))}


def crops(player: int, n: int = 6, noise: float = 0.05) -> np.ndarray:
    """n embeddings of one player: his vector + small noise, L2-normalised."""
    x = PLAYERS[player] + noise * RNG.normal(size=(n, D))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def tr(tid: int, start: int, end: int, player: int, **kw) -> Tracklet:
    return Tracklet(tid, np.arange(start, end + 1), crops(player, **kw))


CFG = StitchConfig(thresh=0.3, margin=0.05)


def test_fragments_of_one_player_are_merged_to_the_earliest_id():
    mapping, _ = stitch([tr(1, 0, 50, 0), tr(7, 80, 120, 0), tr(9, 150, 200, 0)], CFG)
    assert mapping == {1: 1, 7: 1, 9: 1}


def test_two_players_are_never_merged():
    mapping, _ = stitch([tr(1, 0, 50, 0), tr(2, 80, 120, 1)], CFG)
    assert mapping == {1: 1, 2: 2}


def test_tracklets_on_screen_together_are_never_merged_even_if_identical_looking():
    # same appearance (twins / same jersey and no detail), but visible in the same frames
    mapping, _ = stitch([tr(1, 0, 50, 0), tr(2, 40, 90, 0)], CFG)
    assert mapping[1] != mapping[2]


def test_group_overlap_is_checked_after_merges():
    # 3 continues 1, but 3 overlaps 2 in time; 2 must not join the group of 1 once 3 is in it
    mapping, _ = stitch([tr(1, 0, 30, 0), tr(3, 60, 120, 0), tr(2, 40, 100, 0, noise=0.08)], CFG)
    assert mapping[3] == 1 and mapping[2] != 1


def test_ambiguous_candidate_is_refused():
    # a later tracklet that looks like two earlier ones (two team-mates, here the same vector)
    mapping, log = stitch([tr(1, 0, 30, 0), tr(2, 0, 30, 0), tr(3, 60, 90, 0)], CFG)
    assert mapping[3] == 3
    assert any(not e["merged"] and e["why"] == "ambiguous" for e in log)


def test_tracklets_without_enough_crops_are_left_alone():
    t = Tracklet(5, np.arange(80, 90), crops(0, n=2))
    mapping, _ = stitch([tr(1, 0, 50, 0), t], CFG)
    assert mapping[5] == 5


def test_max_gap():
    cfg = StitchConfig(thresh=0.3, margin=0.05, max_gap=20)
    mapping, _ = stitch([tr(1, 0, 50, 0), tr(2, 100, 150, 0)], cfg)
    assert mapping[2] == 2
