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


# ------------------------------------------------------------------ GTA connector (stitch_gta)
from hooptrack.tracking.stitch import GTAConfig, stitch_gta  # noqa: E402


def trb(tid: int, start: int, end: int, player: int, x0: float = 100, x1: float = 100, **kw) -> Tracklet:
    """Tracklet whose box centre moves linearly from x0 to x1 (y fixed)."""
    frames = np.arange(start, end + 1)
    xs = np.linspace(x0, x1, len(frames))
    boxes = np.c_[xs - 20, np.full(len(frames), 300), xs + 20, np.full(len(frames), 400)]
    return Tracklet(tid, frames, crops(player, **kw), boxes)


GCFG = GTAConfig(thresh=0.3, spatial_factor=1.0)


def test_gta_merges_fragments_and_keeps_players_apart():
    trs = [trb(1, 0, 50, 0), trb(2, 0, 50, 1), trb(3, 80, 120, 0), trb(4, 80, 120, 1)]
    mapping, _ = stitch_gta(trs, GCFG)
    assert mapping == {1: 1, 2: 2, 3: 1, 4: 2}


def test_gta_never_merges_groups_on_screen_together():
    mapping, _ = stitch_gta([trb(1, 0, 50, 0), trb(2, 40, 90, 0)], GCFG)
    assert mapping[1] != mapping[2]


def test_gta_refuses_an_implausible_jump():
    # clip range 0..1000 px; factor 0.5 -> a 900 px jump between fragments is refused
    trs = [trb(1, 0, 50, 0, x0=0, x1=50), trb(2, 60, 100, 0, x0=950, x1=1000)]
    mapping, log = stitch_gta(trs, GTAConfig(thresh=0.3, spatial_factor=0.5))
    assert mapping[2] == 2 and log[0]["why"] == "spatial"


def test_gta_merge_is_hierarchical_with_mean_distance():
    # three fragments of one player: all end up in one group with the earliest id
    mapping, log = stitch_gta([trb(5, 100, 150, 2), trb(1, 0, 40, 2), trb(9, 200, 260, 2)], GCFG)
    assert set(mapping.values()) == {1} and sum(e["merged"] for e in log) == 2


# ------------------------------------------------------------------ 015: split at crossings
from hooptrack.tracking.stitch import split_at_crossings  # noqa: E402


def _box(x):
    return [x, 300, x + 40, 400]


def test_split_cuts_both_tracklets_when_a_crossing_ends():
    # ids 1 and 2 walk towards each other, overlap in frames 3-4, then separate
    xs1, xs2 = [0, 10, 38, 40, 42, 70, 80], [100, 90, 42, 40, 38, 10, 0]
    pred = {f: (np.array([1, 2]), np.array([_box(a), _box(b)], float)) for f, (a, b) in enumerate(zip(xs1, xs2))}
    seg = split_at_crossings(pred)
    assert seg[(0, 1)] == 1 and seg[(4, 1)] == 1          # crossing frames stay with the segment before
    assert seg[(5, 1)] not in (1, 2) and seg[(6, 1)] == seg[(5, 1)]
    assert seg[(5, 2)] not in (1, 2, seg[(5, 1)])


def test_split_leaves_lonely_tracklets_alone():
    pred = {f: (np.array([7]), np.array([_box(10 * f)], float)) for f in range(10)}
    assert set(split_at_crossings(pred).values()) == {7}


def test_split_does_not_cut_a_tracklet_that_starts_inside_a_crossing():
    # id 1 is born overlapping id 2 and only becomes clean later: nothing to cut yet
    xs1, xs2 = [40, 40, 0, 0], [45, 45, 200, 200]
    pred = {f: (np.array([1, 2]), np.array([_box(a), _box(b)], float)) for f, (a, b) in enumerate(zip(xs1, xs2))}
    assert {split_at_crossings(pred)[(f, 1)] for f in range(4)} == {1}
