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


def test_gta_forbidden_pairs_stay_forbidden_above_threshold_one():
    """029: a ReID model with another distance scale may need thresh > 1. Pieces on screen together, of
    different teams, or after a refused jump must still never merge (they used to be at distance 1.0)."""
    loose = GTAConfig(thresh=1.5, spatial_factor=1.0)
    mapping, _ = stitch_gta([trb(1, 0, 50, 0), trb(2, 40, 90, 0)], loose)              # on screen together
    assert mapping[1] != mapping[2]
    mapping, _ = stitch_gta([trb(1, 0, 50, 0), trb(2, 60, 90, 0)], loose, teams={1: 0, 2: 1})   # two kits
    assert mapping[2] == 2
    trs = [trb(1, 0, 50, 0, x0=0, x1=50), trb(2, 60, 100, 0, x0=950, x1=1000)]
    mapping, _ = stitch_gta(trs, GTAConfig(thresh=1.5, spatial_factor=0.5))              # implausible jump
    assert mapping[2] == 2


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
    pred = {f: (np.array([1, 2]), np.array([_box(a), _box(b)], float))
            for f, (a, b) in enumerate(zip(xs1, xs2, strict=True))}
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
    pred = {f: (np.array([1, 2]), np.array([_box(a), _box(b)], float))
            for f, (a, b) in enumerate(zip(xs1, xs2, strict=True))}
    assert {split_at_crossings(pred)[(f, 1)] for f in range(4)} == {1}


# ------------------------------------------------------------------ 019: split at team changes
from hooptrack.tracking.stitch import split_at_team_changes  # noqa: E402


def _walk(n, x0=0, step=0):
    return [[x0 + step * f, 300, x0 + step * f + 40, 400] for f in range(n)]


def test_team_split_cuts_a_stolen_id_at_the_end_of_the_crossing():
    # id 1: team 0 for frames 0-39, crossing with id 2 at frames 40-44, team 1 from frame 45 on
    pred = {}
    for f in range(100):
        b1 = [200, 300, 240, 400]
        b2 = [210, 300, 250, 400] if 40 <= f <= 44 else [600, 300, 640, 400]
        pred[f] = (np.array([1, 2]), np.array([b1, b2], float))
    labels = {(f, 1): (0 if f < 40 else 1) for f in range(100) if not 40 <= f <= 44}
    seg = split_at_team_changes(pred, labels, min_run=25)
    assert seg[(39, 1)] == 1 and seg[(44, 1)] == 1          # the crossing stays with the old player
    assert seg[(45, 1)] not in (1, 2) and seg[(99, 1)] == seg[(45, 1)]


def test_team_split_ignores_short_runs_of_wrong_labels():
    pred = {f: (np.array([5]), np.array([[10, 300, 50, 400]], float)) for f in range(200)}
    labels = {(f, 5): 0 for f in range(200)}
    for f in range(80, 90):                                   # 10 wrong labels in a row: classifier noise
        labels[(f, 5)] = 1
    for f in range(120, 125):
        labels[(f, 5)] = -1                                   # unknown labels are skipped
    assert set(split_at_team_changes(pred, labels, min_run=25).values()) == {5}


def test_team_split_without_labels_changes_nothing():
    pred = {f: (np.array([3, 4]), np.array(_walk(1, 0) + _walk(1, 500), float)) for f in range(50)}
    seg = split_at_team_changes(pred, {}, min_run=25)
    assert {seg[(f, 3)] for f in range(50)} == {3} and {seg[(f, 4)] for f in range(50)} == {4}


def test_gta_team_rule_never_merges_two_kits():
    """021: with teams known, the closest pair is refused when the kits differ; unknown teams stay free."""
    from hooptrack.tracking.stitch import GTAConfig, stitch_gta

    def trb(tid, start, end, player):
        t = tr(tid, start, end, player)
        t.boxes = np.tile([100.0, 100.0, 140.0, 200.0], (len(t.frames), 1))
        return t

    a, b = trb(1, 0, 50, 0), trb(2, 80, 120, 0)              # same appearance
    cfg = GTAConfig(thresh=0.4)
    assert stitch_gta([a, b], cfg)[0] == {1: 1, 2: 1}
    assert stitch_gta([a, b], cfg, teams={1: 0, 2: 1})[0] == {1: 1, 2: 2}
    assert stitch_gta([a, b], cfg, teams={1: 0})[0] == {1: 1, 2: 1}


def test_gta_number_rule_separates_team_mates():
    """023: two pieces that look alike but carry different jersey numbers are never joined."""
    from hooptrack.tracking.stitch import GTAConfig, stitch_gta

    def trb(tid, start, end, player):
        t = tr(tid, start, end, player)
        t.boxes = np.tile([100.0, 100.0, 140.0, 200.0], (len(t.frames), 1))
        return t

    a, b, c = trb(1, 0, 50, 0), trb(2, 80, 120, 0), trb(3, 150, 200, 0)
    cfg = GTAConfig(thresh=0.4)
    mapping, _ = stitch_gta([a, b, c], cfg, numbers={1: "23", 2: "7", 3: "23"})
    assert mapping[2] != mapping[1] and mapping[3] == mapping[1]


def test_tracklet_number_needs_agreeing_confident_reads():
    from hooptrack.reid.jersey import tracklet_number

    assert tracklet_number([("23", 0.9), ("23", 0.8), ("28", 0.75)]) == "23"
    assert tracklet_number([("23", 0.9)]) is None                         # one read is not enough
    assert tracklet_number([("23", 0.9), ("28", 0.9)]) is None            # no clear majority
    assert tracklet_number([("23", 0.9), ("23", 0.5), ("CO", 0.99)]) is None  # weak or non-digit reads do not vote


def test_partial_number_reads_do_not_conflict_when_allowed():
    from hooptrack.tracking.stitch import conflicting_numbers

    assert conflicting_numbers("3", "23") and not conflicting_numbers("3", "23", substring_ok=True)
    assert conflicting_numbers("7", "23", substring_ok=True)
    assert not conflicting_numbers(None, "23") and not conflicting_numbers("23", "23")


# ------------------------------------------------------------------ 035: speed rule on the court
def with_ground(t: Tracklet, x0: float, x1: float) -> Tracklet:
    """Feet on the court moving linearly from (x0, 7) to (x1, 7) metres over the tracklet."""
    t.ground = np.c_[np.linspace(x0, x1, len(t.frames)), np.full(len(t.frames), 7.0)]
    return t


SPEED = GTAConfig(thresh=0.3, spatial_factor=1.0, max_speed=9.0, speed_slack=1.0, fps=25.0)


def test_speed_rule_refuses_an_impossible_run_on_the_court():
    # same look, same screen position, but 20 m apart on the court after 10 frames (0.4 s): 50 m/s
    trs = [with_ground(trb(1, 0, 50, 0), 2, 5), with_ground(trb(2, 60, 100, 0), 25, 27)]
    mapping, log = stitch_gta(trs, SPEED)
    assert mapping[2] == 2 and log[0]["why"] == "speed"
    for t in trs:
        t.ground = None
    mapping, _ = stitch_gta(trs, GCFG)                       # rule off: the old behaviour, merged
    assert mapping[2] == 1


def test_speed_rule_allows_a_possible_run_and_unknown_positions():
    # 6 m in 1 s (25 frames) is a sprint, allowed
    trs = [with_ground(trb(1, 0, 50, 0), 2, 5), with_ground(trb(2, 75, 100, 0), 11, 12)]
    assert stitch_gta(trs, SPEED)[0][2] == 1
    # the later fragment has no court at all: nothing to check, merged
    trs[1].ground = np.full((len(trs[1].frames), 2), np.nan)
    assert stitch_gta(trs, SPEED)[0][2] == 1


def test_speed_rule_uses_the_last_and_first_known_positions():
    # the end of the first fragment has no court; its last known feet (frame 40, x = 5) are used:
    # 20 m in 20 frames (0.8 s) = 25 m/s -> refused
    a = with_ground(trb(1, 0, 50, 0), 2, 5)
    a.ground[41:] = np.nan
    a.ground[40] = (5, 7)
    b = with_ground(trb(2, 60, 100, 0), 25, 27)
    mapping, log = stitch_gta([a, b], SPEED)
    assert mapping[2] == 2 and log[0]["why"] == "speed"


# ------------------------------------------------------------------ 036: roster of 10 players
from hooptrack.tracking.stitch import assign_roster  # noqa: E402


def test_roster_gives_a_fragment_to_its_player_and_respects_time_and_teams():
    # team 0: players 0 and 1 (long), a fragment of player 1 later; team 1: player 2, and a fragment that
    # looks like player 2 but belongs to the other kit
    trs = [tr(1, 0, 100, 0), tr(2, 0, 100, 1), tr(3, 150, 170, 1), tr(4, 0, 100, 2), tr(5, 120, 140, 2)]
    mapping = {t.tid: t.tid for t in trs}
    teams = {1: 0, 2: 0, 3: 0, 4: 1, 5: 0}
    final, leftovers, log = assign_roster(trs, mapping, teams, per_team=2)
    assert final[3] == 2                                  # the fragment goes back to its own player
    assert final[5] in (1, 2)                             # team 0 only, even if it looks like player 2 (team 1)
    assert not leftovers


def test_roster_never_gives_two_simultaneous_fragments_to_one_player():
    # one roster player per team; two fragments of team 0 on screen together: only one can be him
    trs = [tr(1, 0, 100, 0), tr(2, 0, 100, 1), tr(3, 150, 170, 0), tr(4, 150, 170, 0)]
    final, leftovers, _ = assign_roster(trs, {t.tid: t.tid for t in trs}, {1: 0, 2: 1, 3: 0, 4: 0}, per_team=1)
    assert sorted([final[3], final[4]]).count(1) == 1 and len(leftovers) == 1


# ------------------------------------------------------------------ 037: closed set of players
from hooptrack.tracking.stitch import assign_closed_set  # noqa: E402


def test_closed_set_seeds_on_a_shared_frame_and_gives_pieces_back():
    # 3 players on screen together in 0-50; after a crossing, pieces of each come back (60-120)
    trs = [tr(1, 0, 50, 0), tr(2, 0, 50, 1), tr(3, 0, 50, 2), tr(4, 60, 120, 2), tr(5, 60, 120, 0), tr(6, 60, 90, 1)]
    mapping, leftovers, _ = assign_closed_set(trs, n_players=3)
    assert mapping[4] == 3 and mapping[5] == 1 and mapping[6] == 2 and not leftovers


def test_closed_set_keeps_an_eleventh_person_apart():
    # 2 players; a third person on screen with both of them can be neither
    trs = [tr(1, 0, 100, 0), tr(2, 0, 100, 1), tr(3, 20, 40, 0)]
    mapping, leftovers, _ = assign_closed_set(trs, n_players=2)
    assert mapping[3] == 3 and leftovers == {3}


def test_closed_set_leaves_pieces_without_crops_alone():
    trs = [tr(1, 0, 100, 0), tr(2, 0, 100, 1), Tracklet(3, np.arange(120, 130), np.zeros((0, 1), np.float32))]
    mapping, leftovers, _ = assign_closed_set(trs, n_players=2)
    assert mapping[3] == 3 and leftovers == {3}


def test_continuity_keeps_the_online_identity_unless_appearance_clearly_disagrees():
    # players 0 and 1 on screen together (seeds); later piece 3 continues online tracklet of seed 1 but
    # looks a bit more like player 1 (seed 2): without continuity it goes to 2, with a large bonus it stays with 1
    trs = [tr(1, 0, 50, 0), tr(2, 0, 50, 1), Tracklet(3, np.arange(60, 90), crops(1, noise=0.6))]
    m0, _, _ = assign_closed_set(trs, n_players=2)
    m1, _, _ = assign_closed_set(trs, n_players=2, links={3: {1}, 1: {3}}, continuity=2.0)
    assert m0[3] == 2 and m1[3] == 1


from hooptrack.tracking.stitch import smooth_sandwiches  # noqa: E402


def test_smoothing_gives_a_sandwiched_piece_back_and_respects_time():
    trs = [tr(1, 0, 50, 0), tr(2, 55, 60, 0), tr(3, 65, 120, 0), tr(4, 0, 120, 1)]
    links = {1: {2}, 2: {1, 3}, 3: {2}}
    m, log = smooth_sandwiches(trs, {1: 1, 2: 4, 3: 1, 4: 4}, links)
    assert m[2] == 1 and len(log) == 1
    # if player 1 already has a box in the middle piece's frames, nothing moves
    trs.append(tr(5, 56, 58, 0))
    m, log = smooth_sandwiches(trs, {1: 1, 2: 4, 3: 1, 4: 4, 5: 1}, links)
    assert m[2] == 4 and not log


def test_refine_closed_set_moves_piece_and_respects_overlap():
    """043: a piece wrongly given to A moves to B when B is free; two overlapping pieces never share a player."""
    from hooptrack.tracking.stitch import Tracklet, refine_closed_set

    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    box = np.zeros((10, 4))
    pa = Tracklet(1, np.arange(0, 10), np.tile(a, (3, 1)), box)       # seed A
    pb = Tracklet(2, np.arange(0, 10), np.tile(b, (3, 1)), box)       # seed B
    wrong = Tracklet(3, np.arange(20, 30), np.tile(b, (3, 1)), box)   # looks like B, given to A
    other = Tracklet(4, np.arange(20, 30), np.tile(a, (3, 1)), box)   # overlaps piece 3, looks like A
    mapping = {1: 1, 2: 2, 3: 1, 4: 4}
    new, _ = refine_closed_set([pa, pb, wrong, other], mapping, players=[1, 2])
    assert new[3] == 2 and new[4] == 1
