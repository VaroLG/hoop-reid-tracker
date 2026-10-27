"""016: team-mate ReID benchmark metrics."""

import numpy as np

from hooptrack.eval.reid_bench import auc, pair_distances, piece_level, pieces, rank1_among_teammates


def test_auc_perfect_chance_and_ties():
    assert auc([0.1, 0.2], [0.5, 0.6]) == 1.0
    assert auc([0.5, 0.6], [0.1, 0.2]) == 0.0
    assert auc([0.3, 0.3], [0.3, 0.3]) == 0.5


def _toy(sep_mate: float):
    """4 players, 2 per team, 10 crops each, one every 10 frames. Team-mates differ by `sep_mate`."""
    rng = np.random.default_rng(0)
    team_dir = {0: np.eye(8)[0], 1: np.eye(8)[1]}
    own = {p: np.eye(8)[2 + p] for p in range(4)}
    feats, pid, team, frame = [], [], [], []
    for p in range(4):
        t = p // 2
        for k in range(10):
            v = team_dir[t] + sep_mate * own[p] + 0.02 * rng.normal(size=8)
            feats.append(v / np.linalg.norm(v))
            pid.append(p)
            team.append(t)
            frame.append(10 * k)
    return np.array(feats), np.array(pid), np.array(team), np.array(frame)


def test_identical_teammates_are_at_chance_and_distinct_ones_are_separated():
    for sep, lo, hi in ((0.0, 0.3, 0.7), (1.0, 0.99, 1.01)):
        f, p, t, fr = _toy(sep)
        d = pair_distances(f, p, t, fr, min_gap=20)
        assert lo <= auc(d["same"], d["mate"]) <= hi
        assert auc(d["same"], d["opp"]) > 0.99            # other kit: always easy


def test_min_gap_removes_close_pairs():
    f, p, t, fr = _toy(1.0)
    d_all = pair_distances(f, p, t, fr, min_gap=0)
    d_far = pair_distances(f, p, t, fr, min_gap=50)
    assert len(d_far["same"]) < len(d_all["same"])


def test_rank1_and_pieces():
    f, p, t, fr = _toy(1.0)
    assert rank1_among_teammates(f, p, t, fr, min_gap=20) == 1.0
    groups = pieces(p, fr, length=30, min_crops=2)
    assert all(len(set(p[g])) == 1 for g in groups)         # a piece belongs to one player
    _, r1 = piece_level(f, p, t, fr, groups, min_gap=20)
    assert r1 == 1.0


def test_crossing_events_and_decisions():
    from hooptrack.eval.reid_bench import crossing_decisions, crossing_events

    # players 1 and 2 overlap in frames 10-14 and again in 40-41; player 3 is always alone
    gt = {}
    for f in range(60):
        x2 = 20 if (10 <= f <= 14 or 40 <= f <= 41) else 300
        gt[f] = (np.array([1, 2, 3]), np.array([[0, 0, 40, 100], [x2, 0, x2 + 40, 100], [600, 0, 640, 100]], float))
    assert crossing_events(gt) == [(1, 2, 10, 14), (1, 2, 40, 41)]
    # decisions: two clearly different players, crops every frame
    f, p, t, fr = _toy(1.0)
    ev = [(0, 1, 40, 45)]                  # players 0 and 1 (team-mates) "cross" around frames 40-45
    dec = crossing_decisions(f, p, t, fr, ev, window=30)
    assert len(dec) == 1 and dec[0]["kind"] == "mate" and dec[0]["keep"] < dec[0]["swap"]
