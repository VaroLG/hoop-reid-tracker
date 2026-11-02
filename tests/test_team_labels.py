"""020: unsupervised team labels from any box descriptor (tracklet means -> k-means -> nearest centre)."""

import numpy as np

from hooptrack.tracking.team_labels import OTHER, cluster_team_labels, torso_boxes

RNG = np.random.default_rng(0)
D = 16
KITS = {k: v / np.linalg.norm(v) for k, v in enumerate(RNG.normal(size=(3, D)))}


def boxes_of(kit: int, tid: int, n: int = 10, noise: float = 0.1, start: int = 0):
    """n labelled boxes of one tracklet wearing ``kit``: (keys, vectors)."""
    x = KITS[kit] + noise * RNG.normal(size=(n, D))
    return [(start + 5 * i, tid) for i in range(n)], x.astype(np.float32)


def clip(players: dict[int, int], n: int = 10, **kw):
    keys, feats = [], []
    for tid, kit in players.items():
        k, f = boxes_of(kit, tid, n, **kw)
        keys += k
        feats.append(f)
    return keys, np.concatenate(feats)


def same_partition(labels: dict, players: dict[int, int]) -> bool:
    """All boxes of one kit share one label, and different kits get different labels."""
    by_kit = {}
    for (_, tid), lab in labels.items():
        by_kit.setdefault(players[tid], set()).add(lab)
    return all(len(v) == 1 for v in by_kit.values()) and len({next(iter(v)) for v in by_kit.values()}) == len(by_kit)


def test_two_teams_are_separated_with_k2():
    players = {t: t % 2 for t in range(10)}                 # 5 + 5 players, two kits
    labels = cluster_team_labels(*clip(players), k=2)
    assert len(labels) == 100 and same_partition(labels, players)


def test_k3_puts_the_smallest_group_in_other():
    players = {**{t: t % 2 for t in range(10)}, 10: 2, 11: 2}   # + 2 referees in a third kit
    labels = cluster_team_labels(*clip(players), k=3)
    assert same_partition(labels, players)
    assert {lab for (_, t), lab in labels.items() if t >= 10} == {OTHER}


def test_k3_without_referees_turns_players_into_other():
    """020's diagnosis: with only players on screen, the third cluster is carved out of a team."""
    players = {t: t % 2 for t in range(10)}
    keys, feats = clip(players, noise=0.3)
    labels = cluster_team_labels(keys, feats, k=3)
    assert OTHER in set(labels.values())


def test_rejection_drops_ambiguous_boxes_only():
    players = {t: t % 2 for t in range(10)}
    keys, feats = clip(players)
    mid = (KITS[0] + KITS[1]) / 2                            # a box halfway between the two kits
    keys, feats = keys + [(999, 0)], np.vstack([feats, mid[None]]).astype(np.float32)
    labels = cluster_team_labels(keys, feats, k=2, reject_ratio=0.7)
    assert (999, 0) not in labels
    assert len(labels) == 100


def test_short_tracklets_are_labelled_but_do_not_vote():
    players = {t: t % 2 for t in range(10)}
    keys, feats = clip(players)
    k, f = boxes_of(1, 50, n=2)                               # 2 boxes only: below min_boxes = 5
    labels = cluster_team_labels(keys + k, np.vstack([feats, f]), k=2)
    assert labels[k[0]] == labels[(0, 1)]                    # same team as the other kit-1 tracklets


def test_torso_box_is_inside_the_player_box():
    t = torso_boxes(np.array([[100, 200, 140, 300]]))[0]
    assert np.allclose(t, [110, 220, 130, 250])
