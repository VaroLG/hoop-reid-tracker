"""Experiment 006: team classifier + team gate."""

import numpy as np

from hooptrack.detection.teams import OTHER, UNKNOWN, TeamClassifier, TeamClassifierConfig, build_team_classifier
from hooptrack.tracking import TrackerConfig
from hooptrack.tracking.track import Track
from hooptrack.tracking.tracker import ReIDTracker

WHITE, BLUE, STRIPES = (240, 240, 240), (200, 60, 20), None


def _frame(boxes, colours):
    """Synthetic frame with each box painted: white, saturated blue, or b/w stripes."""
    img = np.full((400, 1000, 3), 90, np.uint8)
    for (x1, y1, x2, y2), c in zip(boxes.astype(int), colours, strict=True):
        if c is None:
            for k, x in enumerate(range(x1, x2, 4)):
                img[y1:y2, x:x + 4] = 250 if k % 2 else 10
        else:
            img[y1:y2, x1:x2] = c
    return img


def _scene():
    xs = np.arange(12) * 80 + 10
    boxes = np.stack([xs, np.full(12, 50), xs + 60, np.full(12, 250)], 1).astype(float)
    colours = [WHITE] * 5 + [BLUE] * 5 + [STRIPES] * 2       # 5 + 5 players, 2 referees
    return boxes, colours


def test_disabled_by_default():
    assert build_team_classifier(None) is None
    assert build_team_classifier({"enabled": False}) is None


def test_classifier_separates_teams_and_referees():
    boxes, colours = _scene()
    frame, scores = _frame(boxes, colours), np.full(len(boxes), 0.9)
    clf = TeamClassifier(TeamClassifierConfig(warmup_frames=10, min_samples=30))
    for _ in range(10):
        assert (clf(frame, boxes, scores) == UNKNOWN).all()   # warm-up: no labels
    assert clf.reliable
    lab = clf(frame, boxes, scores)
    assert len(set(lab[:5])) == 1 and len(set(lab[5:10])) == 1 and lab[0] != lab[5]
    assert set(lab[:10]) <= {0, 1}
    assert (lab[10:] == OTHER).all()
    assert lab[0] == 0                                         # teams numbered by whiteness


def test_track_majority_vote():
    t = Track(np.array([0, 0, 10, 20.0]), 0.9, None, 1)
    assert t.team() == -1
    for lab in (1, 1, 0, 1):
        t.add_team_vote(lab)
    t.add_team_vote(-1)                                        # unknown labels do not vote
    assert t.team(min_votes=3) == 1
    t.add_team_vote(0)
    t.add_team_vote(0)
    assert t.team(min_votes=3, min_share=0.6) == -1            # 3 vs 3: no majority


def _run(cfg, frames):
    tr = ReIDTracker(cfg)
    out = []
    for boxes, teams in frames:
        out.append(tr.update(np.array(boxes, float), np.full(len(boxes), 0.9), None, None, teams=np.array(teams)))
    return out


def test_team_gate_blocks_cross_team_jump():
    # Player of team 0 tracked for 10 frames, then disappears; a team-1 box appears
    # nearby with ambiguous overlap. Without the gate it continues the id.
    base = [([[100, 100, 140, 200]], [0])] * 10
    jump = [([[125, 100, 165, 200]], [1])] * 3                  # IoU ~0.45 < 0.5
    off = _run(TrackerConfig(gmc="none"), base + jump)
    on = _run(TrackerConfig(gmc="none", team_gate=True), base + jump)
    first = off[5][0].track_id
    assert off[-1][0].track_id == first                        # stolen by the other team
    assert all(o.track_id != first for o in on[-1])            # gate: a new identity


def test_team_gate_keeps_clear_motion_match():
    # A wrong label on a box with high IoU must not break the track.
    frames = [([[100, 100, 140, 200]], [0])] * 10 + [([[102, 100, 142, 200]], [1])]
    on = _run(TrackerConfig(gmc="none", team_gate=True), frames)
    assert on[-1][0].track_id == on[5][0].track_id


def test_suppress_other_hides_referees():
    frames = [([[100, 100, 140, 200], [400, 100, 440, 200]], [0, OTHER])] * 10
    off = _run(TrackerConfig(gmc="none"), frames)
    on = _run(TrackerConfig(gmc="none", suppress_other=True), frames)
    assert len(off[-1]) == 2 and len(on[-1]) == 1


def test_occluded_box_labelled_from_visible_torso():
    from hooptrack.detection.teams import torso_colour_features
    boxes, colours = _scene()
    boxes = boxes + 0.4                     # non-integer coordinates, as from a detector
    frame = _frame(boxes, colours)
    # a blue player (in front: lower bottom edge) covers the left 40 % of a white player
    front = np.array([[10, 60, 34, 260.0]])
    frame[60:260, 10:34] = BLUE
    f_all = torso_colour_features(frame, boxes[:1])
    f_vis = torso_colour_features(frame, boxes[:1], occluders=np.vstack([boxes[:1], front]))
    f_self = torso_colour_features(frame, boxes[:1], occluders=boxes)   # itself + players behind
    assert np.allclose(f_self, f_all)
    assert f_vis[0, 0] > f_all[0, 0]          # whiter once the blue occluder is masked out
    assert f_vis[0, 0] > 0.95
    hidden = np.array([[0, 60, 80, 260.0]])   # fully covered -> cannot be labelled
    assert np.isnan(torso_colour_features(frame, boxes[:1], occluders=hidden)).all()


def test_team_cost_prefers_same_team_after_separation():
    # Two players merge (one box) then separate; the team-0 track must follow the team-0 box
    # even though the team-1 box overlaps it MORE. A lone wrong label never cuts the track.
    a0 = [([[100, 100, 140, 200]], [0])] * 10
    split = [([[104, 100, 144, 200], [90, 100, 130, 200]], [1, 0])] * 2
    soft = _run(TrackerConfig(gmc="none", team_cost=0.4), a0 + split)
    tid = soft[5][0].track_id
    box = [o.tlbr for o in soft[-1] if o.track_id == tid][0]
    assert box[0] < 100                                       # followed the team-0 box (left)
    lone = _run(TrackerConfig(gmc="none", team_cost=0.4), a0 + [([[102, 100, 142, 200]], [1])])
    assert lone[-1][0].track_id == tid


def test_early_fit_labels_sooner_then_refits():
    boxes, colours = _scene()
    frame, scores = _frame(boxes, colours), np.full(len(boxes), 0.9)
    clf = TeamClassifier(TeamClassifierConfig(warmup_frames=20, early_fit_frames=3, min_samples=30))
    labels = [clf(frame, boxes, scores) for _ in range(25)]
    assert (labels[2] == UNKNOWN).all()                 # frame 3: model fitted, labels from frame 4
    assert (labels[3][10:] == OTHER).all()              # early model already labels referees
    assert clf._final and clf.reliable                  # refit at frame 20 kept a reliable model
    assert (labels[-1] == labels[3]).all()


def test_default_warmup_unchanged():
    """early_fit_frames=0 must reproduce 006: no label before warmup_frames + 1."""
    boxes, colours = _scene()
    frame, scores = _frame(boxes, colours), np.full(len(boxes), 0.9)
    clf = TeamClassifier(TeamClassifierConfig(warmup_frames=10, min_samples=30))
    labels = [clf(frame, boxes, scores) for _ in range(12)]
    assert all((lab == UNKNOWN).all() for lab in labels[:10]) and (labels[10] >= 0).all()


def test_probation_hides_unlabelled_new_tracks():
    unknown = [([[100, 100, 140, 200]], [-1])] * 8
    labelled = [([[100, 100, 140, 200]], [0])] * 8
    cfg = TrackerConfig(gmc="none", team_probation=5)
    out_u = _run(cfg, unknown)
    out_l = _run(cfg, labelled)
    assert [len(o) for o in out_u] == [0, 0, 0, 0, 0, 1, 1, 1]      # shown once 5 frames old
    assert len(out_l[3]) == 1                                        # a team -> shown as soon as confirmed
    # no team labels passed at all (teams disabled) -> probation never applies
    tr = ReIDTracker(cfg)
    outs = [tr.update(np.array([[100, 100, 140, 200.0]]), np.array([0.9])) for _ in range(3)]
    assert len(outs[1]) == 1


def test_final_team_recorded_for_offline_filter():
    frames = [([[100, 100, 140, 200], [400, 100, 440, 200]], [-1, -1])] * 3 \
        + [([[100, 100, 140, 200], [400, 100, 440, 200]], [0, OTHER])] * 5
    tr = ReIDTracker(TrackerConfig(gmc="none"))
    early = None
    for k, (boxes, teams) in enumerate(frames):
        out = tr.update(np.array(boxes, float), np.full(2, 0.9), teams=np.array(teams))
        if k == 2:
            early = {o.track_id for o in out}
    assert len(early) == 2                                   # referee visible before its votes
    assert sorted(tr.final_team.values()) == [0, OTHER]      # ... but the offline filter can drop it


def test_classify_is_stateless():
    boxes, colours = _scene()
    frame, scores = _frame(boxes, colours), np.full(len(boxes), 0.9)
    clf = TeamClassifier(TeamClassifierConfig(warmup_frames=5, min_samples=30))
    assert (clf.classify(frame, boxes) == UNKNOWN).all()      # no model yet
    for _ in range(6):
        clf(frame, boxes, scores)
    n = clf.frame
    lab = clf.classify(frame, boxes[10:], others=boxes[:10])
    assert (lab == OTHER).all() and clf.frame == n
