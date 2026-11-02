"""030: identity check at crossing exits, on a synthetic crossing (two players of different colours)."""

from types import SimpleNamespace

import numpy as np

from hooptrack.tracking.crossing_check import CrossingCheck, CrossingCheckConfig, swap_identity
from hooptrack.tracking.tracker import TrackOutput

RED, BLUE = (0, 0, 220), (220, 0, 0)


def colour_reid(frame, boxes):
    """Toy ReID: the mean colour of the crop. Red and blue players are far apart, as with a good model."""
    out = []
    for x1, y1, x2, y2 in np.asarray(boxes, int):
        out.append(frame[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0) + 1.0)
    out = np.array(out, float)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_track(tid):
    return SimpleNamespace(track_id=tid, frame_id=0, smooth_feat=np.full(3, tid), gallery=[tid],
                           team_votes=np.array([tid, 0, 0]), start_frame=tid)


def run(swap_by_tracker: bool, frames: int = 80, cross=(30, 40)):
    """Player RED walks right, BLUE walks left; they overlap during ``cross``. Track 1 starts on RED, track 2 on
    BLUE. If ``swap_by_tracker``, the (simulated) tracker exchanges them at the end of the crossing."""
    check = CrossingCheck(CrossingCheckConfig(enabled=True, min_height=10, margin=0.0), colour_reid)
    t1, t2 = make_track(1), make_track(2)
    tracker = SimpleNamespace(tracked=[t1, t2], frame_id=0)
    shown_ids = []
    for f in range(1, frames + 1):
        tracker.frame_id = f
        t1.frame_id = t2.frame_id = f
        # x positions: they meet in the middle during the crossing, then go on
        xr = 100 + 6 * f
        xb = 600 - 6 * f
        frame = np.zeros((300, 900, 3), np.uint8)
        frame[100:200, xr:xr + 40] = RED
        frame[100:200, xb:xb + 40] = BLUE
        red_box, blue_box = [xr, 100, xr + 40, 200], [xb, 100, xb + 40, 200]
        # which track object is on which player
        if swap_by_tracker and f > cross[1]:
            boxes = {id(t1): blue_box, id(t2): red_box}
        else:
            boxes = {id(t1): red_box, id(t2): blue_box}
        outputs = [TrackOutput(t.track_id, np.array(boxes[id(t)], float), 0.9) for t in (t1, t2)]
        outputs = check.step(frame, f, tracker, outputs)
        red_id = next(o.track_id for o in outputs if list(o.tlbr) == red_box)
        shown_ids.append(red_id)
    return shown_ids, check


def test_a_swap_made_by_the_tracker_is_undone():
    ids, check = run(swap_by_tracker=True)
    assert ids[0] == 1 and ids[-1] == 1                  # RED is called 1 again at the end
    assert [e["outcome"] for e in check.log] == ["swap"]


def test_no_swap_when_the_tracker_was_right():
    ids, check = run(swap_by_tracker=False)
    assert set(ids) == {1}
    assert [e["outcome"] for e in check.log] == ["keep"]


def test_swap_identity_exchanges_who_not_where():
    a, b = make_track(1), make_track(2)
    a.mean, b.mean = "kalman A", "kalman B"
    swap_identity(a, b)
    assert (a.track_id, b.track_id) == (2, 1) and list(a.gallery) == [2]
    assert (a.mean, b.mean) == ("kalman A", "kalman B")   # the motion state stays with the object


# ------------------------------------------------------------------ 031: groups of any size
from hooptrack.tracking.crossing_check import GroupCrossingCheck  # noqa: E402

GREEN = (0, 200, 0)


def run_group(colours, paths, assign, frames=90, visible=None):
    """Players of ``colours`` walk along ``paths`` (f -> x). ``assign(f)`` = which player each track object is on
    at frame f (the simulated tracker, possibly wrong). ``visible(f, k)`` hides a track object."""
    cfg = CrossingCheckConfig(enabled=True, mode="group", min_height=10, margin=0.0)
    check = GroupCrossingCheck(cfg, colour_reid)
    tracks = [make_track(k + 1) for k in range(len(colours))]
    tracker = SimpleNamespace(tracked=tracks, frame_id=0)
    for f in range(1, frames + 1):
        tracker.frame_id = f
        frame = np.zeros((300, 1200, 3), np.uint8)
        boxes = []
        for col, path in zip(colours, paths, strict=True):
            x = int(path(f))
            frame[100:200, x:x + 40] = col
            boxes.append([x, 100, x + 40, 200])
        outputs = []
        for k, t in enumerate(tracks):
            if visible is None or visible(f, k):
                t.frame_id = f
                outputs.append(TrackOutput(t.track_id, np.array(boxes[assign(f)[k]], float), 0.9))
        outputs = check.step(frame, f, tracker, outputs)
    # which id does each player carry at the end?
    final = {}
    for o in outputs:
        final[next(p for p, b in enumerate(boxes) if list(o.tlbr) == b)] = o.track_id
    return final, check


def converge(x0):
    """Three players meet at x = 500 around frame 40, then go back where they came from."""
    return lambda f: x0 + (500 - x0) * (1 - abs(f - 40) / 40) if f < 80 else x0


def test_group_undoes_a_three_player_rotation():
    paths = [converge(100), converge(500), converge(900)]
    final, check = run_group([RED, BLUE, GREEN], paths,
                             assign=lambda f: [0, 1, 2] if f <= 40 else [1, 2, 0])   # tracker rotates them
    assert final == {0: 1, 1: 2, 2: 3}                       # each player has his own id again
    assert [e["outcome"] for e in check.log if e["outcome"] != "not_decided"] == ["reassign"]


def test_group_does_nothing_when_the_tracker_was_right():
    paths = [converge(100), converge(500), converge(900)]
    final, check = run_group([RED, BLUE, GREEN], paths, assign=lambda f: [0, 1, 2])
    assert final == {0: 1, 1: 2, 2: 3}
    assert "reassign" not in {e["outcome"] for e in check.log}


def test_group_gives_a_lost_players_id_to_the_track_that_took_his_place():
    # RED and BLUE cross; the tracker's track 1 continues on BLUE, track 2 (BLUE's) is lost after the crossing
    paths = [converge(100), converge(900)]
    final, check = run_group([RED, BLUE], paths, assign=lambda f: [0, 1] if f <= 40 else [1, 0],
                             visible=lambda f, k: not (k == 1 and f > 40))
    assert final == {1: 2}                                   # BLUE carries his own id 2 again
