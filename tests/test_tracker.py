"""Behavioural tests: the scenarios the tracker was designed for."""

import numpy as np

from hooptrack.eval.metrics import clear_and_identity, reentry_recovery
from hooptrack.synthetic import SceneConfig, make_scene
from hooptrack.tracking import ReIDTracker, TrackerConfig


def run(frames, cfg: TrackerConfig, use_app: bool = True):
    trk = ReIDTracker(cfg)
    gt, pred = {}, {}
    for f, fr in enumerate(frames, start=1):
        out = trk.update(fr.det_boxes, fr.det_scores, fr.det_features if use_app else None)
        gt[f] = (fr.gt_ids, fr.gt_boxes)
        pred[f] = (np.array([o.track_id for o in out], dtype=int),
                   np.array([o.tlbr for o in out]).reshape(-1, 4))
    return trk, gt, pred


def box(cx, cy, w=40, h=100):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_single_player_keeps_id():
    trk = ReIDTracker(TrackerConfig(gmc="none"))
    ids = set()
    for t in range(50):
        out = trk.update(np.array([box(100 + 3 * t, 300)]), np.array([0.9]))
        ids |= {o.track_id for o in out}
    assert ids == {1}


def test_low_score_detection_keeps_track_alive():
    """ByteTrack stage 2: an occluded (low-score) box still updates the track."""
    trk = ReIDTracker(TrackerConfig(gmc="none"))
    for t in range(10):
        trk.update(np.array([box(100 + 3 * t, 300)]), np.array([0.9]))
    out = trk.update(np.array([box(130, 300)]), np.array([0.3]))   # below high_thresh
    assert [o.track_id for o in out] == [1]


def test_crossing_teammates_do_not_swap():
    """Two players with similar (but distinct) appearance cross paths."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=32)
    fa = base + 0.5 * rng.normal(size=32)
    fb = base + 0.5 * rng.normal(size=32)
    trk = ReIDTracker(TrackerConfig(gmc="none"))
    history = {1: [], 2: []}
    for t in range(60):
        xa, xb = 100 + 6 * t, 460 - 6 * t            # they meet at t=30
        out = trk.update(
            np.array([box(xa, 300), box(xb, 305)]), np.array([0.9, 0.9]),
            np.stack([fa, fb]) + 0.05 * rng.normal(size=(2, 32)),
        )
        for o in out:
            history[o.track_id].append((o.tlbr[0] + o.tlbr[2]) / 2)
    # id 1 started on the left and must end on the right (it kept moving +x)
    assert history[1][-1] > history[1][0]
    assert history[2][-1] < history[2][0]


def test_memory_recovers_identity_after_long_exit():
    frames = make_scene(SceneConfig(n_players=6, n_frames=260, exits=[(2, 80, 150)], seed=3))
    cfg = TrackerConfig(gmc="none", max_time_lost=30)          # exit >> max_time_lost
    _, gt, pred_mem = run(frames, cfg)
    _, _, pred_nomem = run(frames, TrackerConfig(gmc="none", max_time_lost=30, memory_enabled=False))

    r_mem = reentry_recovery(gt, pred_mem)
    r_nomem = reentry_recovery(gt, pred_nomem)
    assert r_mem.events >= 1
    assert r_mem.rate == 1.0
    assert r_nomem.rate == 0.0          # without memory the player gets a new id
    assert r_mem.stolen == 0


def test_full_tracker_beats_motion_only_on_idf1():
    frames = make_scene(SceneConfig(n_players=10, n_frames=300,
                                    exits=[(0, 100, 80), (3, 150, 120), (6, 60, 90)], seed=1))
    _, gt, pred_full = run(frames, TrackerConfig(gmc="none"))
    _, _, pred_motion = run(frames, TrackerConfig(gmc="none", memory_enabled=False), use_app=False)
    idf1_full = clear_and_identity({"s": (gt, pred_full)}).loc["OVERALL", "idf1"]
    idf1_motion = clear_and_identity({"s": (gt, pred_motion)}).loc["OVERALL", "idf1"]
    assert idf1_full > idf1_motion
    assert idf1_full > 0.95


def test_reset_restarts_ids():
    trk = ReIDTracker(TrackerConfig(gmc="none"))
    trk.update(np.array([box(100, 100)]), np.array([0.9]))
    trk.reset()
    out = trk.update(np.array([box(500, 100)]), np.array([0.9]))
    assert out[0].track_id == 1
