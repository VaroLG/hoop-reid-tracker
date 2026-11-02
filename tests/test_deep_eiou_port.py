"""026: Deep-EIoU's fusion rule and "last detected box" association, ported into ReIDTracker."""

import numpy as np
import pytest

from hooptrack.reid.cache import CachedReID
from hooptrack.tracking.matching import fuse_min
from hooptrack.tracking.tracker import ReIDTracker, TrackerConfig


# ------------------------------------------------------------------ fuse_min
def test_fuse_min_takes_the_better_of_motion_and_trusted_appearance():
    iou = np.array([[0.6, 0.6, 0.3]])
    app = np.array([[0.1, 0.9, 0.1]])
    cost = fuse_min(iou, app, app_gate=0.5, proximity=0.5)
    # close (1 - IoU = 0.4 <= 0.5) and similar -> appearance wins
    assert cost[0, 0] == pytest.approx(0.1)
    # close but NOT similar -> appearance ignored, motion decides (no veto)
    assert cost[0, 1] == pytest.approx(0.4)
    # similar but NOT close (1 - IoU = 0.7 > 0.5) -> appearance ignored
    assert cost[0, 2] == pytest.approx(0.7)


def test_fuse_min_nan_appearance_is_no_evidence():
    cost = fuse_min(np.array([[0.7]]), np.array([[np.nan]]), 0.5, 0.5)
    assert cost[0, 0] == pytest.approx(0.3)
    assert fuse_min(np.array([[0.7]]), None, 0.5, 0.5)[0, 0] == pytest.approx(0.3)


# ------------------------------------------------------------------ tracker options
def _run(cfg: TrackerConfig, frames: int = 30):
    """Two players walking; the second stops and turns back (a constant-velocity prediction overshoots)."""
    trk, out = ReIDTracker(cfg), []
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(2, 16))
    for f in range(frames):
        x2 = 400 + 8 * min(f, 15) - 8 * max(0, f - 15)
        boxes = np.array([[100 + 4 * f, 100, 150 + 4 * f, 220], [x2, 120, x2 + 50, 240]], dtype=float)
        o = trk.update(boxes, np.array([0.9, 0.9]), feats + 0.01 * rng.normal(size=feats.shape))
        out.append(sorted((t.track_id, *np.round(t.tlbr, 3)) for t in o))
    return out


def test_defaults_are_the_previous_tracker():
    base = TrackerConfig(gmc="none")
    same = TrackerConfig(gmc="none", fusion="weighted", assoc_box="kalman")
    assert _run(base) == _run(same)


@pytest.mark.parametrize("fusion,box", [("min", "kalman"), ("weighted", "last"), ("min", "last"), ("min", "last_gmc")])
def test_new_modes_keep_two_identities(fusion, box):
    out = _run(TrackerConfig(gmc="none", fusion=fusion, assoc_box=box, eiou_expand=0.7, eiou_iterations=2))
    ids = {row[0] for frame in out for row in frame}
    assert ids == {1, 2}


def test_bad_option_is_refused():
    with pytest.raises(ValueError):
        TrackerConfig.from_dict({"fusion": "sum"})
    with pytest.raises(ValueError):
        TrackerConfig.from_dict({"assoc_box": "predicted"})


def test_last_box_follows_the_detection():
    trk = ReIDTracker(TrackerConfig(gmc="none", assoc_box="last"))
    trk.update(np.array([[0, 0, 10, 20.0]]), np.array([0.9]))
    trk.update(np.array([[5, 0, 15, 20.0]]), np.array([0.9]))
    assert np.allclose(trk.tracked[0].last_tlbr, [5, 0, 15, 20])


def test_last_box_is_warped_by_the_camera_only_in_last_gmc():
    from hooptrack.tracking.track import Track

    shift = np.array([[1.0, 0, 7], [0, 1.0, 0]])
    a = Track(np.array([0, 0, 10, 20.0]), 0.9, None, 1)
    b = Track(np.array([0, 0, 10, 20.0]), 0.9, None, 1)
    a.apply_camera_motion(shift)
    b.apply_camera_motion(shift, warp_last=True)
    assert np.allclose(a.last_tlbr, [0, 0, 10, 20])
    assert np.allclose(b.last_tlbr, [7, 0, 17, 20])


# ------------------------------------------------------------------ embedding cache
def test_cached_reid_reads_known_boxes_and_computes_the_rest(tmp_path):
    det = tmp_path / "dets"
    emb = tmp_path / "emb"
    det.mkdir()
    emb.mkdir()
    (det / "clip.txt").write_text("1,-1,10.00,20.00,30.00,40.00,0.9\n1,-1,100.00,20.00,30.00,40.00,0.8\n")
    saved = np.eye(2, 4, dtype=np.float32)
    np.savez(emb / "clip.npz", **{"1": saved})
    calls = []

    def base(frame, boxes):
        calls.append(len(boxes))
        return np.full((len(boxes), 4), 0.5, np.float32)

    reid = CachedReID(base, emb, det)
    boxes = np.array([[10, 20, 40, 60], [100, 20, 130, 60], [300, 0, 310, 10]], dtype=float)
    out = reid(None, boxes)
    assert np.allclose(out[:2], saved)              # from the cache
    assert np.allclose(out[2], 0.5)                 # computed
    assert calls == [1] and (reid.hits, reid.misses) == (2, 1)
