"""Tests for the improvements introduced after experiment 001.

Both improvements are switchable and OFF by default: the first tests in each
section check that "off" reproduces the previous behaviour exactly.
"""

import cv2
import numpy as np

from hooptrack.detection.filters import CourtMask, CourtMaskConfig, build_filters
from hooptrack.pipeline import embed_needed
from hooptrack.tracking import TrackerConfig

# ----------------------------------------------------------------- 002 · ReID


class CountingReID:
    """Fake ReID: embedding = [x1, y1], and counts how many crops it saw."""

    def __init__(self):
        self.crops = 0

    def __call__(self, frame, boxes):
        boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
        self.crops += len(boxes)
        return boxes[:, :2].astype(np.float32) + 1.0


def test_embed_needed_skips_low_score_boxes():
    reid = CountingReID()
    boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 25], [9, 9, 19, 29]], dtype=float)
    scores = np.array([0.9, 0.2, 0.55])
    feats = embed_needed(reid, None, boxes, scores, TrackerConfig(high_thresh=0.5))
    assert reid.crops == 2                       # the 0.2 box was not embedded
    np.testing.assert_allclose(feats[0], [1, 1])
    np.testing.assert_allclose(feats[1], [0, 0])  # placeholder row, never read
    np.testing.assert_allclose(feats[2], [10, 10])


def test_embed_needed_uses_the_lower_of_both_thresholds():
    """If appearance may be LEARNT below high_thresh, those boxes must be embedded."""
    reid = CountingReID()
    boxes = np.array([[0, 0, 10, 20], [5, 5, 15, 25]], dtype=float)
    cfg = TrackerConfig(high_thresh=0.5, feature_update_min_score=0.3)
    embed_needed(reid, None, boxes, np.array([0.4, 0.2]), cfg)
    assert reid.crops == 1


def test_skipping_does_not_change_tracking():
    """Same detections + same appearance -> identical tracks with and without skipping."""
    from hooptrack.synthetic import SceneConfig, make_scene
    from hooptrack.tracking import ReIDTracker

    frames = make_scene(SceneConfig(n_players=8, n_frames=200, exits=[(1, 60, 90)],
                                    fp_rate=0.3, seed=4))

    class LookupReID:
        """Returns the precomputed embedding of each requested box."""

        def __init__(self, boxes, feats):
            self.table = {tuple(b): f for b, f in zip(boxes, feats, strict=True)}
            self.dim = feats.shape[1]

        def __call__(self, _frame, boxes):
            if len(boxes) == 0:
                return np.zeros((0, self.dim), np.float32)
            return np.stack([self.table[tuple(b)] for b in boxes])

    def run(skip: bool):
        cfg = TrackerConfig(gmc="none")
        trk = ReIDTracker(cfg)
        out = []
        for fr in frames:
            feats = fr.det_features.astype(np.float32)
            if skip:
                reid = LookupReID(fr.det_boxes, feats)
                feats = embed_needed(reid, None, fr.det_boxes, fr.det_scores, cfg)
            res = trk.update(fr.det_boxes, fr.det_scores, feats)
            out.append(sorted((o.track_id, *np.round(o.tlbr, 3)) for o in res))
        return out

    assert run(skip=False) == run(skip=True)


# ----------------------------------------------------------- 003 · court mask


def court_frame(h=720, w=1280):
    """Synthetic broadcast frame: dark crowd on top, hardwood trapezoid below."""
    img = np.full((h, w, 3), (40, 30, 30), np.uint8)              # dark stands
    court = np.array([[100, 250], [1180, 250], [1270, 710], [10, 710]], np.int32)
    cv2.fillPoly(img, [court], (60, 140, 200))                    # wood-like (BGR)
    cv2.line(img, (640, 250), (640, 710), (255, 255, 255), 3)     # court line
    return img, court


def test_build_filters_off_by_default():
    assert build_filters(None) == []
    assert build_filters({"court_mask": False}) == []
    assert len(build_filters({"court_mask": True})) == 1


def test_court_mask_keeps_players_and_drops_crowd():
    img, _ = court_frame()
    boxes = np.array([
        [400, 300, 450, 450],   # player in the middle of the court   -> keep
        [1100, 520, 1150, 690], # player near the sideline            -> keep
        [300, 40, 340, 140],    # spectator in the stands             -> drop
        [700, 60, 730, 170],    # spectator                           -> drop
    ], dtype=float)
    keep = CourtMask().keep_mask(img, boxes)
    assert keep.tolist() == [True, True, False, False]


def test_court_mask_tolerates_feet_just_over_the_line():
    img, _ = court_frame()
    # feet 10 px above the top line (250): within 15 % of a 150 px box
    keep = CourtMask(CourtMaskConfig(margin=0.15)).keep_mask(img, np.array([[600, 90, 650, 240.0]]))
    assert keep.tolist() == [True]


def test_court_mask_passes_everything_when_no_floor_is_visible():
    crowd_only = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    boxes = np.array([[10, 10, 50, 120], [600, 300, 650, 450]], dtype=float)
    mask = CourtMask()
    assert mask.keep_mask(crowd_only, boxes).tolist() == [True, True]
    assert mask.frames_passthrough == 1


def test_court_mask_reuses_last_good_hull_briefly():
    img, _ = court_frame()
    mask = CourtMask(CourtMaskConfig(reuse_frames=2))
    crowd = np.array([[300, 40, 340, 140]], dtype=float)
    assert mask.keep_mask(img, crowd).tolist() == [False]
    black = np.zeros_like(img)                                   # e.g. a cut to black
    assert mask.keep_mask(black, crowd).tolist() == [False]      # reused hull
    assert mask.keep_mask(black, crowd).tolist() == [False]
    assert mask.keep_mask(black, crowd).tolist() == [True]       # too long: pass-through


def test_court_mask_adaptive_margin():
    """008: the margin tightens only when this video's players clearly stay inside."""
    import numpy as np

    from hooptrack.detection.filters import CourtMask, CourtMaskConfig

    hull = np.array([[[0, 100]], [[1000, 100]], [[1000, 700]], [[0, 700]]], dtype=np.int32)
    inside = np.array([[100, 300, 140, 400.0]] * 10)        # feet well inside the floor
    cfg = CourtMaskConfig(adaptive_margin=True, adaptive_after=3, adaptive_min_samples=20)
    m = CourtMask(cfg)
    for _ in range(3):
        m.frame += 1
        m.last_hull = hull
        m.observe_players(inside)
    assert m.adapted and m.margin < 0                      # stricter than the default tolerance
    loose = CourtMask(cfg)
    edge = np.array([[100, 0, 140, 90.0]] * 10)             # players with feet just outside the floor
    for _ in range(3):
        loose.frame += 1
        loose.last_hull = hull
        loose.observe_players(edge)
    assert loose.adapted and loose.margin == cfg.margin     # never looser than the configured margin
    off = CourtMask(CourtMaskConfig())
    off.last_hull = hull
    off.observe_players(inside)
    assert not off.adapted and off.margin == 0.15
