"""C05: camera model, plausibility, and the annotation-free court search (synthetic scenes)."""

import cv2
import numpy as np

from hooptrack.court.camera import Camera, camera_from_homography
from hooptrack.court.leagues import FIBA, NBA, NCAA_PRE2019, lines
from hooptrack.court.propagate import disagreement
from hooptrack.court.refine import line_evidence
from hooptrack.court.search import explained, plausible, search_lines

SPEC = NBA


def _broadcast(spec=SPEC, target=(21.0, 7.0), focal=1700.0, back=18.0, height=12.0, x=None):
    """A broadcast-like camera: behind the near sideline, above the floor."""
    pos = (spec.length / 2 if x is None else x, spec.width + back, height)
    return Camera(pos, target, focal)


def _render(H, spec=SPEC):
    img = np.full((720, 1280, 3), 175, np.uint8)
    for pl in lines(spec):
        p = cv2.perspectiveTransform(pl[None].astype(np.float64), H)[0]
        cv2.polylines(img, [np.clip(p, -1e4, 1e4).astype(np.int32)], False, (40, 40, 40), 3)
    return img


def test_camera_roundtrip():
    """Homography from a camera, and the camera read back from the homography."""
    for cam in (_broadcast(), _broadcast(target=(SPEC.length / 2, 7.0), x=SPEC.length / 2),   # straight across
                _broadcast(target=(4.0, 9.0), focal=1100.0, back=9.0, height=7.0)):
        est = camera_from_homography(cam.homography())
        assert abs(est["focal"] - cam.focal) / cam.focal < 1e-6
        assert np.allclose(est["position"], cam.position, atol=1e-4)


def test_camera_view_is_not_mirrored():
    """Court x grows to the right of the image, the far sideline (y = 0) is above the near one."""
    H = _broadcast(target=(SPEC.length / 2, 7.0), x=SPEC.length / 2).homography()
    p = cv2.perspectiveTransform(np.array([[[0, 7.6], [SPEC.length, 7.6], [14, 0], [14, 15.24]]]), H)[0]
    assert p[0, 0] < p[1, 0] and p[2, 1] < p[3, 1]


def test_plausible_rejects_squashed_and_mirrored_courts():
    H = _broadcast().homography()
    assert plausible(H, SPEC, (1280, 720))
    mirror_y = H @ np.array([[1, 0, 0], [0, -1, SPEC.width], [0, 0, 1.0]])         # seen from the far side
    assert not plausible(mirror_y, SPEC, (1280, 720))
    tiny = np.diag([0.02, 0.02, 1.0]) @ H                                            # court squashed to a dot
    assert not plausible(tiny, SPEC, (1280, 720))


def test_squashing_fools_a_per_sample_count_but_not_explained_nor_plausible():
    """All court lines piled on the far sideline: every sample 'matches', but only one line is explained."""
    from hooptrack.court.refine import sample_lines
    from hooptrack.court.search import batch_scores
    H = _broadcast().homography()
    ev = line_evidence(_render(H))
    S, T = sample_lines(lines(SPEC))
    squash = H @ np.array([[1, 0, 0], [0, 0.01, 0], [0, 0, 1.0]])                    # court width -> 15 cm
    assert batch_scores(squash[None], S, T, ev)[0] > batch_scores(H[None], S, T, ev)[0]   # fooled
    assert explained(H, S, T, ev) > explained(squash, S, T, ev)                           # not fooled
    assert not plausible(squash, SPEC, (1280, 720))


def test_search_lines_finds_the_court_without_annotation():
    for spec, target in ((NBA, (21.0, 7.0)), (FIBA, (7.0, 7.5)), (NCAA_PRE2019, (14.0, 7.0))):
        H = _broadcast(spec, target).homography()
        ev = line_evidence(_render(H, spec))
        cands = search_lines(ev, spec)
        assert cands, spec.name
        px, m = disagreement(cands[0].H, H, spec)
        assert m < 0.15, (spec.name, m)
