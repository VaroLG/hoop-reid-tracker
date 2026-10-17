import numpy as np

from hooptrack.config import apply_overrides
from hooptrack.data.sportsmot import load_mot_txt, write_mot_txt
from hooptrack.eval.metrics import reentry_recovery


def test_mot_roundtrip(tmp_path):
    rows = [(1, 3, np.array([10, 20, 50, 120]), 0.9), (2, 3, np.array([12, 20, 52, 120]), 0.8)]
    p = tmp_path / "seq.txt"
    write_mot_txt(p, rows)
    data = load_mot_txt(p)
    ids, boxes = data[1]
    assert ids.tolist() == [3]
    np.testing.assert_allclose(boxes[0], [10, 20, 50, 120])


def test_overrides_parse_types():
    cfg = apply_overrides({"tracker": {"gmc": "sparse"}},
                          ["tracker.gmc=none", "tracker.memory_enabled=false", "reid.batch_size=8"])
    assert cfg["tracker"]["gmc"] == "none"
    assert cfg["tracker"]["memory_enabled"] is False
    assert cfg["reid"]["batch_size"] == 8


def _frames(spec):
    """spec: frame -> list of (id, x)  ->  MOT-like dict."""
    return {f: (np.array([i for i, _ in v], dtype=int),
                np.array([[x, 0, x + 10, 30] for _, x in v], dtype=float).reshape(-1, 4))
            for f, v in spec.items()}


def test_reentry_metric_classifies_outcomes():
    # gt player 1 visible frames 1-5, gone 6-29, back 30-35. gt 2 always visible.
    gt = _frames({f: ([(1, 0)] if f <= 5 or f >= 30 else []) + [(2, 100)] for f in range(1, 36)})

    recovered = _frames({f: ([(7, 0)] if f <= 5 or f >= 30 else []) + [(8, 100)] for f in range(1, 36)})
    new_id = _frames({f: ([(7 if f <= 5 else 9, 0)] if f <= 5 or f >= 30 else []) + [(8, 100)]
                      for f in range(1, 36)})
    stolen = _frames({f: ([(7 if f <= 5 else 8, 0)] if f <= 5 or f >= 30 else []) + [(8, 100)]
                      for f in range(1, 36)})

    assert reentry_recovery(gt, recovered).recovered == 1
    assert reentry_recovery(gt, new_id).new_id == 1
    assert reentry_recovery(gt, stolen).stolen == 1
