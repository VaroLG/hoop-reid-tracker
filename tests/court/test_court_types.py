import json

import numpy as np
import pytest

from hooptrack.court.leagues import SPECS, get_spec, keypoints, random_spec, spec_from_npz, spec_json


def test_preset_and_custom_yaml(tmp_path):
    assert get_spec("FIBA") is SPECS["FIBA"]
    p = tmp_path / "my_court.yaml"
    p.write_text("length: 26\nwidth: 14\nhoop_x: 1.6\nlane_w: 4\nft_x: 5.8\nthree_r: 6.5\n"
                 "three_side: 1.0\nmark_x: 8\ncircle_r: 1.8\n")
    s = get_spec(p)
    assert s.name == "my_court" and s.lane_w == 4 and s.lane_ft == 4
    assert keypoints(s)["37"] == (26.0, 5.0)                         # lane corner on the far baseline
    with pytest.raises(ValueError):
        get_spec("NOT_A_COURT")


def test_custom_measurements_survive_the_npz(tmp_path):
    s = random_spec(np.random.default_rng(3))
    np.savez(tmp_path / "h.npz", league=s.name, spec=spec_json(s))
    assert spec_from_npz(np.load(tmp_path / "h.npz")) == s
    np.savez(tmp_path / "old.npz", league="NCAA")                     # files written before: preset by name
    assert spec_from_npz(np.load(tmp_path / "old.npz")) is SPECS["NCAA"]


def test_random_specs_vary_and_are_drawable():
    rng = np.random.default_rng(0)
    specs = [random_spec(rng) for _ in range(200)]
    lanes = np.array([s.lane_w for s in specs])
    assert lanes.min() < 3.7 and lanes.max() > 5.5                    # narrow NCAA-like to wide trapezoids
    assert any(s.lane_w_ft is not None for s in specs)
    for s in specs:
        json.loads(spec_json(s))
        assert s.hoop_x + s.three_r < s.length / 2 - s.circle_r        # 3-pt arc stays in its half
        assert s.ft_x + s.circle_r < s.length / 2
