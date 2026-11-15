"""scripts/run_pipeline.py: the stage alternatives and their output folders, and the web app's command."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline as rp  # noqa: E402


def test_final_selection_keeps_the_plain_folder_names():
    """The default of every stage is the final selection: no suffix, so earlier outputs are reused."""
    assert rp.tag(**{k: v[0] for k, v in rp.STAGES.items()}) == ""


def test_alternatives_get_their_own_folders():
    assert rp.tag(detector="coco", court_mask="segmentation", reid="none") == "__coco-none"
    assert rp.tag(ball="yolo") == "__yolo"


def test_device_is_passed_through_unless_auto():
    assert rp.resolve_device("cpu") == "cpu"
    assert rp.resolve_device("cuda:1") == "cuda:1"
    assert rp.resolve_device("auto") in ("cpu", "cuda")


def test_app_builds_the_pipeline_command():
    """The app always runs the final selection: it passes no stage alternative, only the video and the court."""
    pytest.importorskip("gradio")
    sys.path.insert(0, str(ROOT / "app"))
    import app

    cmd = app.pipeline_cmd("game.mp4", "g1", 0, 30, "FIBA", "hall", "cpu")
    s = " ".join(cmd)
    assert "--no-prompt" in cmd and "--end 30" in s and "--venue hall" in s and "--court FIBA" in s
    assert not any(f"--{k.replace('_', '-')}" in cmd for k in rp.STAGES)
    assert "--court" not in app.pipeline_cmd("game.mp4", "g1", 0, None, "none", "", "auto")


def test_app_passes_team_names_and_colours():
    pytest.importorskip("gradio")
    sys.path.insert(0, str(ROOT / "app"))
    import app

    job = app.make_job("g.mp4", "g1", 0, None, "FIBA", "", "cpu", "Mamba", "rgba(242, 169, 0, 1)", "Rivals",
                       "#3b5bdb", True, True)
    s = " ".join(app.job_cmd(job))
    assert "--team-names Mamba,Rivals" in s and "--team-colours #f2a900,#3b5bdb" in s and "--swap-teams" in s
    plain = app.make_job("g.mp4", "g1", 0, None, "FIBA", "", "cpu")
    assert "--team" not in " ".join(app.job_cmd(plain)) and "--swap" not in " ".join(app.job_cmd(plain))


def test_team_colours_go_to_the_kit_they_resemble():
    pytest.importorskip("PIL")
    sys.path.insert(0, str(ROOT / "scripts" / "possession"))
    import p06_render_demo as p06

    yellow, purple = (0, 170, 240), (90, 30, 60)                       # BGR kits of kit groups 0 and 1
    centres = [p06.lab(yellow), p06.lab(purple)]
    # the user gives the purple team first: it must land on kit group 1
    colours, names, order = p06.team_style(centres, [yellow, purple], ["Rivals", "Mamba"],
                                           [p06.hex_bgr("#3b1f5c"), p06.hex_bgr("#f2a900")], swap=False)
    assert names == ["Mamba", "Rivals"] and order == [1, 0]
    assert p06.initials(["Mamba", "Rivals"]) == ["M", "R"]
    assert p06.initials(["Real", "Rayo"]) == ["RE", "RA"]
    # two kits too alike get clearly different colours; the holder's colour differs from both teams
    orange, brownish = (41, 106, 170), (80, 92, 170)
    a, b = p06.distinct([orange, brownish])
    assert a == orange and b != brownish
    assert p06.possession_colour([a, b]) not in (a, b)
