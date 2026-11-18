"""HoopTrack web app: upload a basketball video, run the final pipeline on it, and get the annotated video back.

    python app/app.py                 # then open http://127.0.0.1:7860

What it does (a thin layer over ``scripts/run_pipeline.py``, which does all the work):

1. **Check the court** — you upload a video and give the court type, an optional venue (a fixed camera spot whose
   court is reused for all its clips) and CPU / GPU. Only the frames and the court are computed (a few minutes),
   and a preview shows the court drawn over 3 frames: you know whether the video will have a minimap BEFORE the
   long run.
2. **Court** — when the court cannot be found automatically (and the venue has no court yet), this tab shows the
   3 frames with the most court in view. You click the crossings of the court lines on each; the court is drawn
   over the frame as you go, so a wrong click is visible at once. Saving computes the court from your clicks and
   shows its preview. With a venue, your clicks become its fixed court.
3. **Process the full video** — players, identities, ball, possession and the video, reusing the court. Every stage
   is the final selection of the README, the one with the best metrics (the alternatives of the experiments stay
   in ``run_pipeline.py``). The log is streamed to the page.

Nothing leaves the computer: the app is served locally and the videos stay in ``data/`` and ``outputs/``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]                 # the repository
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "scripts" / "court")]

from annotate_anchors import CLEAR  # noqa: E402  (the 18 points where two painted lines cross)

from hooptrack.court.anchors import homography_from_points  # noqa: E402
from hooptrack.court.leagues import SPECS, keypoints, lines  # noqa: E402

OUT = ROOT / "outputs" / "pipeline"
CLIPS = ROOT / "data" / "videos" / "clips"
VENUES = ROOT / "data" / "venues"
COURTS = ["none", "FIBA", "NBA", "NCAA", "FIBA_PRE2010", "auto"]
SOURCE = "https://github.com/VaroLG/hoop-reid-tracker"          # AGPL-3.0 section 13: users of a service get the source


# ------------------------------------------------------------------------------------------------- 1 · run
def pipeline_cmd(video: str | None, name: str, start: float, end: float | None, court: str, venue: str,
                 device: str, anchors: Path | None = None, court_only: bool = False,
                 teams: dict | None = None) -> list[str]:
    """The ``run_pipeline.py`` command: the final selection of every stage (its defaults), the showcase video.
    ``--no-prompt``: when the court cannot be found, it is asked for in the page, not in a desktop window."""
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_pipeline.py"), "--video", str(video), "--name", name,
           "--start", str(start or 0), "--device", device, "--no-prompt"]
    if end:
        cmd += ["--end", str(end)]
    if court and court != "none":
        cmd += ["--court", court]
    if venue:
        cmd += ["--venue", venue]
    if anchors is not None:
        cmd += ["--anchors", str(anchors)]
    if court_only:
        cmd += ["--court-only"]
    teams = teams or {}
    if teams.get("names"):
        cmd += ["--team-names", teams["names"]]
    if teams.get("colours"):
        cmd += ["--team-colours", teams["colours"]]
    if teams.get("swap"):
        cmd += ["--swap-teams"]
    return cmd


def stream(cmd: list[str]):
    """Run a command and yield its growing log (the last lines; tqdm's carriage returns become new lines)."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    log: list[str] = []
    for line in proc.stdout:
        log.extend(s for s in line.replace("\r", "\n").splitlines() if s.strip())
        yield "\n".join(log[-300:]), None
    yield "\n".join(log[-300:]), proc.wait()


def read_status(name: str) -> dict:
    path = OUT / name / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summary(status: dict) -> str:
    if not status:
        return "The pipeline stopped before the end: see the log."
    court = {"automatic": "found automatically", "anchors": "from the hand anchors",
             "not_found": "**not found**: annotate it in the *Court* tab",
             "not_matched": "the venue's court does not match this clip's framing (no minimap)",
             "none": "not computed (no court type given)"}.get(status.get("court"), status.get("court"))
    minutes = sum(status.get("minutes", {}).values())
    return f"**Done** on `{status.get('device')}` in {minutes:.1f} min of computation · court: {court}"


def make_job(video, name, start, end, court, venue, device, name1="", colour1="", name2="", colour2="",
             use_colours=False, swap=False) -> dict:
    if not video:
        raise gr.Error("Upload a video first.")
    name = (name or Path(video).stem).strip().replace(" ", "_")
    names = f"{(name1 or 'A').strip().replace(',', ' ')},{(name2 or 'B').strip().replace(',', ' ')}"
    teams = {"names": names if (name1 or name2) else None,
             "colours": f"{hex_colour(colour1)},{hex_colour(colour2)}" if use_colours else None, "swap": swap}
    return {"video": video, "name": name, "start": start, "end": end, "court": court,
            "venue": (venue or "").strip(), "device": device, "teams": teams}


def hex_colour(value: str) -> str:
    """Gradio's colour picker may give '#rrggbb' or 'rgba(r, g, b, a)': always '#rrggbb'."""
    value = (value or "#ffffff").strip()
    if value.startswith("rgb"):
        r, g, b = (int(float(x)) for x in value[value.index("(") + 1:value.index(")")].split(",")[:3])
        return f"#{r:02x}{g:02x}{b:02x}"
    return value[:7]


def job_cmd(job: dict, **kw) -> list[str]:
    return pipeline_cmd(job["video"], job["name"], job["start"], job["end"], job["court"], job["venue"],
                        job["device"], teams=job.get("teams"), **kw)


def court_message(status: dict) -> str:
    return {"automatic": "✅ **Court found automatically.** Check the preview, then process the full video.",
            "anchors": "✅ **Court from the hand-clicked points.** Check the preview, then process the full video.",
            "not_found": "⚠️ **The court was not found automatically.** Click it in the *2 · Court* tab "
                         "(or process the video without a minimap).",
            "not_matched": "⚠️ **The venue's court does not fit this clip** (another framing or a strong zoom): "
                           "the video will have no minimap.",
            "none": "No court type given: the video will have no minimap."}.get(status.get("court"), "")


def check_court(video, name, start, end, court, venue, device):
    """Button 1: frames + court only, then the preview, or the frames to click when the court was not found."""
    job = make_job(video, name, start, end, court, venue, device)
    for log, code in stream(job_cmd(job, court_only=True)):
        if code is None:
            yield log, "Looking for the court…", None, job, gr.update()
    status = read_status(job["name"])
    text = court_message(status) if code == 0 else f"**Failed** (exit code {code}): see the log."
    tab = gr.update(selected="court") if status.get("court") == "not_found" else gr.update()
    yield log, text, status.get("preview"), job, tab


def run_full(video, name, start, end, court, venue, device, name1, colour1, name2, colour2, use_colours, swap,
             last: dict):
    """Button 2: the whole pipeline (a court checked before is reused, clicked points included). Run again with
    other team names / colours: only the video is redrawn."""
    job = make_job(video, name, start, end, court, venue, device, name1, colour1, name2, colour2, use_colours, swap)
    anchors = last.get("anchors") if last and last.get("name") == job["name"] else None
    job["anchors"] = anchors
    for log, code in stream(job_cmd(job, anchors=Path(anchors) if anchors else None)):
        if code is None:
            yield log, "Processing the full video… (on a CPU, about 1 h per 40 s of video)", None, job
    status = read_status(job["name"])
    text = summary(status) if code == 0 else f"**The pipeline failed** (exit code {code}): see the log."
    yield log, text, status.get("video"), job


# ----------------------------------------------------------------------------------------------- 2 · court
def court_plan(league: str, placed: set[str], selected: str, scale: float = 22.0, pad: int = 30) -> np.ndarray:
    """The court plan with the 18 clickable points: green = placed on this frame, red = the one to place now."""
    spec = SPECS[league]
    w, h = int(spec.length * scale) + 2 * pad, int(spec.width * scale) + 2 * pad
    img = np.full((h, w, 3), (40, 45, 55), np.uint8)

    def px(p):
        return int(pad + p[0] * scale), int(pad + p[1] * scale)

    for pl in lines(spec):
        cv2.polylines(img, [np.array([px(p) for p in pl], np.int32)], False, (200, 205, 215), 2, cv2.LINE_AA)
    for k, p in keypoints(spec).items():
        if k not in CLEAR:
            continue
        colour = (60, 60, 230) if k == selected else (80, 200, 80) if k in placed else (150, 150, 150)
        cv2.circle(img, px(p), 9 if k == selected else 6, colour, -1, cv2.LINE_AA)
        cv2.putText(img, k, (px(p)[0] + 8, px(p)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def frame_view(ann: dict) -> tuple[np.ndarray, str]:
    """The current frame with its clicked points and, from 4 points on, the court they imply."""
    f = ann["frames"][ann["i"]]
    img = cv2.imread(str(CLIPS / ann["clip"] / "img1" / f"{f:06d}.jpg"))
    pts = ann["points"].get(str(f), {})
    info = f"Frame {f} ({ann['i'] + 1} of {len(ann['frames'])}) · {len(pts)} points"
    if len(pts) >= 4:
        spec = SPECS[ann["league"]]
        H = homography_from_points({k: tuple(v) for k, v in pts.items()}, spec)
        for pl in lines(spec):
            q = cv2.perspectiveTransform(pl[None].astype(np.float64), H)[0].astype(np.int32)
            cv2.polylines(img, [q], False, (0, 230, 255), 2, cv2.LINE_AA)
        kp = keypoints(spec)
        err = [float(np.linalg.norm(cv2.perspectiveTransform(np.array([[kp[k]]], np.float64), H)[0, 0] - v))
               for k, v in pts.items()]
        info += f" · court drawn from them (point error: median {np.median(err):.1f} px, worst {max(err):.1f} px)"
        info += " — if the yellow lines are not on the white ones, a point is wrong" if max(err) > 8 else ""
    for k, (x, y) in pts.items():
        cv2.circle(img, (int(x), int(y)), 7, (60, 60, 230), 2, cv2.LINE_AA)
        cv2.putText(img, k, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), info


def court_outputs(ann: dict):
    """Everything the Court tab shows, from the annotation state."""
    if not ann:
        return None, None, "Run a video first: this tab is used when its court cannot be found.", gr.update(), ann
    f = str(ann["frames"][ann["i"]])
    frame, info = frame_view(ann)
    plan = court_plan(ann["league"], set(ann["points"].get(f, {})), ann["selected"])
    return frame, plan, info, gr.update(value=ann["selected"]), ann


def start_annotation(job: dict):
    """Fill the Court tab with the frames the pipeline suggested (the most court in view, one per third)."""
    status = read_status(job["name"]) if job else {}
    if not status.get("suggested_frames"):
        return court_outputs({})
    league = job["court"] if job["court"] in SPECS else "FIBA"
    names = sorted(k for k in keypoints(SPECS[league]) if k in CLEAR)
    ann = {"clip": job["name"], "league": league, "frames": status["suggested_frames"], "i": 0, "points": {},
           "selected": names[0], "names": names}
    return court_outputs(ann)


def next_unplaced(ann: dict) -> str:
    """After a click, the next point of the list that is not placed on this frame yet."""
    placed = ann["points"].get(str(ann["frames"][ann["i"]]), {})
    order = ann["names"][ann["names"].index(ann["selected"]) + 1:] + ann["names"]
    return next((k for k in order if k not in placed), ann["selected"])


def click_frame(ann: dict, evt: gr.SelectData):
    """A click on the frame places the selected point there (Gradio gives the click in image pixels)."""
    if not ann:
        return court_outputs(ann)
    x, y = evt.index
    ann["points"].setdefault(str(ann["frames"][ann["i"]]), {})[ann["selected"]] = [float(x), float(y)]
    ann["selected"] = next_unplaced(ann)
    return court_outputs(ann)


def click_plan(ann: dict, evt: gr.SelectData, scale: float = 22.0, pad: int = 30):
    """A click on the court plan selects the nearest point."""
    if not ann:
        return court_outputs(ann)
    x, y = evt.index
    kp = keypoints(SPECS[ann["league"]])
    ann["selected"] = min(ann["names"],
                          key=lambda k: (pad + kp[k][0] * scale - x) ** 2 + (pad + kp[k][1] * scale - y) ** 2)
    return court_outputs(ann)


def select_point(ann: dict, name: str):
    if ann and name:
        ann["selected"] = name
    return court_outputs(ann)


def undo(ann: dict):
    pts = ann["points"].get(str(ann["frames"][ann["i"]]), {}) if ann else {}
    if pts:
        last = list(pts)[-1]
        del pts[last]
        ann["selected"] = last
    return court_outputs(ann)


def step_frame(ann: dict, delta: int):
    """Previous / next of the 3 frames."""
    if ann:
        ann["i"] = int(np.clip(ann["i"] + delta, 0, len(ann["frames"]) - 1))
    return court_outputs(ann)


def shift_frame(ann: dict, delta: int):
    """Replace this frame by one ``delta`` frames away (when it shows too few lines). Its points are dropped:
    they belong to this image, and the camera moves."""
    if ann:
        n = len(list((CLIPS / ann["clip"] / "img1").glob("*.jpg")))
        old = ann["frames"][ann["i"]]
        new = int(np.clip(old + delta, 1, n))
        ann["points"].pop(str(old), None)
        ann["frames"][ann["i"]] = new
    return court_outputs(ann)


def save_court(ann: dict, job: dict):
    """Write the clicks as an anchors file and compute the court from them (court only), with its preview."""
    if not ann or not job:
        raise gr.Error("Nothing to save.")
    anchors = [{"frame": int(f), "points": {k: [round(x, 2), round(y, 2)] for k, (x, y) in sorted(p.items())},
                "note": "clicked in the web app"} for f, p in sorted(ann["points"].items(), key=lambda kv: int(kv[0]))
               if len(p) >= 4]
    if not anchors:
        raise gr.Error("Place at least 4 points on at least one frame (better: 5 or more on each of the 3).")
    path = OUT / ann["clip"] / "court" / "anchors_clicked" / f"anchors_{ann['clip']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sequence": ann["clip"], "league": ann["league"], "anchors": anchors}, indent=1))
    job = dict(job, anchors=str(path))                              # the full run will use these points
    for log, code in stream(job_cmd(job, anchors=path, court_only=True)):
        if code is None:
            yield log, "Computing the court from your points…", None, job
    status = read_status(job["name"])
    text = court_message(status) if code == 0 else f"**Failed** (exit code {code}): see the log."
    yield log, text + "  Back to *1 · Video* to process it.", status.get("preview"), job


# -------------------------------------------------------------------------------------------------- layout
def venues() -> list[str]:
    return sorted(p.name for p in VENUES.glob("*") if (p / "anchors.json").is_file()) if VENUES.is_dir() else []


def build() -> gr.Blocks:
    with gr.Blocks(title="HoopTrack") as demo:
        job, ann = gr.State({}), gr.State({})
        gr.Markdown("# 🏀 HoopTrack\nPlayers, identities, court, ball and possession on a basketball video, "
                    "with the final selection of the project: the combination with the best metrics. "
                    f"Source code (AGPL-3.0): [{SOURCE}]({SOURCE}) · non-commercial use: the models are trained on "
                    "non-commercial data (see docs/data_models_licences.md). Upload only videos you may use, of "
                    "people who agreed to it.")
        with gr.Tabs() as tabs:
            with gr.Tab("1 · Video", id="run"), gr.Row():
                with gr.Column(scale=1):
                    video = gr.Video(label="Video", sources=["upload"])
                    name = gr.Textbox(label="Name (folder under outputs/pipeline/)", placeholder="my_game")
                    with gr.Row():
                        start = gr.Number(label="From (s)", value=0, minimum=0)
                        end = gr.Number(label="To (s, empty = end)", value=None, minimum=0)
                    court = gr.Dropdown(COURTS, value="FIBA", label="Court type (none = no minimap)")
                    venue = gr.Dropdown(venues(), value=None, allow_custom_value=True,
                                        label="Venue (optional): same camera spot = same court for all its clips")
                    device = gr.Dropdown(["auto", "cpu", "cuda"], value="auto",
                                         label="Device (auto = NVIDIA GPU if any: then SAM 2 keeps the ids)")
                    with gr.Accordion("Teams in the video (optional)", open=False):
                        gr.Markdown("Names and colours of the two teams. Each colour goes to the team whose kit it "
                                    "resembles most (pick something close to each kit); if they come out the wrong "
                                    "way round, tick *swap* and process again: only the video is redrawn.")
                        with gr.Row():
                            name1 = gr.Textbox(label="Team 1", placeholder="Home")
                            colour1 = gr.ColorPicker(label="Colour 1", value="#f2a900")
                        with gr.Row():
                            name2 = gr.Textbox(label="Team 2", placeholder="Away")
                            colour2 = gr.ColorPicker(label="Colour 2", value="#3b5bdb")
                        with gr.Row():
                            use_colours = gr.Checkbox(label="Use these colours (else: the kits' own)", value=False)
                            swap = gr.Checkbox(label="Swap the teams", value=False)
                    check_btn = gr.Button("1 · Check the court (a few minutes)")
                    run_btn = gr.Button("2 · Process the full video", variant="primary")
                with gr.Column(scale=1):
                    status_md = gr.Markdown("Upload a video and **check the court** first: you will know whether "
                                            "the video gets a minimap before the long run (on a CPU, about 1 h "
                                            "per 40 s of video, mostly the player detector).")
                    preview = gr.Image(label="Court preview", interactive=False)
                    result = gr.Video(label="Result", interactive=False)
                    log = gr.Textbox(label="Log", lines=14, max_lines=14, autoscroll=True)
            with gr.Tab("2 · Court", id="court"):
                gr.Markdown("Used when the court is not found automatically. On each of the 3 frames, pick a point "
                            "on the plan (or in the list) and click where it is on the frame: only crossings of the "
                            "**white basketball lines**, at least 5 per frame, spread out. From 4 points the court "
                            "is drawn in yellow: it must lie on the white lines. If a frame shows too little court, "
                            "move it ±10 frames.")
                with gr.Row():
                    with gr.Column(scale=3):
                        frame_img = gr.Image(label="Frame: click to place the selected point", interactive=False)
                        frame_info = gr.Markdown()
                    with gr.Column(scale=2):
                        plan_img = gr.Image(label="Court plan: click a point to select it", interactive=False)
                        point = gr.Dropdown(sorted(CLEAR), label="Point to place")
                        with gr.Row():
                            prev_b, next_b = gr.Button("◀ Previous frame"), gr.Button("Next frame ▶")
                        with gr.Row():
                            back_b, fwd_b = gr.Button("Frame −10"), gr.Button("Frame +10")
                        undo_b = gr.Button("Undo last point")
                        save_b = gr.Button("Save and compute the court", variant="primary")
                court_status = gr.Markdown()
                court_prev = gr.Image(label="Court preview", interactive=False)
                court_log = gr.Textbox(label="Log", lines=10, max_lines=10, autoscroll=True)

        outs = [frame_img, plan_img, frame_info, point, ann]
        inputs = [video, name, start, end, court, venue, device]
        check_btn.click(check_court, inputs, [log, status_md, preview, job, tabs]).then(start_annotation, job, outs)
        run_btn.click(run_full, [*inputs, name1, colour1, name2, colour2, use_colours, swap, job],
                      [log, status_md, result, job])
        frame_img.select(click_frame, ann, outs)
        plan_img.select(click_plan, ann, outs)
        point.input(select_point, [ann, point], outs)
        prev_b.click(lambda a: step_frame(a, -1), ann, outs)
        next_b.click(lambda a: step_frame(a, 1), ann, outs)
        back_b.click(lambda a: shift_frame(a, -10), ann, outs)
        fwd_b.click(lambda a: shift_frame(a, 10), ann, outs)
        undo_b.click(undo, ann, outs)
        save_b.click(save_court, [ann, job], [court_log, court_status, court_prev, job]).then(
            lambda img: img, court_prev, preview)                  # the preview also appears in tab 1
    return demo


if __name__ == "__main__":
    build().queue().launch(inbrowser=True)
