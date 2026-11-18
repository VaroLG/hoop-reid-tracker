"""The final pipeline on ONE clip, end to end: players, court, ball, possession and the demo video.

Every stage is the one chosen in the README ("The final selection"); this script only chains the scripts that
the experiments already use, in the right order, with the right inputs. Nothing here needs hand annotations:
ground truth, when the clip has it, is only used to print the metrics and to draw the honest video.

Input, one of:

* ``--clip v_00HRwkvvjtQ_c005`` — a SportsMOT clip already on disk (``scripts/download_data.py``);
* ``--video game.mp4 --name my_game`` — any basketball broadcast. Its frames are written as a sequence
  (1280x720, 25 fps, the format every model was trained on) under ``data/videos/clips/<name>/``.
  ``--start`` / ``--end`` (seconds) cut a piece: the pipeline costs minutes per second of video on a CPU.

Reuse: for the SportsMOT clips the experiments already computed the three slowest parts with the same models and
inputs (player detections, ~1 h per clip on a CPU; court homographies; WASB ball candidates). They are reused unless
``--no-reuse``, which runs the full inference as on a new video. A table of the time of each step is printed at the end.

Steps (outputs in ``outputs/pipeline/<clip>/``; a step whose output exists is skipped, ``--force`` redoes all):

1. **players, online** (011b): player detector + C06 court mask + tracker            -> ``players_online/``
2. **identities** (038 + 048): pieces cut at crossings, exactly 10 players, smoothing -> ``players/``
3. **court** (C09 -> C07 -> completion), only with ``--court`` (the court type)        -> ``court/final/``
4. **ball** (B08): WASB fine-tuned candidates + one-ball tracker                      -> ``ball/tracks/``
5. **possession** (P04): closest player moving with the ball, last state kept          -> ``possession/``
6. **video** (``p06_render_demo.py``): showcase, and honest when there is ground truth  -> ``<clip>_showcase.mp4``

``--court``: NCAA, NBA, FIBA, FIBA_PRE2010 or a YAML with your own measurements (``configs/courts/``). The court
type must come from a person (chosen automatically it was wrong on 4 of 6 clips, C09); without it the video has
no minimap. ``--court auto`` tries anyway.

``--venue NAME``: several clips filmed from the SAME camera spot (a fixed camera in a sports hall). The court is
solved once for the venue and kept in ``data/venues/<name>/``: the first clip tries the automatic court and, if
C07 rejects it, opens the annotator on the 3 frames with the most court in view (C06 mask); those clicks become
the venue's fixed court. Every later clip of the venue uses them, carried over by matching the static background;
a clip that does not match (another framing, a strong zoom) gets no court. ``--no-prompt`` never opens the
annotator (unattended runs).

``--court-only``: only the frames and the court, with ``court/preview.jpg`` (the court drawn over 3 frames): a check
of minutes before the run of an hour, which then reuses that court.

``--anchors anchors_<clip>.json``: when C07 rejects the automatic court (a sports hall with several courts painted
on the floor, say), click court points on ~3 frames with ``scripts/court/annotate_anchors.py`` and pass the file:
they replace C09 (C04 propagation), and the video is redrawn with the minimap. Anchors of ANOTHER clip filmed from
the same spot are carried over first by matching the static background (``scripts/court/transfer_anchors.py``);
frames that do not match well enough are not used, and a clip with none gets no court rather than a guessed one.

Alternatives of every stage (``--detector``, ``--court-mask``, ``--reid``, ``--identities``, ``--smoothing``,
``--ball``, ``--possession``; see ``STAGES``): the default of each is the final selection, the others are the
experiments' alternatives, so that any of them can be compared on any clip. A non-default choice writes to its own
folders (suffix ``__<choice>``) and video, so variants never overwrite each other. ``--mode`` picks the video style.

``--device auto`` runs every model on an NVIDIA GPU when PyTorch sees one (a CUDA build of torch: see the README),
else on the CPU; ``--device cpu`` / ``cuda`` / ``cuda:1`` force it. ``status.json`` sums up the run (video, court
found or not, the frames to annotate when it was not, the time of each step): the web app (``app/app.py``) reads it.

Usage (repo root):
    python scripts/run_pipeline.py --clip v_00HRwkvvjtQ_c005 --court NCAA
    python scripts/run_pipeline.py --video C:/videos/final.mp4 --name final_q4 --start 120 --end 160 --court FIBA
    python scripts/run_pipeline.py --video hall/game1.mp4 --name game1 --court FIBA --venue my_hall   # asks once
    python scripts/run_pipeline.py --video hall/game2.mp4 --name game2 --venue my_hall                 # reuses it

The models must be trained first (README, Quick start, step 1). Licences: docs/data_models_licences.md.
"""

from __future__ import annotations

import argparse
import configparser
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PY = sys.executable                       # the same Python (and environment) that runs this script
SCRIPTS = Path(__file__).resolve().parent
TIMES: dict[str, float] = {}              # step -> seconds, printed at the end


# --------------------------------------------------------------------------------------------- helpers
def run(step: str, cmd: list, done: Path, force: bool) -> None:
    """Run one stage as a subprocess, unless its output ``done`` already exists."""
    if done.exists() and not force:
        print(f"[skip] {step}: {done} exists")
        return
    print(f"\n=== {step}\n    {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], check=True)       # stop at the first stage that fails
    TIMES[step] = time.time() - t0
    print(f"    done in {TIMES[step]:.0f} s")


def video_to_sequence(video: Path, name: str, root: Path, start: float, end: float | None) -> Path:
    """Write a video as a MOTChallenge sequence (img1/000001.jpg ..., seqinfo.ini) at 1280x720 and 25 fps.

    25 fps and 1280x720 are what SportsMOT has, so every model and every value tuned in frames
    (ball physics, possession gaps, smoothing windows) keeps its meaning.
    """
    import cv2  # only needed for --video

    seq = root / name
    if (seq / "seqinfo.ini").is_file():
        print(f"[skip] frames: {seq} exists")
        return seq
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    (seq / "img1").mkdir(parents=True, exist_ok=True)
    k, i = 0, 0                                              # k: frames written, i: source frame index
    want = int(round(start * src_fps))                       # next source frame to keep
    while True:
        ok, img = cap.read()
        if not ok or (end is not None and i / src_fps > end):
            break
        if i == want:
            k += 1
            cv2.imwrite(str(seq / "img1" / f"{k:06d}.jpg"), cv2.resize(img, (1280, 720)),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            want = int(round((start + k / 25.0) * src_fps))  # resample to 25 fps by time
        i += 1
    if k == 0:
        raise SystemExit(f"no frames read from {video} between {start} s and {end} s")
    info = configparser.ConfigParser()
    info.optionxform = str                                   # keep the MOTChallenge key case
    info["Sequence"] = {"name": name, "imDir": "img1", "frameRate": "25", "seqLength": str(k),
                        "imWidth": "1280", "imHeight": "720", "imExt": ".jpg"}
    with open(seq / "seqinfo.ini", "w") as f:
        info.write(f)
    print(f"frames: {k} at 25 fps ({k / 25:.1f} s) from {video} -> {seq}")
    return seq


def find_sportsmot_split(cfg: dict, clip: str) -> str:
    for split in ("val", "train", "test"):
        if (Path(cfg["data"]["root"]) / split / clip / "seqinfo.ini").is_file():
            return split
    raise SystemExit(f"{clip} not found under {cfg['data']['root']}: run scripts/download_data.py, or use --video")


def save_venue(venue: Path, anchors: Path, source_config: Path) -> None:
    """Keep a clip's hand anchors as the fixed court of a venue (one camera spot), for every clip filmed there."""
    venue.mkdir(parents=True, exist_ok=True)
    shutil.copy(anchors, venue / "anchors.json")
    shutil.copy(source_config, venue / "source_config.yaml")     # where the annotated clip's frames are
    print(f"venue court saved: {venue / 'anchors.json'} (from {json.loads(anchors.read_text())['sequence']})")


def wide_frames(cfg: dict, seq_dir: Path, n: int = 3, device: str = "cpu") -> list[int]:
    """The frame with the most court in view (C06 court mask) in each third of the clip: the best to annotate."""
    import cv2

    from hooptrack.court.segment import CourtSegmenter

    s = cfg["filters"]["segmentation"]
    seg = CourtSegmenter(s["weights"], s.get("imgsz", 512), s.get("conf", 0.25), device)
    frames = sorted(seq_dir.glob("img1/*.jpg"))
    picks = []
    for part in np.array_split(np.arange(len(frames)), n):
        best = max(part[::10], key=lambda i: seg(cv2.imread(str(frames[i]))).mean())
        picks.append(int(best) + 1)                          # MOT frames are 1-based
    return picks


def court_automatic(clip: str, conf: Path, out: Path, court_dir: Path, args) -> str:
    """C09 keypoints -> C07 one-camera check -> completion. Returns the C07 verdict ("ok" or "review")."""
    raw, ver = out / "court" / "raw", out / "court" / "verified"
    # same method run before (scripts/court/c09_pipeline.ps1): reused when it was given the same court type
    old_raw = Path("outputs/court_c09_pipeline") / clip
    old_final = Path("outputs/court_complete/c09_pipeline") / clip
    if (old_final / "homographies.npz").is_file() and (old_raw / "report.json").is_file() and not args.no_reuse \
            and not (court_dir / clip / "homographies.npz").exists():
        league = json.loads((old_raw / "report.json").read_text()).get("league", "")
        if league.upper() == str(args.court).upper():
            shutil.copytree(old_final, court_dir / clip, dirs_exist_ok=True)
            print(f"[reuse] court ({league}) from {old_final}")
    if (court_dir / clip / "homographies.npz").exists() and not (ver / clip / "report.json").exists():
        return "ok"
    run("3a · court keypoints and homographies (C09)",
        [PY, SCRIPTS / "court" / "c09_detect_homographies.py", "--weights", args.court_weights, "--imgsz", "640",
         "--stride", "1", "--sequence", clip, "--league", args.court, "--config", conf, "--device", args.device,
         "--out", raw],
        raw / clip / "homographies.npz", args.force)
    run("3b · one-camera check (C07)",
        [PY, SCRIPTS / "court" / "verify_clip_camera.py", "--in", raw, "--sequence", clip, "--out", ver],
        ver / clip / "report.json", args.force)
    verdict = json.loads((ver / clip / "report.json").read_text())["verdict"]
    if verdict == "ok":
        run("3c · court completion",
            [PY, SCRIPTS / "court" / "c09_complete_court.py", "--in", ver, "--sequence", clip, "--config", conf,
             "--out", court_dir], court_dir / clip / "homographies.npz", args.force)
    else:
        print(f"C07 verdict '{verdict}': no camera explains this clip's automatic courts (nothing guessed)")
    return verdict


def court_from_anchors(anchors: Path, clip: str, conf: Path, out: Path, court_dir: Path, venue: Path | None,
                       args) -> None:
    """C04: hand anchors (this clip's, or a venue's carried over by the static background) -> propagation ->
    C07 check -> completion. A clip that the anchors cannot be carried to gets no court."""
    anc, ver = out / "court" / "anchors", out / "court" / "verified_anchors"
    source = json.loads(anchors.read_text())["sequence"]
    if source != clip:
        moved = out / "court" / "anchors_transferred"
        if venue is not None and (venue / "source_config.yaml").is_file():
            src_conf = venue / "source_config.yaml"
        else:
            src_conf = args.out / source / "config.yaml"
        run(f"3 · court anchors carried over from {source} (same camera spot)",
            [PY, SCRIPTS / "court" / "transfer_anchors.py", "--anchors", anchors, "--sequence", clip,
             "--config", conf, "--source-config", src_conf if src_conf.is_file() else conf, "--out", moved],
            moved / "done", args.force)
        (moved / "done").touch()
        anchors = moved / f"anchors_{clip}.json"
        if not anchors.is_file():
            print(f"{clip} does not match the frames of {source} (another framing or zoom): no court, no minimap")
            return
    people = out / "players" / clip / "results" / f"{clip}.txt"     # masked out of the floor motion, if tracked yet
    run("3a · court from hand anchors (C04)",
        [PY, SCRIPTS / "court" / "propagate_homographies.py", "--anchors", anchors, "--config", conf, "--out", anc]
        + (["--people", people] if people.is_file() else []),
        anc / clip / "homographies.npz", args.force)
    run("3b · one-camera check (C07)",
        [PY, SCRIPTS / "court" / "verify_clip_camera.py", "--in", anc, "--sequence", clip, "--out", ver],
        ver / clip / "report.json", args.force)
    if json.loads((ver / clip / "report.json").read_text())["verdict"] == "ok":
        run("3c · court completion",
            [PY, SCRIPTS / "court" / "c09_complete_court.py", "--in", ver, "--sequence", clip, "--config", conf,
             "--out", court_dir], court_dir / clip / "homographies.npz", args.force)
    elif not (court_dir / clip / "homographies.npz").exists() or args.force:
        # no single camera explains them (a hand-held phone): the hand-anchored chains are used as they are
        (court_dir / clip).mkdir(parents=True, exist_ok=True)
        shutil.copy(anc / clip / "homographies.npz", court_dir / clip / "homographies.npz")
        print("C07 verdict 'review' on the hand-anchored court: used as propagated, without the camera model")


# ------------------------------------------------------------------------------------------------ main
# Alternatives of every stage. The first value of each is the final selection (README, "The final selection"); the
# others are the experiments' alternatives, kept so that they can be compared on any clip. A stage whose choice is
# not the default writes to its own folder (suffix "__<choice>"), so variants never overwrite each other.
STAGES = {
    "detector": ["player", "coco"],                  # 011 fine-tuned on players | COCO "person" + referee filter (006)
    "court_mask": ["segmentation", "colour", "none"],  # C06 learned court area | floor colour (003) | off
    "reid": ["resnet", "osnet", "histogram", "none"],  # appearance inside the online tracker (005, 012) | motion only
    "tracker": ["online", "sam2"],                   # online tracker (011b) | SAM 2 masks carry ids (049, GPU)
    "identities": ["closed10", "stitch", "online"],  # exactly 10 players (038) | SigLIP stitching (013) | as tracked
    "smoothing": ["on", "off"],                      # P -> Q -> P excursions shown as P (048)
    "ball": ["wasb", "yolo", "none"],                # WASB fine-tuned (B08) | our YOLO26s + static filter (B05c) | off
    "possession": ["keep", "rules", "none"],         # last state kept (P04) | plain rules (P03) | off
}


def court_stage(args, clip: str, cfg: dict, conf: Path, out: Path, seq_dir: Path, status: dict) -> Path:
    """Step 3, the court: fills ``status["court"]`` (and ``suggested_frames`` when it must be clicked)."""
    # 3 · court geometry. Order: the venue's fixed anchors (if any) -> the automatic court (C09 -> C07 ->
    #     completion) -> if that fails: with a venue, ask a person for anchors once, for every clip of it;
    #     with --no-prompt, report it (status.json) with the frames to annotate.
    court_dir = out / "court" / "final"
    venue = Path("data/venues") / args.venue if args.venue else None
    if venue is not None and args.anchors is not None and not (venue / "anchors.json").is_file():
        source = json.loads(args.anchors.read_text())["sequence"]
        save_venue(venue, args.anchors, conf if source == clip else args.out / source / "config.yaml")
    anchors = args.anchors or (venue / "anchors.json" if venue is not None and (venue / "anchors.json").is_file()
                               else None)
    status["court"] = "none"
    if anchors is not None:
        court_from_anchors(anchors, clip, conf, out, court_dir, venue, args)
        status["court"] = "anchors"
    elif args.court:
        verdict = court_automatic(clip, conf, out, court_dir, args)
        status["court"] = "automatic" if verdict == "ok" else "not_found"
        if verdict != "ok" and venue is not None and not args.no_prompt:
            print(f"\nThe court of {clip} could not be found automatically. Click it once for the venue "
                  f"'{args.venue}': the 3 frames shown are the ones with the most court in view. Mark at least 5 "
                  "crossings of the white (basketball) lines on each, check the drawn court lies on them, press q.")
            frames = wide_frames(cfg, seq_dir, device=args.device)
            league = args.court if args.court.upper() in ("NBA", "NCAA", "FIBA", "FIBA_PRE2010") else "FIBA"
            clicked = out / "court" / "anchors_clicked"
            subprocess.run([PY, SCRIPTS / "court" / "annotate_anchors.py", "--sequence", clip, "--league",
                            league.upper(), "--config", conf, "--out", clicked,
                            "--frames", *[str(f) for f in frames]], check=True)
            clicked_file = clicked / f"anchors_{clip}.json"
            if clicked_file.is_file() and json.loads(clicked_file.read_text())["anchors"]:
                save_venue(venue, clicked_file, conf)
                court_from_anchors(venue / "anchors.json", clip, conf, out, court_dir, venue, args)
                status["court"] = "anchors"
            else:
                print("no anchors clicked: no court for this clip")
        elif verdict != "ok":
            status["suggested_frames"] = wide_frames(cfg, seq_dir, device=args.device)
            print(f"no court: annotate frames {status['suggested_frames']} (web app, or "
                  "scripts/court/annotate_anchors.py) and run again with --anchors, or use --venue")
    else:
        print("no --court, --venue or --anchors: the court geometry is skipped and the video has no minimap")
    if status["court"] == "anchors" and not (court_dir / clip / "homographies.npz").exists():
        status["court"] = "not_matched"               # venue anchors that do not fit this clip's framing
    return court_dir



def court_preview(clip: str, seq_dir: Path, court_dir: Path, dst: Path, n: int = 3) -> Path | None:
    """The court drawn over ``n`` frames spread over the clip: a quick look before (or after) the long run."""
    import cv2

    from hooptrack.court.clipcam import court_homographies
    from hooptrack.court.leagues import lines, spec_from_npz

    npz = court_dir / clip / "homographies.npz"
    if not npz.is_file():
        return None
    z = np.load(npz)
    spec, H = spec_from_npz(z), court_homographies(z, use_ptz=True, smooth=True)
    frames = sorted(H)
    tiles = []
    for f in [frames[int(q * (len(frames) - 1))] for q in np.linspace(0.1, 0.9, n)]:
        img = cv2.imread(str(seq_dir / "img1" / f"{f:06d}.jpg"))
        for pl in lines(spec):
            q = cv2.perspectiveTransform(pl[None].astype(np.float64), H[f])[0].astype(np.int32)
            cv2.polylines(img, [q], False, (0, 230, 255), 3, cv2.LINE_AA)
        cv2.putText(img, f"frame {f}", (24, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4, cv2.LINE_AA)
        tiles.append(cv2.resize(img, (640, 360)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), np.hstack(tiles))
    return dst


def import_players(src: Path, clip: str, out: Path, label: str) -> tuple[Path, str]:
    """Identities computed elsewhere (``--players-from``): copied as ``<out>/players__<label>/<clip>/results``, the
    layout of the steps below. Returns that folder and the suffix of everything made from it."""
    found = [f for f in (src / clip / "results" / f"{clip}.txt", src / "results" / f"{clip}.txt") if f.is_file()]
    if not found:
        raise SystemExit(f"--players-from: no {clip}.txt in {src} (<dir>/<clip>/results/ or <dir>/results/)")
    players = out / f"players__{label}"
    (players / clip / "results").mkdir(parents=True, exist_ok=True)
    shutil.copy(found[0], players / clip / "results" / f"{clip}.txt")
    print(f"[import] identities from {found[0]}")
    return players, f"__{label}"


def tag(**choices: str) -> str:
    """Folder suffix of a variant: empty for the final selection, '__<choice>-...' otherwise."""
    extra = [v for k, v in choices.items() if v != STAGES[k][0]]
    return "__" + "-".join(extra) if extra else ""


def resolve_device(device: str) -> str:
    """'auto' -> 'cuda' when PyTorch sees an NVIDIA GPU, else 'cpu'. Anything else is passed as given."""
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--clip", help="a SportsMOT clip on disk, e.g. v_00HRwkvvjtQ_c005")
    src.add_argument("--video", type=Path, help="any basketball broadcast video file")
    ap.add_argument("--name", help="with --video: the clip's name (default: the file name)")
    ap.add_argument("--start", type=float, default=0.0, help="with --video: first second used")
    ap.add_argument("--end", type=float, default=None, help="with --video: last second used")
    ap.add_argument("--court", default=None, help="court type: NCAA, NBA, FIBA, FIBA_PRE2010, a YAML, or auto")
    ap.add_argument("--venue", default=None,
                    help="name of the camera spot: its court is found once (automatically, or clicked by you on 3 "
                         "frames when that fails) and reused for every clip filmed from there (data/venues/<name>/)")
    ap.add_argument("--no-prompt", action="store_true",
                    help="never open the annotator: a court that cannot be found is reported in status.json, with "
                         "the frames to annotate (the web app uses this)")
    ap.add_argument("--court-only", action="store_true",
                    help="only the frames and the court (with a preview image): a quick check before the long run; "
                         "the full run reuses that court")
    ap.add_argument("--anchors", type=Path, default=None,
                    help="court from hand anchors instead of C09 (scripts/court/annotate_anchors.py): for courts the "
                         "keypoint model cannot read, e.g. a sports hall with several courts painted on the floor")
    for stage, choices in STAGES.items():
        if stage == "tracker":                        # the only stage whose default depends on the machine
            ap.add_argument("--tracker", choices=["auto", *choices], default="auto",
                            help="auto (default): sam2 when there is an NVIDIA GPU (049, better identities), online "
                                 "on a CPU (011b, the final selection; SAM 2 would take hours) | online | sam2")
            continue
        ap.add_argument(f"--{stage.replace('_', '-')}", choices=choices, default=choices[0],
                        help=f"{' | '.join(choices)} (default: {choices[0]}, the final selection)")
    ap.add_argument("--mode", choices=["showcase", "tracking", "ball"], default="showcase", help="style of the video")
    ap.add_argument("--team-names", default=None, help="video: two team names, comma-separated (default A, B)")
    ap.add_argument("--team-colours", default=None,
                    help="video: two colours '#rrggbb,#rrggbb', each given to the team whose kit it resembles most")
    ap.add_argument("--swap-teams", action="store_true", help="video: exchange the two teams' names / colours")
    ap.add_argument("--device", default="auto", help="auto (an NVIDIA GPU if PyTorch sees one), cpu, cuda, cuda:1 ...")
    ap.add_argument("--config", default="configs/player.yaml", help="the online tracker (011b)")
    ap.add_argument("--siglip", default="weights/siglip_reid_sportsmot", help="ReID of the identities (028)")
    ap.add_argument("--osnet", default="weights/osnet_x1_0_sportsmot.pth", help="with --reid osnet (012)")
    ap.add_argument("--court-weights", default="outputs/court_runs/c09_randdims/weights/best.pt")
    ap.add_argument("--ball-weights", default="weights/wasb_ft_b08.pt")
    ap.add_argument("--ball-yolo-weights", default="weights/ball_yolo26s_1280.pt", help="with --ball yolo (B03)")
    ap.add_argument("--sam2-model", default="facebook/sam2.1-hiera-small", help="with --tracker sam2 (049)")
    ap.add_argument("--players-only", action="store_true",
                    help="stop after the identities (steps 1-2): e.g. SAM 2 on a GPU (Colab), the rest on the laptop")
    ap.add_argument("--players-from", type=Path, default=None,
                    help="identities computed elsewhere (e.g. SAM 2 + 038 + 048 in Colab): <dir>/<clip>/results/"
                         "<clip>.txt or <dir>/results/<clip>.txt; steps 1-2 are skipped")
    ap.add_argument("--players-label", default="imported",
                    help="with --players-from: suffix of the folders and videos made from them, e.g. sam2")
    ap.add_argument("--out", type=Path, default=Path("outputs/pipeline"))
    ap.add_argument("--force", action="store_true", help="redo every step of this clip's own outputs")
    ap.add_argument("--no-reuse", action="store_true",
                    help="do not reuse what the experiments already computed for this clip (player detections, "
                         "court, ball candidates): the full inference, as on a new video")
    args = ap.parse_args()
    dev = args.device = resolve_device(args.device)
    if args.tracker == "auto":
        args.tracker = "sam2" if dev.startswith("cuda") else "online"

    # 0 · where the frames are, and a config pointing at them (saved with the outputs: reproducible)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.video is not None:
        clip = args.name or args.video.stem
        video_to_sequence(args.video, clip, Path("data/videos/clips"), args.start, args.end)
        # no split / sport lists in data/videos: every sequence folder is taken (hooptrack.data.sportsmot)
        cfg["data"].update(root="data/videos", split="clips", splits_dir="data/videos", only=None)
    else:
        clip = args.clip
        cfg["data"].update(split=find_sportsmot_split(cfg, clip), only=None)
    split = cfg["data"]["split"]
    out = args.out / clip
    out.mkdir(parents=True, exist_ok=True)
    conf = out / "config.yaml"
    conf.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    seq_dir = Path(cfg["data"]["root"]) / split / clip
    has_gt = (seq_dir / "gt" / "gt.txt").is_file()
    status: dict = {"clip": clip, "device": dev, "options": {k: getattr(args, k) for k in STAGES},
                    "ground_truth": has_gt}
    print(f"clip {clip} ({seq_dir}), ground truth: {'yes' if has_gt else 'no'}, device: {dev}\n"
          f"stages: " + ", ".join(f"{k}={getattr(args, k)}" for k in STAGES))

    if args.court_only:                               # a quick check: frames + court, nothing else
        court_dir = court_stage(args, clip, cfg, conf, out, seq_dir, status)
        preview = court_preview(clip, seq_dir, court_dir, out / "court" / "preview.jpg")
        status.update(stage="court_only", preview=str(preview) if preview else None,
                      minutes={k: round(v / 60, 2) for k, v in TIMES.items()})
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"\ncourt: {status['court']}" + (f", preview {preview}" if preview else ""))
        return

    if args.players_from is not None:                 # identities computed elsewhere (e.g. SAM 2 in Colab)
        players, s_players = import_players(args.players_from, clip, out, args.players_label)
    else:
        # 1 · players, online tracker (011b, or the chosen detector / court mask / appearance model)
        s1 = tag(detector=args.detector, court_mask=args.court_mask, reid=args.reid)
        online = out / f"players_online{s1}"
        sets = [f"data.only=[{clip}]", f"output.dir={online}", f"detector.device={dev}", f"reid.device={dev}",
                f"filters.segmentation.device={dev}"]
        if args.detector == "coco":                       # COCO also sees referees and the bench: 006 hides referees
            sets += ["detector.weights=yolo26m.pt", "teams.enabled=true", "tracker.suppress_other=true"]
        if args.court_mask == "colour":
            sets += ["filters.court_source=colour"]
        elif args.court_mask == "none":
            sets += ["filters.court_mask=false"]
        if args.reid == "osnet":
            sets += ["reid.type=osnet", f"reid.weights={args.osnet}"]
        elif args.reid in ("histogram", "none"):
            sets += [f"reid.type={args.reid}"] + (["tracker.memory_enabled=false"] if args.reid == "none" else [])
        cache = Path("outputs/detections/yolo26m_player") / f"{clip}.txt"
        if args.detector == "player" and cache.is_file() and not args.no_reuse:
            sets += ["detector.type=cached", f"detector.cache_dir={cache.parent}"]
            print(f"[reuse] player detections from {cache}")
        run("1 · players, online tracker", [PY, "-m", "hooptrack.cli", "track", "--config", conf, "--set", *sets]
            + (["--eval"] if has_gt else []), online / "results" / f"{clip}.txt", args.force)
        boxes = online                                    # the identities below start from these tracks
        if args.tracker == "sam2":                        # 049: a SAM 2 mask carries each player through crossings
            s1 = tag(detector=args.detector, court_mask=args.court_mask, reid=args.reid, tracker=args.tracker)
            boxes = out / f"players_sam2{s1}"
            if dev == "cpu":
                print("warning: SAM 2 on the CPU is very slow (about 1 fps on a T4 GPU; far less on a CPU)")
            run("1b · SAM 2 carries the identities (049: window 120, min share 0.4, keep bonus 0.3)",
                [PY, SCRIPTS / "sam2_track.py", "--run", online, "--config", conf, "--clips", clip,
                 "--model", args.sam2_model, "--window", "120", "--min-share", "0.4", "--keep-bonus", "0.3",
                 "--device", dev, "--dtype", "float16" if dev.startswith("cuda") else "float32", "--out", boxes],
                boxes / "results" / f"{clip}.txt", args.force)

        # 2 · identities: exactly 10 players (038) | SigLIP stitching | the online ids, then the smoothing (048)
        s2 = s1 + ("" if args.identities == "closed10" else f"__{args.identities}")
        if args.identities == "online":                   # same layout as the other runs: <run>/<clip>/results
            ids = out / f"players_ids{s2}"
            (ids / clip / "results").mkdir(parents=True, exist_ok=True)
            shutil.copy(boxes / "results" / f"{clip}.txt", ids / clip / "results" / f"{clip}.txt")
        else:
            ids = out / f"players_038{s1}" if args.identities == "closed10" else out / f"players_stitch{s1}"
            roster = ["--split-crossings", "--closed-set", "10", "--continuity", "0.3"] \
                if args.identities == "closed10" else []
            run(f"2a · identities ({args.identities})",
                [PY, SCRIPTS / "stitch_tracklets.py", "--run", boxes, "--config", conf, "--clips", clip, "--set",
                 "reid.type=siglip", f"reid.weights={args.siglip}", f"reid.device={dev}", *roster, "--out", ids],
                ids / clip / "results" / f"{clip}.txt", args.force)
        players = ids
        if args.smoothing == "on":
            players = out / f"players{s2}"
            run("2b · body smoothing (048)", [PY, SCRIPTS / "smooth_body_labels.py", "--run", ids, "--out", players],
                players / clip / "results" / f"{clip}.txt", args.force)
        s_players = s2 + ("" if args.smoothing == "on" else "__nosmooth")
    if args.players_only:                             # e.g. on a GPU: the rest is done where the court / ball are
        status.update(stage="players_only", players=str(players))
        (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"\nidentities: {players / clip / 'results' / (clip + '.txt')}")
        return

    court_dir = court_stage(args, clip, cfg, conf, out, seq_dir, status)
    court_npz = court_dir / clip / "homographies.npz"

    # 4 · ball: WASB fine-tuned (B08) | our YOLO26s + static filter (B05c) | none
    if args.ball == "wasb":
        cand, tracks = out / "ball" / "candidates", out / "ball" / "tracks"
        cached = Path("outputs/ball/dets_wasb_ft") / f"{clip}.csv"     # same model, run before (B08 experiment)
        if cached.is_file() and not (cand / f"{clip}.csv").exists() and not args.no_reuse:
            cand.mkdir(parents=True, exist_ok=True)
            shutil.copy(cached, cand / f"{clip}.csv")
            print(f"[reuse] ball candidates from {cached}")
        run("4a · ball candidates, WASB fine-tuned (B08)",
            [PY, SCRIPTS / "ball" / "b07_wasb_probe.py", "--config", conf, "--split", split, "--clips", clip,
             "--weights", args.ball_weights, "--players", players, "--device", dev, "--out", cand],
            cand / f"{clip}.csv", args.force)
        run("4b · one-ball tracker (B08)",
            [PY, SCRIPTS / "ball" / "b05_track_ball.py", "--config", conf, "--split", split, "--clips", clip,
             "--dets", cand, "--name", "b08", "--out", out / "ball", "--save-tracks", tracks],
            tracks / f"{clip}.csv", args.force)
    elif args.ball == "yolo":
        cand, tracks = out / "ball_yolo" / "candidates", out / "ball_yolo" / "tracks"
        run("4a · ball candidates, our YOLO26s (B03)",
            [PY, SCRIPTS / "ball" / "b05_detect_all.py", "--config", conf, "--split", split, "--clips", clip,
             "--weights", args.ball_yolo_weights, "--device", dev, "--out", cand], cand / f"{clip}.csv", args.force)
        run("4b · one-ball tracker, still look-alikes removed on the court plan (B05c)",
            [PY, SCRIPTS / "ball" / "b06_save_tracks.py", "--dets", cand, "--out", tracks, "--court-dirs", court_dir],
            tracks / f"{clip}.csv", args.force)
    else:                                             # no ball: empty files, so the video is drawn without it
        tracks = out / "ball_none"
        tracks.mkdir(exist_ok=True)
        (tracks / f"{clip}.csv").write_text("frame,cx,cy,d,source\n")
    s_ball = "" if args.ball == "wasb" else f"__{args.ball}"

    # 5 · possession: last state kept (P04) | plain rules (P03) | none
    s_poss = "" if args.possession == "keep" else f"__{args.possession}"
    poss = out / f"possession{s_players}{s_ball}{s_poss}"
    if args.possession == "none" or args.ball == "none":
        poss.mkdir(parents=True, exist_ok=True)
        (poss / f"{clip}_holder.csv").write_text("frame,holder\n")
    else:
        run(f"5 · possession ({'P04' if args.possession == 'keep' else 'P03'})",
            [PY, SCRIPTS / "possession" / "p01_possession.py", "--config", conf, "--split", split,
             "--tracks", tracks, "--players", players, "--out", poss]
            + (["--set", "keep_state=true"] if args.possession == "keep" else []),
            poss / f"{clip}_holder.csv", args.force)

    # 6 · videos, and the metrics when there is ground truth
    variant = s_players + s_ball + s_poss
    common = ["--config", conf, "--clip", clip, "--players", players, "--ball", tracks,
              "--possession", poss, "--court-dir", court_dir]
    look = {"team_names": args.team_names, "team_colours": args.team_colours, "swap_teams": args.swap_teams}
    common += (["--team-names", args.team_names] if args.team_names else []) \
        + (["--team-colours", args.team_colours] if args.team_colours else []) \
        + (["--swap-teams"] if args.swap_teams else [])

    def stale(video: Path) -> bool:
        """Redraw a video when the court changed after it was drawn (e.g. --anchors) or the team names / colours
        changed: only the video is redone, in seconds."""
        newer = court_npz.exists() and video.exists() and court_npz.stat().st_mtime > video.stat().st_mtime
        side = video.with_suffix(".json")
        old = json.loads(side.read_text()) if side.is_file() else {"team_names": None, "team_colours": None,
                                                                     "swap_teams": False}
        return args.force or newer or old != look

    video = out / f"{clip}_{args.mode}{variant}.mp4"
    run(f"6a · video, {args.mode}", [PY, SCRIPTS / "possession" / "p06_render_demo.py", *common,
                                      "--mode", args.mode, "--out", video], video, stale(video))
    video.with_suffix(".json").write_text(json.dumps(look))     # the look it was drawn with
    status["video"] = str(video)
    if has_gt:
        honest = out / f"{clip}_honest{variant}.mp4"
        run("6b · video, honest (identity switches against the ground truth)",
            [PY, SCRIPTS / "possession" / "p06_render_demo.py", *common, "--mode", "honest", "--out", honest],
            honest, stale(honest))
        honest.with_suffix(".json").write_text(json.dumps(look))
        metrics = out / f"metrics{s_players}.csv"
        run("metrics (HOTA, IDF1, MOTA)",
            [PY, SCRIPTS / "eval_hota.py", "--config", conf, "--set", f"data.only=[{clip}]",
             "--runs", f"final={players}", "--out", metrics], metrics, args.force)
        status.update(honest=str(honest), metrics=str(metrics))
    print("\ntime per step (steps skipped or reused are not listed):")
    for step, s in TIMES.items():
        print(f"  {s / 60:6.1f} min  {step}")
    print(f"  {sum(TIMES.values()) / 60:6.1f} min  total")
    status["minutes"] = {k: round(v / 60, 2) for k, v in TIMES.items()}
    (out / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"\nDone: {out} (summary: {out / 'status.json'})")


if __name__ == "__main__":
    main()
