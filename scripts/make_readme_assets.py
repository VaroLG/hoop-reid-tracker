"""Rebuild the README's videos of the final pipeline (``assets/final/``) from the outputs of the final selection.

``v_00HRwkvvjtQ_c001`` drawn by ``scripts/possession/p06_render_demo.py`` with the final identities (online tracker +
roster of 10 + id smoothing), the WASB ball tracks, the possession rules and the court homographies:

* ``showcase_c001.mp4`` (the whole clip) and ``showcase_c001.gif`` (its first 26 s, 7 fps, 560 px wide);
* ``showcase_vs_boxes_c001.mp4`` (showcase | honest side by side) and ``.gif`` (2.56 s -> 7.56 s, 8 fps, 960 px).

The videos show who has the ball, not the running possession share (``--possession-share``, off by default).
Needs ffmpeg on the PATH. A render that already exists in ``--work`` is reused (``--force`` redoes it).

Usage (repo root):
    python scripts/make_readme_assets.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIP = "v_00HRwkvvjtQ_c001"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def gif(src: Path, dst: Path, start: float, seconds: float, fps: int, width: int) -> None:
    """A GIF with its own palette (much cleaner than ffmpeg's default one)."""
    scale = f"fps={fps},scale={width}:-1:flags=lanczos"
    ffmpeg("-ss", str(start), "-t", str(seconds), "-i", str(src),
           "-vf", f"{scale},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
                  "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
           str(dst))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", type=Path, default=Path("outputs/runs/048_body_smooth"), help="final identities")
    ap.add_argument("--ball", type=Path, default=Path("outputs/ball/tracks_b08"))
    ap.add_argument("--possession", type=Path, default=Path("outputs/possession/p04"))
    ap.add_argument("--court-dir", default="outputs/court_complete/c09_pipeline")
    ap.add_argument("--work", type=Path, default=Path("outputs/videos/readme"), help="full-size renders")
    ap.add_argument("--out", type=Path, default=Path("assets/final"))
    ap.add_argument("--only", choices=["render", "assets"], default=None, help="one half of the job")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg is needed (on the PATH)")
    args.work.mkdir(parents=True, exist_ok=True)
    renders = {mode: args.work / f"{mode}_{CLIP}.mp4" for mode in ("showcase", "honest")}

    if args.only != "assets":
        for mode, video in renders.items():
            if video.exists() and not args.force:
                print(f"[skip] {video} exists")
                continue
            subprocess.run([sys.executable, str(ROOT / "scripts" / "possession" / "p06_render_demo.py"),
                            "--clip", CLIP, "--players", str(args.players), "--ball", str(args.ball),
                            "--possession", str(args.possession), "--court-dir", args.court_dir,
                            "--mode", mode, "--out", str(video)], check=True)
    if args.only == "render":
        return

    args.out.mkdir(parents=True, exist_ok=True)
    show, honest = renders["showcase"], renders["honest"]
    ffmpeg("-i", str(show), "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(args.out / "showcase_c001.mp4"))
    gif(show, args.out / "showcase_c001.gif", 0, 26, 7, 560)
    side = args.work / f"showcase_vs_boxes_{CLIP}.mp4"
    ffmpeg("-i", str(show), "-i", str(honest), "-filter_complex",
           "[0:v]scale=640:360[a];[1:v]scale=640:360[b];[a][b]hstack", "-c:v", "libx264", "-preset", "slow",
           "-crf", "27", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(side))
    shutil.copy(side, args.out / "showcase_vs_boxes_c001.mp4")
    gif(side, args.out / "showcase_vs_boxes_c001.gif", 2.56, 5, 8, 960)
    for f in sorted(args.out.glob("showcase*")):
        print(f"{f}  {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
