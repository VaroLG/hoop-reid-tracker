"""Court homographies of a clip from the learned keypoint detector, frame by frame (court chapter, C09).

    keypoints (YOLO-pose)  ->  RANSAC homography (league template)  ->  polished on the painted lines

No chains: every frame is solved on its own, so one bad frame cannot spoil its
neighbours, and a clip is never "lost". Whether a frame is right is decided
afterwards by the clip camera (C07, ``verify_clip_camera.py``), exactly as for
C05's homographies; the output has the same format.

The league (NBA / NCAA / FIBA) is the one whose template, after polishing, puts
the court lines on the most painted lines over a few frames: the detector says
*which* point it sees ("lane corner on the baseline"), the league says *where*
it is in metres.

Frames listed in the clip's test anchors (``--anchors``) are always processed,
so the test set can measure every method on the same frames.

Usage:
    python scripts/court/c09_detect_homographies.py --weights outputs/court_runs/c09_yolo26n_pose/weights/best.pt \
        --sequence v_5ekaksddqrc_c003 v_BgwzTUxJaeU_c012 --out outputs/court_c09
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from _clip import Clip
from ultralytics import YOLO

from hooptrack.court.homography import fit_homography
from hooptrack.court.leagues import SPECS, lines, template
from hooptrack.court.refine import line_evidence, refine, sample_lines, score

ANCHORS = Path("experiments/court/C09_learned_keypoints/test_anchors")


def kpt_names(weights: Path) -> list[str]:
    """Keypoint names in the model's output order (from the training data.yaml next to the run)."""
    args = yaml.safe_load((weights.parent.parent / "args.yaml").read_text())
    d = yaml.safe_load(Path(args["data"]).read_text())
    names = d["kpt_names"][0] if isinstance(d["kpt_names"], dict) else d["kpt_names"]
    return [f"{int(n):02d}" if not isinstance(n, str) else n for n in names]


class Solver:
    def __init__(self, weights: Path, imgsz: int, conf: float, ignore_below: int) -> None:
        self.model = YOLO(str(weights))
        self.names = kpt_names(weights)
        self.imgsz, self.conf, self.ignore_below = imgsz, conf, ignore_below
        self.lines = {k: sample_lines(lines(s)) for k, s in SPECS.items()}

    def keypoints(self, img):
        r = self.model.predict(img, imgsz=self.imgsz, verbose=False)[0]
        if r.keypoints is None or not len(r.boxes):
            return None
        i = int(r.boxes.conf.argmax())
        return r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()

    def solve(self, img, people, league: str, kp=None):
        """(H, agreement) for one frame and league, or None."""
        kp = kp if kp is not None else self.keypoints(img)
        if kp is None:
            return None
        hom = fit_homography(self.names, kp[0], kp[1], template(SPECS[league]), self.conf)
        if hom is None:
            return None
        ev = line_evidence(img, None, people)
        ev.valid[self.ignore_below:, :] = 0
        S, T = self.lines[league]
        H = refine(hom.H, S, ev, T, taus=(12.0, 5.0))
        return H, score(H, S, ev, tangents=T)[0]


def pick_league(solver: Solver, clip: Clip, frames) -> tuple[str, dict]:
    totals = dict.fromkeys(SPECS, 0.0)
    for f in frames:
        img = clip.frame(f)
        kp = solver.keypoints(img)
        for lg in SPECS:
            r = solver.solve(img, clip.people(f), lg, kp)
            totals[lg] += r[1] if r else 0.0
    return max(totals, key=totals.get), {k: round(v, 2) for k, v in totals.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--sequence", nargs="+", required=True)
    ap.add_argument("--league", default="auto")
    ap.add_argument("--stride", type=int, default=5, help="solve one frame every N (+ the test-anchor frames)")
    ap.add_argument("--conf", type=float, default=0.5, help="keypoint confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-agreement", type=float, default=0.3, help="drop frames whose lines barely agree")
    ap.add_argument("--ignore-below", type=int, default=575, help="score bar: rows below are not court evidence")
    ap.add_argument("--anchors", type=Path, default=ANCHORS)
    ap.add_argument("--out", type=Path, default=Path("outputs/court_c09"))
    args = ap.parse_args()
    solver = Solver(args.weights, args.imgsz, args.conf, args.ignore_below)

    for seq in args.sequence:
        t0 = time.time()
        clip = Clip(seq)
        frames = set(range(1, clip.length + 1, args.stride))
        anchors = args.anchors / f"anchors_{seq}.json"
        if anchors.is_file():
            frames |= {a["frame"] for a in json.loads(anchors.read_text())["anchors"]}
        if args.league == "auto":
            league, totals = pick_league(solver, clip, range(1, clip.length + 1, max(clip.length // 6, 1)))
        else:
            league, totals = args.league, {}
        out_f, out_H, out_a = [], [], []
        for f in sorted(frames):
            r = solver.solve(clip.frame(f), clip.people(f), league)
            if r is not None and r[1] >= args.min_agreement:
                out_f.append(f)
                out_H.append(r[0])
                out_a.append(r[1])
        out = args.out / seq
        out.mkdir(parents=True, exist_ok=True)
        report = {"sequence": seq, "weights": str(args.weights), "league": league, "league_scores": totals,
                  "frames_tried": len(frames), "frames_with_H": len(out_f), "seconds": round(time.time() - t0)}
        if out_f:
            np.savez(out / "homographies.npz", frames=np.array(out_f), H=np.array(out_H),
                     agreement=np.array(out_a), league=league)
        (out / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
