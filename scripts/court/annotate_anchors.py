"""Click court keypoints on a few frames of a clip: hand anchors for the court test set (court chapter, C09).

Two windows:

* **court** — the court plan with the 18 keypoints that sit where two painted
  lines cross (``CLEAR``: the only ones a person can place exactly). Click one to select it
  (or ``]`` / ``[`` for next / previous). Placed points turn green.
* **frame** — the video frame. Left click places the selected keypoint there
  (and selects the next one); a magnifier follows the cursor, and a click INSIDE
  the magnifier places the point with sub-pixel precision. Right click removes
  the nearest point.

With 4 points or more the court is drawn with the fitted homography (yellow),
each point's reprojection error is shown, and ``r`` polishes it on the painted
lines (C04 ``fit_anchor``) and draws the result in cyan with its line agreement.
If the yellow lines do not lie on the painted ones, a point is wrong.

Keys: ``n`` / ``p`` next / previous frame · ``+`` / ``-`` replace this frame by
the one 10 frames later / earlier, if it shows too few lines (its points are
dropped: with points placed, press twice) · arrows nudge the selected
point by 0.5 px · ``d`` delete the selected point · ``h`` hide the court ·
``s`` save · ``q`` or Esc save and quit. Saving also happens on every frame change.

Output: ``anchors_<sequence>.json`` (format of ``hooptrack.court.anchors``), so
``propagate_homographies.py`` and ``eval_court_test.py`` read it directly.

Usage:
    python scripts/court/annotate_anchors.py --manifest experiments/court/C09_learned_keypoints/test_set.json
    python scripts/court/annotate_anchors.py --sequence v_5ekaksddqrc_c003 --league FIBA
    python scripts/court/annotate_anchors.py --sequence v_5ekaksddqrc_c003 --league FIBA --frames 60 300 520
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from _clip import Clip

from hooptrack.court.anchors import Anchor, fit_anchor, homography_from_points
from hooptrack.court.leagues import SPECS, keypoints, lines

OUT_DIR = Path("experiments/court/C09_learned_keypoints/test_anchors")
# Only points where two PAINTED lines cross: court corners, 3-pt corner lines and lane on the baseline,
# free-throw line ends, halfway line on the sidelines. The others have no mark to click on (hoop centre,
# free-throw line centre, top of the arc, where the arc meets its straight part, sideline marks):
# on the first test annotations they were 14-65 px off while these were within 3 px.
CLEAR = frozenset({"01", "02", "04", "05", "07", "08", "12", "14", "19", "23",
                   "28", "30", "34", "35", "37", "38", "40", "41"})
LOUPE, ZOOM = 64, 4                       # magnifier: LOUPE x LOUPE pixels shown ZOOM times larger
KEYS_LEFT, KEYS_UP, KEYS_RIGHT, KEYS_DOWN = 2424832, 2490368, 2555904, 2621440     # waitKeyEx on Windows


class Annotator:
    def __init__(self, clip: Clip, league: str, frames: list[int], path: Path) -> None:
        self.clip, self.spec, self.path = clip, SPECS[league], path
        self.league = league
        self.kp = {k: v for k, v in keypoints(self.spec).items() if k in CLEAR}
        self.names = sorted(self.kp)
        self.points: dict[int, dict[str, tuple[float, float]]] = {f: {} for f in frames}
        self.notes: dict[int, str] = {}
        if path.is_file():                                          # resume
            d = json.loads(path.read_text())
            for a in d["anchors"]:
                self.points[a["frame"]] = {k: tuple(v) for k, v in a["points"].items() if k in CLEAR}
                self.notes[a["frame"]] = a.get("note", "")
        self.frames = sorted(self.points)
        self.i, self.sel = 0, self.names[0]
        self.mouse = (0, 0)
        self.show_court, self.refined = True, None
        self.message, self.message_until, self.confirm_move = "", 0.0, None
        self.img = None
        self.load()
        self.plan, self.to_px, self.from_px = self._plan()

    # ------------------------------------------------------------------ state
    @property
    def frame(self) -> int:
        return self.frames[self.i]

    def load(self) -> None:
        self.img = self.clip.frame(self.frame)
        self.refined = None

    def save(self) -> None:
        anchors = [{"frame": f, "points": {k: [round(u, 2), round(v, 2)] for k, (u, v) in sorted(p.items())},
                    "note": self.notes.get(f, "")} for f, p in sorted(self.points.items()) if p]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"sequence": self.clip.seq.name, "league": self.league,
                                         "anchors": anchors}, indent=1))
        self.flash(f"saved {self.path.name}")

    def flash(self, msg: str, seconds: float = 2.0) -> None:
        self.message, self.message_until = msg, time.time() + seconds

    def move_frame(self, delta: int) -> None:
        """Replace the current frame by one ``delta`` frames away. Its points belong to THIS image
        (the camera moves), so they are dropped — after a warning, if there are any."""
        new = int(np.clip(self.frame + delta, 1, self.clip.length))
        if new in self.points:
            return
        if self.points[self.frame] and self.confirm_move != (self.frame, delta):
            self.confirm_move = (self.frame, delta)
            self.flash(f"{len(self.points[self.frame])} points would be DROPPED: press {'+' if delta > 0 else '-'} "
                       "again to move", 4.0)
            return
        self.confirm_move = None
        del self.points[self.frame]
        self.points[new] = {}
        self.frames = sorted(self.points)
        self.i = self.frames.index(new)
        self.load()
        self.flash(f"now on frame {new} (empty)")

    # ------------------------------------------------------------------ geometry
    def homography(self) -> np.ndarray | None:
        pts = self.points[self.frame]
        return homography_from_points(pts, self.spec) if len(pts) >= 4 else None

    def reproj(self, H) -> dict[str, float]:
        out = {}
        for k, (u, v) in self.points[self.frame].items():
            p = cv2.perspectiveTransform(np.array([[self.kp[k]]], float), H)[0, 0]
            out[k] = float(np.hypot(p[0] - u, p[1] - v))
        return out

    # ------------------------------------------------------------------ drawing
    def _plan(self, scale=22.0, pad=40, top=40):
        w, h = int(self.spec.length * scale) + 2 * pad, int(self.spec.width * scale) + 2 * pad + top
        img = np.full((h, w, 3), (70, 120, 170), np.uint8)
        off = np.array([pad, pad + top], float)

        def to_px(p):
            return (np.asarray(p, float) * scale + off).astype(np.int32)

        def from_px(p):
            return (np.asarray(p, float) - off) / scale

        for pl in lines(self.spec):
            cv2.polylines(img, [to_px(pl)], False, (255, 255, 255), 2, cv2.LINE_AA)
        return img, to_px, from_px

    def draw_plan(self) -> np.ndarray:
        m = self.plan.copy()
        placed = self.points[self.frame]
        for k, xy in self.kp.items():
            c = tuple(int(v) for v in self.to_px(xy))
            col = (0, 0, 255) if k == self.sel else (0, 200, 0) if k in placed else (40, 40, 40)
            cv2.circle(m, c, 9 if k == self.sel else 6, col, -1, cv2.LINE_AA)
            cv2.putText(m, k, (c[0] + 8, c[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(m, f"selected: {self.sel}   ] next  [ prev", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return m

    def _court(self, img, H, col, thick):
        for pl in lines(self.spec):
            p = cv2.perspectiveTransform(pl[None].astype(np.float64), H)[0]
            cv2.polylines(img, [np.clip(p, -1e4, 1e4).astype(np.int32)], False, col, thick, cv2.LINE_AA)

    def draw_frame(self) -> np.ndarray:
        img = self.img.copy()
        H = self.homography()
        err = {}
        if H is not None and self.show_court:
            self._court(img, H, (0, 255, 255), 1)
            err = self.reproj(H)
            if self.refined is not None:
                self._court(img, self.refined[0], (255, 255, 0), 1)
        for k, (u, v) in self.points[self.frame].items():
            col = (0, 0, 255) if k == self.sel else (0, 255, 0)
            cv2.drawMarker(img, (int(round(u)), int(round(v))), col, cv2.MARKER_CROSS, 14, 2)
            label = f"{k}" + (f" {err[k]:.1f}px" if k in err else "")
            cv2.putText(img, label, (int(u) + 6, int(v) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        status = f"frame {self.frame} ({self.i + 1}/{len(self.frames)})  points {len(self.points[self.frame])}"
        if err:
            status += f"  median reproj {np.median(list(err.values())):.1f}px"
        if self.refined is not None:
            status += f"  lines: {self.refined[1]:.2f} -> {self.refined[2]:.2f}"
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if time.time() < self.message_until:
            h = img.shape[0]
            cv2.rectangle(img, (0, h - 36), (img.shape[1], h), (0, 0, 0), -1)
            cv2.putText(img, self.message, (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2,
                        cv2.LINE_AA)
        return self.draw_loupe(img)

    def loupe_box(self) -> tuple[int, int, int]:
        """(x0, y0, size) of the magnifier on screen: top right, or top left when the cursor is there."""
        size = LOUPE * ZOOM
        x0 = self.img.shape[1] - size - 10 if self.mouse[0] < self.img.shape[1] - size - 40 else 10
        return x0, 44, size

    def draw_loupe(self, img: np.ndarray) -> np.ndarray:
        x0, y0, size = self.loupe_box()
        mx, my = self.mouse
        if x0 <= mx < x0 + size and y0 <= my < y0 + size:        # cursor inside the loupe: keep its view
            mx, my = self.loupe_centre
        self.loupe_centre = (mx, my)
        src = cv2.copyMakeBorder(self.img, LOUPE, LOUPE, LOUPE, LOUPE, cv2.BORDER_CONSTANT)
        crop = src[my + LOUPE // 2: my + 3 * LOUPE // 2, mx + LOUPE // 2: mx + 3 * LOUPE // 2]
        big = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
        # what is drawn on the frame (points, court) is drawn on the loupe too, at the zoomed scale
        ov = cv2.copyMakeBorder(img, LOUPE, LOUPE, LOUPE, LOUPE, cv2.BORDER_CONSTANT)
        ov = cv2.resize(ov[my + LOUPE // 2: my + 3 * LOUPE // 2, mx + LOUPE // 2: mx + 3 * LOUPE // 2],
                        (size, size), interpolation=cv2.INTER_NEAREST)
        big = cv2.addWeighted(big, 0.6, ov, 0.4, 0)
        c = size // 2
        cv2.line(big, (c, 0), (c, size), (0, 0, 255), 1)
        cv2.line(big, (0, c), (size, c), (0, 0, 255), 1)
        img[y0:y0 + size, x0:x0 + size] = big
        cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 1)
        return img

    # ------------------------------------------------------------------ events
    def to_image(self, x: int, y: int) -> tuple[float, float]:
        """Screen click -> image point (sub-pixel when the click is inside the magnifier)."""
        x0, y0, size = self.loupe_box()
        if x0 <= x < x0 + size and y0 <= y < y0 + size:
            cx, cy = self.loupe_centre
            return cx + (x - x0 - size / 2 + 0.5) / ZOOM, cy + (y - y0 - size / 2 + 0.5) / ZOOM
        return float(x), float(y)

    def on_frame(self, ev, x, y, flags, _) -> None:
        if ev == cv2.EVENT_MOUSEMOVE:
            x0, y0, size = self.loupe_box()
            if not (x0 <= x < x0 + size and y0 <= y < y0 + size):
                self.mouse = (x, y)
        elif ev == cv2.EVENT_LBUTTONDOWN:
            self.points[self.frame][self.sel] = self.to_image(x, y)
            self.refined = None
            self.step(+1)
        elif ev == cv2.EVENT_RBUTTONDOWN and self.points[self.frame]:
            u, v = self.to_image(x, y)
            k = min(self.points[self.frame], key=lambda n: np.hypot(self.points[self.frame][n][0] - u,
                                                                    self.points[self.frame][n][1] - v))
            del self.points[self.frame][k]
            self.refined = None

    def on_plan(self, ev, x, y, flags, _) -> None:
        if ev == cv2.EVENT_LBUTTONDOWN:
            p = self.from_px((x, y))
            self.sel = min(self.kp, key=lambda k: np.hypot(self.kp[k][0] - p[0], self.kp[k][1] - p[1]))

    def step(self, d: int) -> None:
        self.sel = self.names[(self.names.index(self.sel) + d) % len(self.names)]

    def run(self) -> None:
        cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("court", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.on_frame)
        cv2.setMouseCallback("court", self.on_plan)
        self.loupe_centre = (0, 0)
        while True:
            cv2.imshow("frame", self.draw_frame())
            cv2.imshow("court", self.draw_plan())
            k = cv2.waitKeyEx(30)
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                self.save()
            elif k in (ord("n"), ord("p")):
                self.save()
                self.i = (self.i + (1 if k == ord("n") else -1)) % len(self.frames)
                self.load()
            elif k in (ord("+"), ord("-")):
                self.move_frame(10 if k == ord("+") else -10)
            elif k == ord("]"):
                self.step(+1)
            elif k == ord("["):
                self.step(-1)
            elif k == ord("d"):
                self.points[self.frame].pop(self.sel, None)
                self.refined = None
            elif k == ord("h"):
                self.show_court = not self.show_court
            elif k == ord("r") and len(self.points[self.frame]) >= 4:
                self.refined = fit_anchor(self.img, Anchor(self.frame, self.points[self.frame]), self.spec,
                                          self.clip.people(self.frame), ignore_below=575)
            elif k in (KEYS_LEFT, KEYS_RIGHT, KEYS_UP, KEYS_DOWN) and self.sel in self.points[self.frame]:
                du, dv = {KEYS_LEFT: (-.5, 0), KEYS_RIGHT: (.5, 0), KEYS_UP: (0, -.5), KEYS_DOWN: (0, .5)}[k]
                u, v = self.points[self.frame][self.sel]
                self.points[self.frame][self.sel] = (u + du, v + dv)
                self.refined = None
        self.save()
        cv2.destroyAllWindows()
        print(f"saved {self.path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence")
    ap.add_argument("--league", choices=sorted(SPECS))
    ap.add_argument("--frames", type=int, nargs="*", help="frames to annotate (default: 3 spread over the clip)")
    ap.add_argument("--manifest", type=Path, help="test_set.json: annotate every clip it lists, one after another")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.manifest:
        jobs = [(c["sequence"], c["league"], c["frames"]) for c in json.loads(args.manifest.read_text())["annotate"]]
    elif args.sequence and args.league:
        jobs = [(args.sequence, args.league, args.frames)]
    else:
        ap.error("give --manifest, or --sequence and --league")
    for seq, league, frames in jobs:
        clip = Clip(seq)
        frames = frames or [int(clip.length * q) for q in (0.15, 0.5, 0.85)]
        print(f"{seq} ({league}): frames {frames}")
        Annotator(clip, league, frames, args.out / f"anchors_{seq}.json").run()


if __name__ == "__main__":
    main()
