"""Tracks on the court plan, in metres (court chapter C08, through the adapter).

The tracker writes boxes in pixels with an identity. With the clip's verified
homographies (C07, ``scripts/court/verify_clip_camera.py``) every box's feet —
the bottom centre — become a position on the court in metres, so each identity
gets a trajectory, a distance run and speeds.

Only frames with a homography the clip camera explains are used: a frame
without one gets no position (a gap in the trajectory), never a guess. Clips
that C07 sent to review have no homographies at all and produce no map.

``H_ptz`` (the homography re-drawn by the clip camera) is preferred over the raw
one: it is just as close to the hand reference (6.2 vs 5.8 cm on NCAA) and, being
4 numbers per frame instead of 8, it cannot wobble in ways no camera can.
Feet positions are then smoothed per track (Savitzky–Golay) because the bottom
of a box jitters by a few pixels from frame to frame, i.e. tens of centimetres
far from the camera, which would inflate distances and speeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..court.leagues import SPECS, CourtSpec, lines


@dataclass
class TopDownConfig:
    dir: str = "outputs/court_verified/auto_edges"   # <dir>/<sequence>/homographies.npz (C07 output)
    use_ptz: bool = True                             # H_ptz (clip camera) instead of the raw homography
    margin_m: float = 1.0                            # positions further outside the court are dropped
    smooth: int = 9                                  # Savitzky–Golay window (frames); 0 = off
    max_gap: int = 5                                 # a track's run is split at gaps longer than this

    @classmethod
    def from_dict(cls, d: dict | None) -> TopDownConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def load_homographies(cfg: TopDownConfig, sequence: str) -> tuple[CourtSpec, dict[int, np.ndarray]] | None:
    path = Path(cfg.dir) / sequence / "homographies.npz"
    if not path.is_file():
        return None
    z = np.load(path)
    key = "H_ptz" if cfg.use_ptz and "H_ptz" in z.files else "H"
    return SPECS[str(z["league"])], {int(f): H for f, H in zip(z["frames"], z[key], strict=True)}


def project_tracks(tracks: dict[int, tuple[np.ndarray, np.ndarray]], H: dict[int, np.ndarray], spec: CourtSpec,
                   cfg: TopDownConfig | None = None) -> pd.DataFrame:
    """frame -> (ids, TLBR boxes)  ->  one row per (frame, id): feet in metres, raw and smoothed."""
    cfg = cfg or TopDownConfig()
    rows = []
    for f, (ids, boxes) in sorted(tracks.items()):
        if f not in H or not len(ids):
            continue
        feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
        m = cv2.perspectiveTransform(feet[None].astype(np.float64), np.linalg.inv(H[f]))[0]
        for i, (x, y) in zip(ids, m, strict=True):
            if -cfg.margin_m <= x <= spec.length + cfg.margin_m and -cfg.margin_m <= y <= spec.width + cfg.margin_m:
                rows.append((int(f), int(i), float(x), float(y)))
    df = pd.DataFrame(rows, columns=["frame", "id", "x_raw", "y_raw"])
    df["x"], df["y"] = df["x_raw"], df["y_raw"]
    if cfg.smooth >= 3 and len(df):
        w = cfg.smooth | 1
        for _, g in df.groupby("id"):
            run = (g["frame"].diff() > cfg.max_gap + 1).cumsum()
            for _, r in g.groupby(run):
                if len(r) >= w:
                    df.loc[r.index, "x"] = savgol_filter(r["x_raw"].to_numpy(), w, 2)
                    df.loc[r.index, "y"] = savgol_filter(r["y_raw"].to_numpy(), w, 2)
    return df


def track_stats(df: pd.DataFrame, fps: float, max_gap: int = 5) -> pd.DataFrame:
    """Per identity: frames on the map, distance (m), mean and top speed (m/s) over consecutive frames."""
    out = []
    for i, g in df.sort_values("frame").groupby("id"):
        df_ = g["frame"].diff().to_numpy()
        step = np.hypot(g["x"].diff(), g["y"].diff()).to_numpy()
        ok = (df_ >= 1) & (df_ <= max_gap + 1)
        speed = step[ok] / (df_[ok] / fps)
        out.append({"id": int(i), "frames": len(g), "distance_m": float(step[ok].sum()),
                    "speed_mean_ms": float(speed.mean()) if len(speed) else np.nan,
                    "speed_p95_ms": float(np.percentile(speed, 95)) if len(speed) else np.nan})
    return pd.DataFrame(out)


def plan_image(spec: CourtSpec, scale: float = 16.0, pad: int = 24):
    """Empty court plan and the metres -> pixels function."""
    w, h = int(spec.length * scale) + 2 * pad, int(spec.width * scale) + 2 * pad
    img = np.full((h, w, 3), (70, 120, 170), np.uint8)

    def to_px(p):
        return (np.asarray(p, float) * scale + pad).astype(np.int32)

    for pl in lines(spec):
        cv2.polylines(img, [to_px(pl)], False, (255, 255, 255), 2, cv2.LINE_AA)
    return img, to_px


def render_video(seq, tracks: dict[int, tuple[np.ndarray, np.ndarray]], df: pd.DataFrame, spec: CourtSpec,
                 H: dict[int, np.ndarray], out: Path, width: int = 1280, trail: int = 25) -> None:
    """Broadcast frame with tracks and projected court lines, the court plan with each identity underneath."""
    from ..viz import draw_tracks, id_color

    class _Out:                                                   # what draw_tracks reads from a track
        def __init__(self, i, b):
            self.track_id, self.tlbr = int(i), b

    plan, to_px = plan_image(spec)
    by_frame = {f: g for f, g in df.groupby("frame")}
    writer = None
    for f in range(1, seq.length + 1):
        img = cv2.imread(str(seq.frame_path(f)))
        ids, boxes = tracks.get(f, (np.zeros(0, int), np.zeros((0, 4))))
        img = draw_tracks(img, [_Out(i, b) for i, b in zip(ids, boxes, strict=True)], f)
        m = plan.copy()
        if f in H:
            for pl in lines(spec):
                p = cv2.perspectiveTransform(pl[None].astype(np.float64), H[f])[0].astype(np.int32)
                cv2.polylines(img, [p], False, (255, 255, 0), 1, cv2.LINE_AA)
            recent = df[(df["frame"] > f - trail) & (df["frame"] <= f)]
            for i, g in recent.groupby("id"):
                cv2.polylines(m, [to_px(g[["x", "y"]].to_numpy())], False, id_color(int(i)), 2, cv2.LINE_AA)
            for _, r in by_frame.get(f, pd.DataFrame(columns=df.columns)).iterrows():
                c = tuple(int(v) for v in to_px([r["x"], r["y"]]))
                cv2.circle(m, c, 8, id_color(int(r["id"])), -1, cv2.LINE_AA)
                cv2.putText(m, str(int(r["id"])), (c[0] + 9, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(m, "no verified court in this frame", (30, m.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
        top = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))
        bottom = cv2.resize(m, (width, int(m.shape[0] * width / m.shape[1])))
        frame = np.vstack([top, bottom])
        if writer is None:
            out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), seq.fps, frame.shape[1::-1])
        writer.write(frame)
    if writer is not None:
        writer.release()
