"""External baseline (experiment 024): Deep-EIoU's tracker on OUR detections and OUR ReID model.

Deep-EIoU (Huang et al., WACV-W 2024) is the reference tracker on SportsMOT and the one 012-014 borrowed ideas from.
Its published numbers use a YOLOX-X detector and its own OSNet, on the whole SportsMOT test set (basketball, football,
volleyball), so they cannot be compared with ours. Here only the TRACKER changes: same 15 basketball val clips, same
players-only detector (011, cached detections), same OSNet (012). The difference measures our tracker against theirs.

The official code is used unchanged (no licence file in their repository, so it is not copied here):

    git clone https://github.com/hsiangwei0903/Deep-EIoU.git third_party/Deep-EIoU
    pip install lap loguru          # cython_bbox is optional: a NumPy fallback is used without it

Settings: the defaults of their ``tools/sport_track.py`` (high 0.6, low 0.1, new track 0.7, buffer 60, match 0.8,
proximity 0.5, appearance 0.25, ``frame_rate=30`` as in their script), and their interpolation
(``tools/sport_interpolation.py``, defaults), reported with and without it. Our detector's scores are not calibrated
like YOLOX's, so the thresholds may not be optimal for it; they are NOT tuned here (no tuning on val).

Usage (repo root):
    python scripts/run_deep_eiou.py --config configs/player.yaml --deep-eiou third_party/Deep-EIoU `
        --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth --out outputs/runs/024_deep_eiou
Outputs: ``<out>/online/results/<clip>.txt`` and ``<out>/interpolated/results/<clip>.txt`` (evaluate with
``scripts/eval_hota.py``). ReID embeddings of every detection are cached in ``<out>/embeddings/<clip>.npz``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.eval.trackeval_runner import _patch_numpy_aliases
from hooptrack.reid import build_reid

# the defaults of Deep-EIoU's tools/sport_track.py
DEEP_EIOU_ARGS = dict(track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=60,
                      match_thresh=0.8, aspect_ratio_thresh=1.6, min_box_area=10, with_reid=True,
                      proximity_thresh=0.5, appearance_thresh=0.25)


def install_bbox_fallback() -> None:
    """Their ``matching.py`` imports ``cython_bbox`` (a compiled IoU). It needs a C compiler on Windows. If it is not
    installed, register a NumPy module with the same function, so their code runs unchanged."""
    try:
        import cython_bbox  # noqa: F401
    except ImportError:
        import types

        def bbox_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            """IoU matrix of TLBR boxes, same convention as cython_bbox (+1 on widths and heights)."""
            a, b = np.asarray(a, float), np.asarray(b, float)
            iw = np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]) + 1
            ih = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]) + 1
            inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
            area_a = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
            area_b = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)
            return inter / (area_a[:, None] + area_b[None, :] - inter)

        mod = types.ModuleType("cython_bbox")
        mod.bbox_overlaps = bbox_overlaps
        sys.modules["cython_bbox"] = mod


def load_dets(path: Path) -> dict[int, np.ndarray]:
    """MOT-det cache (frame,-1,x,y,w,h,score) -> {frame: (N, 5) x1,y1,x2,y2,score}."""
    raw = np.loadtxt(path, delimiter=",", ndmin=2)
    out = {}
    for f in np.unique(raw[:, 0]).astype(int):
        r = raw[raw[:, 0] == f]
        out[f] = np.c_[r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5], r[:, 6]]
    return out


def embeddings(seq, dets: dict, reid, cache: Path) -> dict[int, np.ndarray]:
    """ReID vector of every cached detection (L2-normalised), cached per clip."""
    if cache.is_file():
        z = np.load(cache)
        return {int(f): z[str(f)] for f in z.files}
    out = {}
    for f in range(1, seq.length + 1):
        d = dets.get(f)
        if d is None or not len(d):
            continue
        e = reid(cv2.imread(str(seq.frame_path(f))), d[:, :4])
        out[f] = (e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **{str(f): v for f, v in out.items()})
    return out


# --------------------------------------------------------------------------- 025: ablations
# Each ablation removes ONE piece of Deep-EIoU, from outside: their files stay unchanged, we replace a function of
# their modules in memory (monkeypatching) or change one argument. Their values are kept everywhere else, so the
# difference with the full tracker measures that piece alone. No ablation tunes a value.
ABLATIONS = {
    "no_reid": "appearance off: association by expanded IoU only (with_reid=False)",
    "no_low": "no second stage with low-score boxes (low threshold = high threshold, 0.6)",
    "no_expand": "plain IoU instead of expanded IoU, in every stage",
    "expand_014": "expansion as implemented in our 014 (each side + e x size: 0.7 -> 2.4x) instead of theirs (3.4x)",
    "kalman": "associate the Kalman-PREDICTED box, as our tracker does, instead of the last detected box",
    "kalman_gmc": "kalman + camera-motion compensation (our sparse optical flow GMC), as our tracker does",
    "confirmed_only": "do not output a track on its birth frame (only tracks matched at least twice)",
}


def apply_ablations(names: list[str], tracker_args: dict) -> None:
    """Patch Deep-EIoU's modules in memory for the requested ablations (their files are not touched).

    Why each patch is enough:
    - ``matching.eiou_distance`` looks up ``eious`` and ``expand`` in its own module at call time, so replacing
      ``matching.eious`` / ``matching.expand`` changes every stage at once.
    - Their association reads ``track.last_tlbr`` (the last DETECTED box). Making that property return ``tlbr``
      (the Kalman state) switches every stage to the Kalman box; the prediction step itself is added in the loop
      of ``main`` (their ``update`` never calls ``multi_predict``).
    """
    from tracker import matching
    from tracker.Deep_EIoU import STrack

    if "no_reid" in names:
        tracker_args["with_reid"] = False
    if "no_low" in names:
        tracker_args["track_low_thresh"] = tracker_args["track_high_thresh"]
    if "no_expand" in names:
        def plain_ious(atlbrs, btlbrs, e):                       # same as their eious, minus the expansion
            if len(atlbrs) == 0 or len(btlbrs) == 0:
                return np.zeros((len(atlbrs), len(btlbrs)))
            return matching.bbox_ious(np.ascontiguousarray(atlbrs, dtype=float),
                                      np.ascontiguousarray(btlbrs, dtype=float))
        matching.eious = plain_ious
    if "expand_014" in names:
        def expand_each_side(tlbr, e):                           # x1,y1,x2,y2 grown by e x width / height per side
            x1, y1, x2, y2 = tlbr
            w, h = x2 - x1, y2 - y1
            return [x1 - e * w, y1 - e * h, x2 + e * w, y2 + e * h]
        matching.expand = expand_each_side
    if "kalman" in names or "kalman_gmc" in names:
        # Their STrack.update() never corrects the Kalman filter (only re_activate does): Deep-EIoU does not use
        # the filter at all, it matches the last detected box. To test a real motion model, the filter must also be
        # corrected with every matched detection, as in our tracker; otherwise the prediction drifts away.
        original_update = STrack.update

        def update_with_kalman(self, new_track, frame_id):
            self.mean, self.covariance = self.kalman_filter.update(
                self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
            original_update(self, new_track, frame_id)

        STrack.update = update_with_kalman
        STrack.last_tlbr = property(lambda self: self.tlbr)      # Kalman box (the detection box before activation)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--deep-eiou", type=Path, default=Path("third_party/Deep-EIoU/Deep-EIoU"),
                    help="the Deep-EIoU folder that contains tracker/ and tools/")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/024_deep_eiou"))
    ap.add_argument("--ablate", nargs="*", default=[], choices=sorted(ABLATIONS),
                    help="025: remove one piece of Deep-EIoU (see ABLATIONS)")
    ap.add_argument("--emb-cache", type=Path, default=None,
                    help="folder of cached embeddings to reuse (default <out>/embeddings)")
    ap.add_argument("--no-interp", action="store_true", help="skip their interpolation (online results only)")
    args = ap.parse_args()

    root = args.deep_eiou if (args.deep_eiou / "tracker").is_dir() else args.deep_eiou / "Deep-EIoU"
    sys.path.insert(0, str(root))
    install_bbox_fallback()
    _patch_numpy_aliases()                                       # their code uses np.float (removed in NumPy 1.24)
    from tracker.Deep_EIoU import Deep_EIoU, STrack, joint_stracks  # noqa: E402  (their code, unchanged)

    tracker_args = dict(DEEP_EIOU_ARGS)
    apply_ablations(args.ablate, tracker_args)
    predict = bool({"kalman", "kalman_gmc"} & set(args.ablate))
    print("ablations:", ", ".join(f"{a} ({ABLATIONS[a]})" for a in args.ablate) or "none", flush=True)

    cfg = load_config(args.config, args.set)
    seqs = _sequences(apply_overrides(cfg, ["data.only=null"]))
    if args.clips:
        seqs = [s for s in seqs if s.name in set(args.clips)]
    reid = build_reid(cfg["reid"])
    emb_dir = args.emb_cache or args.out / "embeddings"
    sct = args.out / "SCT"                                       # their folder names, for the interpolation script
    sct.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        dets = load_dets(Path(cfg["detector"]["cache_dir"]) / f"{seq.name}.txt")
        embs = embeddings(seq, dets, reid, emb_dir / f"{seq.name}.npz")
        tracker = Deep_EIoU(SimpleNamespace(**tracker_args), frame_rate=30)
        gmc = None
        if "kalman_gmc" in args.ablate:
            from hooptrack.tracking.gmc import SparseOpticalFlowGMC
            gmc = SparseOpticalFlowGMC(cfg["tracker"].get("gmc_downscale", 2))
        lines = []
        for f in range(1, seq.length + 1):
            d = dets.get(f, np.zeros((0, 5)))
            e = embs.get(f, np.zeros((0, 512), np.float32))
            if predict:                                          # the step their update() does not have
                pool = joint_stracks(tracker.tracked_stracks, tracker.lost_stracks)
                STrack.multi_predict(pool)
                if gmc is not None:                              # warp predictions by the camera motion
                    high = d[d[:, 4] > tracker_args["track_high_thresh"], :4] if len(d) else None
                    STrack.multi_gmc(pool, gmc.apply(cv2.imread(str(seq.frame_path(f))), high))
            for t in tracker.update(d, e):
                if "confirmed_only" in args.ablate and not t.is_activated:
                    continue
                x, y, w, h = t.last_tlwh
                if w * h > tracker_args["min_box_area"]:
                    lines.append(f"{f},{t.track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{t.score:.2f},-1,-1,-1\n")
        (sct / f"{seq.name}.txt").write_text("".join(lines))
        on = args.out / "online" / "results"
        on.mkdir(parents=True, exist_ok=True)
        (on / f"{seq.name}.txt").write_text("".join(lines))
        print(f"{seq.name}: {len({ln.split(',')[1] for ln in lines})} ids", flush=True)
    if args.no_interp:
        print(f"Saved {args.out}/online")
        return
    # their linear interpolation of short gaps, unchanged (writes <out>/interpolation/).
    # It runs from their folder (cwd=root), so every path passed to it must be absolute.
    done = subprocess.run([sys.executable, str((root / "tools" / "sport_interpolation.py").resolve()),
                           "--root_path", str(args.out.resolve())], cwd=root, capture_output=True, text=True)
    if done.returncode != 0:
        sys.exit(f"Deep-EIoU interpolation failed:\n{done.stderr}")
    ip = args.out / "interpolated" / "results"
    ip.mkdir(parents=True, exist_ok=True)
    for p in (args.out / "interpolation").glob("*.txt"):
        (ip / p.name).write_text(p.read_text())
    print(f"Saved {args.out}/online and {args.out}/interpolated")


if __name__ == "__main__":
    main()
