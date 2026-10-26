"""Compare court-mask sources inside the tracker, on many validation clips (experiment 010).

The question
------------
The court mask decides which detections reach the tracker: a person whose feet
are outside the court (bench, crowd, staff) is dropped before tracking. Three
ways to get that mask exist (``filters.court_source``):

    colour        experiment 003: the floor colour (default, no model)
    homography    experiment 009: court lines -> homography (precomputed per video,
                  only where scripts/court/auto_homographies.py was run)
    segmentation  C06: the court area predicted by the YOLO26n-seg model you trained

This script runs the SAME tracker with ONLY ``court_source`` changed, on every
basketball validation clip, and tabulates what changed. Everything else comes
from the config file, so the comparison is fair by construction.

How it works (and why)
----------------------
1. **Detect once.** YOLO26m at 1280 px is the slowest step (~1 s/frame on CPU).
   Detections do not depend on the court mask, so they are computed ONCE per clip
   and cached (``hooptrack detect``); every variant then reads the cache
   (``detector.type=cached``). Clips already in the cache are skipped.
   The cache folder is the config's ``detector.cache_dir`` (or ``--cache-dir``), and
   ``<cache>/detector.info`` records which weights produced it: a run whose
   ``detector.weights`` differs from it stops, instead of silently tracking on
   another detector's boxes (012 first run: COCO boxes read for the player model).
2. **One folder per variant and clip:** ``<out>/<variant>/<clip>/``. Nothing is
   overwritten (the ``summary.json`` of a shared folder only keeps the LAST clip),
   and an interrupted run resumes where it stopped: finished folders are skipped.
3. **Runs one after another, not in parallel:** parallel jobs fight for the same
   CPU cores, so they are not faster in total, and they make timings meaningless.
4. **Aggregate properly.** MOTA / IDF1 over several clips are NOT the average of
   the per-clip values: they are recomputed from all errors together (``OVERALL``,
   what papers report). Also given per arena, because the arenas behave differently
   (NCAA = the clip the tracker was tuned on; the FIBA arenas = new conditions).

Outputs in ``<out>/``:
    per_clip.csv      one row per (variant, clip): IDF1, MOTA, IDSW, FP, misses, filter stats, FPS
    summary.md        tables (overall, per arena, per clip) with the change vs the first variant
    summary.json      the same numbers, machine readable
    compare/<clip>.mp4  side-by-side videos (only with --video): left = first variant, right = second;
                        the green polygon is the court mask the filter used in that frame

Usage (repo root, environment active):
    python scripts/run_court_source_eval.py                                  # colour vs segmentation, 15 clips
    python scripts/run_court_source_eval.py --clips v_5ekaksddqrc_c002 v_BgwzTUxJaeU_c012 --video
    python scripts/run_court_source_eval.py --variants colour segmentation homography
    python scripts/run_court_source_eval.py --summary-only                   # re-tabulate finished runs
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from hooptrack.cli import _sequences, cmd_detect, cmd_evaluate, cmd_track
from hooptrack.config import apply_overrides, load_config
from hooptrack.data.sportsmot import load_mot_txt
from hooptrack.eval import clear_and_identity

# The four validation arenas of SportsMOT basketball, by YouTube video id (prefix of the clip name).
ARENAS = {
    "00HRwkvvjtQ": "NCAA (dev arena)",
    "5ekaksddqrc": "FIBA Tenerife",
    "BgwzTUxJaeU": "FIBA Istanbul",
    "4r8QL_wglzQ": "FIBA Changsha",
}

# Columns kept from motmetrics, and how they are shown.
METRICS = {"idf1": "IDF1", "mota": "MOTA", "num_switches": "IDSW", "num_false_positives": "FP", "num_misses": "Misses"}
# Lower is better for these (used to colour the deltas in the markdown table).
LOWER_IS_BETTER = {"IDSW", "FP", "Misses"}


def arena_of(clip: str) -> str:
    vid = clip[2:].rsplit("_c", 1)[0]          # "v_5ekaksddqrc_c002" -> "5ekaksddqrc"
    return ARENAS.get(vid, vid)


# ---------------------------------------------------------------------------- 1. detect once
def check_cache_detector(base: dict, cache_dir: Path) -> None:
    """The cache must come from the detector of the config: refuse to mix detectors."""
    weights = str(base["detector"]["weights"])
    meta = cache_dir / "detector.info"
    if meta.is_file():
        cached = meta.read_text(encoding="utf-8").strip()
        if Path(cached).name != Path(weights).name:
            raise SystemExit(f"{cache_dir} holds detections of {cached}, but the config uses {weights}.\n"
                             "Use the matching --cache-dir (or detector.cache_dir in the config).")
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        meta.write_text(weights + "\n", encoding="utf-8")
        print(f"[detect] {meta} written: this cache = detections of {weights}")


def ensure_detections(base: dict, clips: list[str], cache_dir: Path) -> None:
    """Run YOLO on every clip that has no cached detections yet."""
    check_cache_detector(base, cache_dir)
    missing = [c for c in clips if not (cache_dir / f"{c}.txt").is_file()]
    if not missing:
        print(f"[detect] all {len(clips)} clips already cached in {cache_dir}")
        return
    print(f"[detect] {len(missing)} clips without detections -> running YOLO once per clip (slow step)")
    for i, clip in enumerate(missing, 1):
        t0 = time.time()
        cfg = apply_overrides(base, ["detector.type=yolo", f"detector.cache_dir={cache_dir.as_posix()}",
                                     f"data.only=[{clip}]"])
        cmd_detect(cfg)
        print(f"[detect] {i}/{len(missing)} {clip} done in {(time.time() - t0) / 60:.1f} min")


# ---------------------------------------------------------------------------- 2. track
def run_dir(out: Path, variant: str, clip: str) -> Path:
    return out / variant / clip


def is_done(d: Path, clip: str) -> bool:
    # timing.csv is written last by cmd_track: if it exists, the run finished
    return (d / "results" / f"{clip}.txt").is_file() and (d / "timing.csv").is_file()


def track_one(base: dict, variant: str, clip: str, cache_dir: Path, out: Path, video: bool) -> None:
    d = run_dir(out, variant, clip)
    cfg = apply_overrides(base, [
        "detector.type=cached",
        f"detector.cache_dir={cache_dir.as_posix()}",
        f"data.only=[{clip}]",
        "filters.court_mask=true",
        f"filters.court_source={variant}",      # <- the ONLY thing that changes between variants
        f"output.dir={d.as_posix()}",
        f"output.save_video={'true' if video else 'false'}",
    ])
    cmd_track(cfg)
    cmd_evaluate(cfg, d, hota=False)            # per-clip summary.json / motmetrics.csv in its own folder


# ---------------------------------------------------------------------------- 3. aggregate
def metrics_row(table_row) -> dict:
    out = {}
    for k, name in METRICS.items():
        v = float(table_row[k])
        out[name] = round(100 * v, 1) if k in ("idf1", "mota") else int(v)
    return out


def collect(base: dict, variants: list[str], clips: list[str], out: Path) -> tuple[pd.DataFrame, dict]:
    """Per-clip rows + OVERALL / per-arena aggregates, on the clips finished in EVERY variant."""
    seqs = {s.name: s for s in _sequences(apply_overrides(base, ["data.only=null"]))}
    done = [c for c in clips if all(is_done(run_dir(out, v, c), c) for v in variants)]
    if not done:
        return pd.DataFrame(), {}
    gt = {c: seqs[c].load_gt() for c in done}
    rows, agg = [], {}
    for v in variants:
        pred = {c: load_mot_txt(run_dir(out, v, c) / "results" / f"{c}.txt") for c in done}
        table = clear_and_identity({c: (gt[c], pred[c]) for c in done})
        for c in done:
            timing = pd.read_csv(run_dir(out, v, c) / "timing.csv", index_col=0).iloc[0].to_dict()
            row = {"variant": v, "clip": c, "arena": arena_of(c), **metrics_row(table.loc[c]),
                   "fps": round(float(timing.get("fps_total", float("nan"))), 1),
                   "ms_filter": round(float(timing.get("ms_filter", float("nan"))), 1)}
            # filter counters (e.g. court_boxes_removed_%, court_frames_no_court, homography coverage)
            row.update({k: (round(val, 2) if isinstance(val, float) else val)
                        for k, val in timing.items() if k.startswith("court")})
            rows.append(row)
        agg[v] = {"OVERALL": metrics_row(table.loc["OVERALL"])}
        for arena in sorted({arena_of(c) for c in done}):
            sub = [c for c in done if arena_of(c) == arena]
            t = clear_and_identity({c: (gt[c], pred[c]) for c in sub})
            agg[v][arena] = {**metrics_row(t.loc["OVERALL"]), "clips": len(sub)}
    return pd.DataFrame(rows), agg


def fmt_delta(name: str, new, ref) -> str:
    d = new - ref
    if abs(d) < 1e-9:
        return f"{new}"
    good = (d < 0) if name in LOWER_IS_BETTER else (d > 0)
    sign = "+" if d > 0 else ""
    d = round(d, 1) if isinstance(d, float) else d
    return f"{new} ({sign}{d} {'✅' if good else '❌'})"


def write_summary(df: pd.DataFrame, agg: dict, variants: list[str], out: Path) -> None:
    ref = variants[0]
    lines = [f"# Court source comparison — reference: `{ref}`", "",
             f"Clips finished in every variant: {df['clip'].nunique()}. "
             "OVERALL = metrics recomputed on all clips together (not an average).", ""]
    # overall and per arena
    groups = list(next(iter(agg.values())).keys())
    for g in groups:
        lines += [f"## {g}", "", "| Variant | " + " | ".join(METRICS.values()) + " |",
                  "|---" * (len(METRICS) + 1) + "|"]
        for v in variants:
            vals = [fmt_delta(m, agg[v][g][m], agg[ref][g][m]) if v != ref else str(agg[v][g][m])
                    for m in METRICS.values()]
            lines.append(f"| {v} | " + " | ".join(vals) + " |")
        lines.append("")
    # per clip, IDF1 / MOTA / FP side by side
    lines += ["## Per clip", "", "| Clip | Arena | " +
              " | ".join(f"{m} {v}" for m in ("IDF1", "MOTA", "FP") for v in variants) + " |",
              "|---" * (2 + 3 * len(variants)) + "|"]
    for clip, g in df.groupby("clip", sort=True):
        g = g.set_index("variant")
        cells = [str(g.loc[v, m]) for m in ("IDF1", "MOTA", "FP") for v in variants]
        lines.append(f"| {clip} | {g['arena'].iloc[0]} | " + " | ".join(cells) + " |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out / "per_clip.csv", index=False)
    print("\n".join(lines))


# ---------------------------------------------------------------------------- 4. side-by-side videos
def compare_videos(variants: list[str], clips: list[str], out: Path) -> None:
    """Left = first variant, right = second, each at half width; written to <out>/compare/<clip>.mp4."""
    import cv2
    import numpy as np

    a, b = variants[:2]
    (out / "compare").mkdir(parents=True, exist_ok=True)
    for clip in clips:
        pa, pb = (run_dir(out, v, clip) / "videos" / f"{clip}.mp4" for v in (a, b))
        if not (pa.is_file() and pb.is_file()):
            continue
        ca, cb = cv2.VideoCapture(str(pa)), cv2.VideoCapture(str(pb))
        fps = ca.get(cv2.CAP_PROP_FPS) or 25
        w, h = int(ca.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2, int(ca.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2
        dst = out / "compare" / f"{clip}.mp4"
        wr = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (2 * w, h))
        while True:
            ok1, f1 = ca.read()
            ok2, f2 = cb.read()
            if not (ok1 and ok2):
                break
            tiles = []
            for f, name in ((f1, a), (f2, b)):
                f = cv2.resize(f, (w, h))
                cv2.putText(f, name, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                tiles.append(f)
            wr.write(np.hstack(tiles))
        wr.release()
        print(f"[video] {dst}")


# ---------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides applied to EVERY variant")
    ap.add_argument("--variants", nargs="+", default=["colour", "segmentation"],
                    choices=["colour", "homography", "segmentation"],
                    help="the first one is the reference for the deltas")
    ap.add_argument("--clips", nargs="*", default=None, help="default: every basketball clip of the split")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="detection cache (default: the config's detector.cache_dir)")
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/010_court_source"))
    ap.add_argument("--video", action="store_true", help="save tracking videos + side-by-side comparisons")
    ap.add_argument("--force", action="store_true", help="re-run finished (variant, clip) folders")
    ap.add_argument("--summary-only", action="store_true", help="only tabulate what is already finished")
    args = ap.parse_args()

    base = load_config(args.config, args.set)
    clips = args.clips or [s.name for s in _sequences(apply_overrides(base, ["data.only=null"]))]
    args.cache_dir = args.cache_dir or Path(base["detector"].get("cache_dir", "outputs/detections"))
    args.out.mkdir(parents=True, exist_ok=True)

    if "segmentation" in args.variants:
        w = Path((base["filters"].get("segmentation") or {}).get(
            "weights", "outputs/court_runs/c06_yolo26n_seg/weights/best.pt"))
        if not w.is_file():
            raise SystemExit(f"C06 weights not found: {w}")

    print(f"Variants: {args.variants} (reference: {args.variants[0]}) | clips: {len(clips)} | out: {args.out}")
    print(f"Detector: {base['detector']['weights']} | detection cache: {args.cache_dir} | "
          f"ReID: {base['reid']['type']} {base['reid'].get('weights') or ''}")

    if not args.summary_only:
        ensure_detections(base, clips, args.cache_dir)
        total = len(clips) * len(args.variants)
        n, failed = 0, []
        for clip in clips:                       # clip-major: after each clip you can already compare
            for v in args.variants:
                n += 1
                d = run_dir(args.out, v, clip)
                if is_done(d, clip) and not args.force:
                    print(f"[track] {n}/{total} {v:12s} {clip}: already done, skipped")
                    continue
                t0 = time.time()
                print(f"[track] {n}/{total} {v:12s} {clip} ...")
                try:
                    track_one(base, v, clip, args.cache_dir, args.out, args.video)
                    print(f"[track] {n}/{total} {v:12s} {clip} done in {(time.time() - t0) / 60:.1f} min")
                except Exception:                # one broken clip must not stop the other 29 runs
                    traceback.print_exc()
                    failed.append(f"{v}/{clip}")
        if failed:
            print(f"\nFAILED runs (re-run the script to retry): {failed}")

    df, agg = collect(base, args.variants, clips, args.out)
    if df.empty:
        raise SystemExit("No clip finished in every variant yet.")
    write_summary(df, agg, args.variants, args.out)
    if args.video and len(args.variants) >= 2:
        compare_videos(args.variants, clips, args.out)
    print(f"\nSaved {args.out / 'summary.md'}, per_clip.csv, summary.json")


if __name__ == "__main__":
    main()
