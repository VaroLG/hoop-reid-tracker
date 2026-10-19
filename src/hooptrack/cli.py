"""Command-line interface.

    hooptrack detect   --config configs/default.yaml                 # cache YOLO detections
    hooptrack track    --config configs/default.yaml --set detector.type=cached
    hooptrack evaluate --run outputs/runs/default --hota

Every command accepts ``--set key=value`` overrides (see ``config.py``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config, save_config
from .data.sportsmot import list_sequences, load_mot_txt


def _sequences(cfg: dict):
    d = cfg["data"]
    seqs = list_sequences(d["root"], d["split"], d.get("sport"), d["splits_dir"])
    if d.get("only"):  # e.g. --set "data.only=[v_00HRwkvvjtQ_c001]" for a quick demo
        seqs = [s for s in seqs if s.name in set(d["only"])]
    if not seqs:
        raise SystemExit(
            f"No sequences found for split={d['split']} sport={d.get('sport')} under {d['root']}.\n"
            "Run `python scripts/download_data.py` first."
        )
    return seqs


# --------------------------------------------------------------------- detect
def cmd_detect(cfg: dict) -> None:
    import cv2
    from tqdm import tqdm

    from .detection import YoloDetector, save_detections

    d = cfg["detector"]
    det = YoloDetector(d["weights"], d["conf"], d["iou"], d["imgsz"], d.get("device"))
    for seq in _sequences(cfg):
        rows = []
        for idx, path in tqdm(seq.frames(), total=seq.length, desc=seq.name):
            boxes, scores = det(cv2.imread(str(path)))
            rows.append((idx, boxes, scores))
        save_detections(Path(d["cache_dir"]) / f"{seq.name}.txt", rows)
    print(f"Detections cached in {d['cache_dir']}")


# ---------------------------------------------------------------------- track
def cmd_track(cfg: dict) -> Path:
    from .detection import build_detector
    from .detection.filters import build_filters
    from .detection.teams import build_team_classifier
    from .pipeline import run_sequence
    from .reid import build_reid
    from .tracking import TrackerConfig

    out_dir = Path(cfg["output"]["dir"])
    save_config(cfg, out_dir / "config.yaml")
    detector = build_detector(cfg["detector"])
    reid = build_reid(cfg["reid"])
    tcfg = TrackerConfig.from_dict(cfg["tracker"])
    # Both options default to OFF so older configs reproduce their original runs.
    filters = build_filters(cfg.get("filters"))
    only_usable = bool(cfg["reid"].get("only_usable_boxes", False))
    team_clf = build_team_classifier(cfg.get("teams"))

    timings = {}
    for seq in _sequences(cfg):
        video = out_dir / "videos" / f"{seq.name}.mp4" if cfg["output"].get("save_video") else None
        t = run_sequence(seq, detector, reid, tcfg, out_dir / "results" / f"{seq.name}.txt", video,
                         filters=filters, only_usable_boxes=only_usable, team_classifier=team_clf)
        timings[seq.name] = t.summary()
        s = timings[seq.name]
        print(f"{seq.name}: {s['fps_total']:.1f} FPS | ids={s['ids_created']} "
              f"| revivals={s['memory_revivals']}")

    pd.DataFrame(timings).T.to_csv(out_dir / "timing.csv")
    print(f"Results written to {out_dir / 'results'}")
    return out_dir


# ------------------------------------------------------------------- evaluate
def cmd_evaluate(cfg: dict, run_dir: Path, hota: bool) -> dict:
    from .eval import clear_and_identity, reentry_recovery
    from .eval.metrics import ReentryResult

    results = run_dir / "results"
    seqs = [s for s in _sequences(cfg) if (results / f"{s.name}.txt").is_file()]
    per_seq, reentry = {}, ReentryResult()
    for s in seqs:
        gt, pred = s.load_gt(), load_mot_txt(results / f"{s.name}.txt")
        per_seq[s.name] = (gt, pred)
        reentry = reentry + reentry_recovery(gt, pred)

    table = clear_and_identity(per_seq)
    overall = table.loc["OVERALL"]
    summary = {
        "IDF1": 100 * float(overall["idf1"]),
        "MOTA": 100 * float(overall["mota"]),
        "IDSW": int(overall["num_switches"]),
        "Frag": int(overall["num_fragmentations"]),
        "reentry_events": reentry.events,
        "reentry_recovered_%": 100 * reentry.rate if reentry.events else None,
        "reentry_stolen": reentry.stolen,
    }
    if hota:
        from .eval.trackeval_runner import run_trackeval

        te = run_trackeval(seqs, results)
        te.to_csv(run_dir / "trackeval.csv", float_format="%.2f")
        summary.update({k: float(te.loc["OVERALL", k]) for k in ("HOTA", "DetA", "AssA")})

    table.to_csv(run_dir / "motmetrics.csv", float_format="%.4f")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hooptrack", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("detect", "track", "evaluate"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default="configs/default.yaml")
        sp.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
        if name == "evaluate":
            sp.add_argument("--run", type=Path, default=None, help="run dir (default: output.dir)")
            sp.add_argument("--hota", action="store_true", help="also run TrackEval (HOTA/AssA)")
        if name == "track":
            sp.add_argument("--eval", action="store_true", help="evaluate right after tracking")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)

    if args.command == "detect":
        cmd_detect(cfg)
    elif args.command == "track":
        run_dir = cmd_track(cfg)
        if args.eval:
            cmd_evaluate(cfg, run_dir, hota=False)
    else:
        cmd_evaluate(cfg, args.run or Path(cfg["output"]["dir"]), args.hota)


if __name__ == "__main__":
    main()
