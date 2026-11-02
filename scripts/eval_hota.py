"""HOTA / DetA / AssA (official TrackEval) of one or more finished runs, on the same clips (experiment 024).

The experiments so far report IDF1 and MOTA (motmetrics). SportsMOT and the papers that use it rank trackers by HOTA,
the geometric mean of a detection score (DetA: are the players found and well placed?) and an association score
(AssA: does each player keep one id?). This script puts every run of the project on that scale.

A run may be either layout: ``<run>/<clip>/results/<clip>.txt`` (evaluation scripts, stitching) or
``<run>/results/<clip>.txt`` (``hooptrack track``). Only clips present in every run are evaluated, so runs are compared
on the same clips.

Usage (repo root):
    python scripts/eval_hota.py --config configs/player.yaml `
        --runs 011b=outputs/runs/011b_no_referee_filter/segmentation 013b=outputs/runs/013b_stitch_15 `
               020=outputs/runs/020_siglip_k2_15 --out outputs/hota_024.csv
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from hooptrack.cli import _sequences
from hooptrack.config import apply_overrides, load_config
from hooptrack.eval.trackeval_runner import run_trackeval


def result_file(run: Path, clip: str) -> Path | None:
    for p in (run / clip / "results" / f"{clip}.txt", run / "results" / f"{clip}.txt"):
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/player.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides (e.g. data.root=...)")
    ap.add_argument("--runs", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--out", type=Path, default=None, help="optional CSV with every clip and run")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    seqs = _sequences(apply_overrides(cfg, ["data.only=null"]))
    runs = dict(r.split("=", 1) for r in args.runs)
    clips = [s for s in seqs if all(result_file(Path(p), s.name) for p in runs.values())]
    print(f"{len(clips)} clips present in every run")
    if not clips:                                             # say which run is empty instead of failing in TrackEval
        for name, path in runs.items():
            n = sum(result_file(Path(path), s.name) is not None for s in seqs)
            print(f"  {name}: {n} of {len(seqs)} clips in {path}")
        raise SystemExit("no clip is present in every run")
    tables = []
    for name, path in runs.items():
        flat = Path(tempfile.mkdtemp(prefix="hota_"))         # TrackEval wants <dir>/<clip>.txt
        try:
            for s in clips:
                shutil.copy(result_file(Path(path), s.name), flat / f"{s.name}.txt")
            t = run_trackeval(clips, flat)
        finally:
            shutil.rmtree(flat, ignore_errors=True)
        tables.append(t.assign(run=name).reset_index())
    df = pd.concat(tables)
    overall = df[df.sequence == "OVERALL"].set_index("run")[["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]]
    print("\n" + overall.round(1).to_string())
    print("\nHOTA per clip:\n" + df[df.sequence != "OVERALL"].pivot(index="sequence", columns="run", values="HOTA")
          [list(runs)].round(1).to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.round(2).to_csv(args.out, index=False)
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
