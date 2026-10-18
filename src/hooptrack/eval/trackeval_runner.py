"""Official HOTA / AssA / DetA via TrackEval.

TrackEval expects a specific folder layout. This module builds it in a
temporary directory (symlinks, so nothing is copied), runs the evaluator
programmatically and returns a tidy table — the same protocol the SportsMOT
authors use (their ``scripts/sportsmot_to_trackeval.py``).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.sportsmot import Sequence


def _patch_numpy_aliases() -> None:
    """TrackEval still uses ``np.float``/``np.int``, removed in NumPy >= 1.24."""
    for alias, real in (("float", float), ("int", int), ("bool", bool)):
        if not hasattr(np, alias):
            setattr(np, alias, real)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Symlink when possible (no copy); fall back to copying.

    On Windows, creating symlinks needs admin rights or Developer Mode, so a
    plain ``symlink_to`` fails for most users.
    """
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy(src, dst)


def run_trackeval(seqs: list[Sequence], results_dir: str | Path, benchmark: str = "SportsMOT") -> pd.DataFrame:
    """Evaluate every ``<results_dir>/<seq>.txt`` against its ground truth."""
    _patch_numpy_aliases()
    try:
        import trackeval
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pip install 'hooptrack[eval]'  (installs TrackEval)") from e

    results_dir = Path(results_dir)
    split = "eval"
    tracker = "hooptrack"
    tmp = Path(tempfile.mkdtemp(prefix="trackeval_"))
    try:
        gt_root = tmp / "gt" / f"{benchmark}-{split}"
        tr_root = tmp / "trackers" / f"{benchmark}-{split}" / tracker / "data"
        tr_root.mkdir(parents=True)
        seqmap = tmp / "seqmap.txt"
        seqmap.write_text("name\n" + "\n".join(s.name for s in seqs) + "\n")

        for s in seqs:
            (gt_root / s.name / "gt").mkdir(parents=True)
            _link_or_copy(s.gt_path, gt_root / s.name / "gt" / "gt.txt")
            _link_or_copy(s.root / "seqinfo.ini", gt_root / s.name / "seqinfo.ini")
            _link_or_copy(results_dir / f"{s.name}.txt", tr_root / f"{s.name}.txt")

        eval_cfg = trackeval.Evaluator.get_default_eval_config()
        eval_cfg.update({"PRINT_RESULTS": False, "PRINT_CONFIG": False, "OUTPUT_SUMMARY": False,
                         "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "USE_PARALLEL": False})
        ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
        ds_cfg.update({
            "GT_FOLDER": str(tmp / "gt"),
            "TRACKERS_FOLDER": str(tmp / "trackers"),
            "BENCHMARK": benchmark,
            "SPLIT_TO_EVAL": split,
            "TRACKERS_TO_EVAL": [tracker],
            "SEQMAP_FILE": str(seqmap),
            "DO_PREPROC": False,   # SportsMOT has no distractor classes to remove
            "PRINT_CONFIG": False,
        })
        evaluator = trackeval.Evaluator(eval_cfg)
        dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
        metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
        out, _ = evaluator.evaluate([dataset], metrics)

        res = out["MotChallenge2DBox"][tracker]
        rows = []
        for seq_name, r in res.items():
            r = r["pedestrian"]
            rows.append({
                "sequence": "OVERALL" if seq_name == "COMBINED_SEQ" else seq_name,
                "HOTA": 100 * float(np.mean(r["HOTA"]["HOTA"])),
                "DetA": 100 * float(np.mean(r["HOTA"]["DetA"])),
                "AssA": 100 * float(np.mean(r["HOTA"]["AssA"])),
                "IDF1": 100 * float(r["Identity"]["IDF1"]),
                "MOTA": 100 * float(r["CLEAR"]["MOTA"]),
                "IDSW": int(r["CLEAR"]["IDSW"]),
            })
        df = pd.DataFrame(rows).set_index("sequence")
        # OVERALL last
        return pd.concat([df.drop(index="OVERALL"), df.loc[["OVERALL"]]])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
