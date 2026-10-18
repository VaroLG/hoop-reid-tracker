# Experiment log

This project is developed **result by result**: each change is motivated by the
diagnosis of the previous run, measured, and kept only if it helps. Every
experiment has its own folder with the hypothesis, the exact config, the raw
metrics and a conclusion.

## Rules

1. **Nothing is deleted, everything is switchable.** A new idea is added as a new
   module or a new config key. The old behaviour stays available.
2. **Code defaults = previous behaviour.** A new key's default in the code
   reproduces what happened before the key existed; the *experiment* config turns
   it on. That way every older `track_config.yaml` snapshot still reproduces its
   original numbers. `configs/default.yaml` is only updated once a change wins.
3. **One folder per experiment:** `NNN_short_name/` with `README.md`
   (question → change → results → conclusion), the config snapshot,
   `summary.json`, `timing.csv`.
4. **One commit + one tag per milestone** (`v0.1-baseline`, `v0.2-...`), so any
   result can be checked out and re-run.
5. Large artefacts (videos, cached detections, MOT result files) stay in
   `outputs/`, which is not versioned.

## Summary

Clip `v_00HRwkvvjtQ_c001` (SportsMOT val, basketball, 10 players) unless stated otherwise.

| # | Change | IDF1 | MOTA | IDSW | IDs (GT 10) | Re-entries | ReID ms/frame | Verdict |
|---|---|---|---|---|---|---|---|---|
| [001](001_baseline_yolo26m_coco/) | Baseline: YOLO26m COCO + ReIDTracker, no tuning | 37.8 | −27.7 | 49 | 123 | 0 / 4 | 2014 | Detector false positives dominate |
| [002](002_reid_only_usable_boxes/) | ReID only on boxes the tracker can use | 37.8 | −27.7 | 49 | 123 | 0 / 4 | **544** | ✅ Identical tracks, ReID 3.7× faster |
| [003](003_court_mask_filter/) | Court mask removes people off the playing surface | _pending_ | | | | | | Validated: −71 % false positives, 0 players lost |
