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
| [003](003_court_mask_filter/) | Court mask removes people off the playing surface | **56.6** | **68.0** | 48 | **58** | **1 / 4** | **379** | ✅ −79 % false positives, 0 players lost, no training |
| [004a](004a_component_ablation/) | Ablation on court-masked boxes: motion → +GMC → +ReID → +memory | 58.2 * | 69.1 * | 36 * | 50 * | 0 / 4 | — | GMC wins; ReID adds 14 switches (lost tracks), fixes no cross-team swap. \* = best config: motion + GMC, 52 FPS |
| [005](005_make_reid_earn_its_place/) | ReID as soft cost (veto only when IoU < 0.5) + memory; thresholds measured on train | **60.5** | 68.0 | **34** | **42** | 1 / 4 | 314 | ✅ First time ReID beats motion-only (+2.3 IDF1, +1.9 HOTA). Generic ResNet cannot separate players (calibration) |
| [006](006_team_classifier/) | Jersey-colour team/referee classifier: team gate (a–c), hide referees (d) | **64.0** † | **89.6** † | 36 † | — | — | 0 † | ⏳ Gate: 0 cross-team swaps fixed (they happen inside occlusions). Hiding referees: −83 % FP. † = motion-only, preliminary; official ResNet run pending |
