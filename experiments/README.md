# Experiment log

Two chapters, kept separate:

* **Chapter 1 · Tracking** — this page, experiments `001`, `002`, …
* **Chapter 2 · Court geometry** — [`court/`](court/), steps `C01`, `C02`, … (court keypoints, homography, top-down map)

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
| [006](006_team_classifier/) | Jersey-colour team/referee classifier: hide referees (d); team gate (a–c) | **66.9** | **90.1** | 34 | 42 | 1 / 4 | 314 | ✅ Hiding referees: −87 % FP, HOTA 58.3 → 63.7. ❌ Team gate fixes 0 cross-team swaps (they happen inside occlusions) |
| [007](007_referee_leftovers/) | Referee leftovers: early colour model (a), probation for new tracks (b), offline filter of whole referee tracks (c) | **67.7** ‡ | **92.2** ‡ | **33** ‡ | 42 | 1 / 4 | 323 | ✅ c: −64 % FP (390 → 142). ❌ a: no effect. ➖ b: trades FP for misses 1 : 1. ‡ = offline mode; online default unchanged |
| [008](008_court_border/) | Court border: line detection (prototype), stricter margin calibrated on train, margin learned per video | 63.9 § | 89.5 § | 35 § | — | — | 0 § | ❌ No gain: border people are referees, already hidden by 006. Train calibration shows the 003 mask drops up to 10 % of players in another arena. § = motion-only, VM |
