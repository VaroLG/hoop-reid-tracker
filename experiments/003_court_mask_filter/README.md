# 003 · Court mask: drop people whose feet are not on the court

| | |
|---|---|
| **Date** | 2026-10-18 |
| **Builds on** | [001](../001_baseline_yolo26m_coco/) — limitation 1, hypothesis H1 · uses [002](../002_reid_only_usable_boxes/) |
| **Switch** | `filters.court_mask: true` (default `false` = no filtering, as in 001) |
| **Code** | `src/hooptrack/detection/filters.py` (`CourtMask`) · tests in `tests/test_improvements.py` |

## Hypothesis

In 001, half of the confident detections were not players, and most of them
were spectators, bench players and photographers. They have one thing in
common: **their feet are not on the playing surface.** A geometric filter
should remove most of them **without training anything**, on CPU, while
keeping every player.

## Change

A post-detection filter, applied before ReID and tracking:

1. Estimate the floor colour (median HSV of a central band, people excluded).
2. Threshold, clean, keep the largest region, take its **convex hull**.
3. Keep a box if its foot point is inside the hull, with a tolerance of 15 %
   of the box height (players step on the lines).
4. Safety nets: if too little floor is visible, reuse the last good hull for a
   few frames, then let everything through. The filter only ever removes boxes.

![Court mask examples](figures/court_mask_examples.jpg)

*Green polygon: estimated court. Green boxes: players (kept). Grey crossed
boxes: removed by the filter. Red boxes: false positives the filter keeps —
referees and bench players standing on the baseline.*

## Results

### Detector level (exact, full clip — [`detection_analysis.md`](detection_analysis.md))

| score ≥ 0.5 | Detections / frame | Precision | Players kept | False positives removed |
|---|---|---|---|---|
| without filter | 17.6 | 50.3 % | — | — |
| **with court mask** | **11.4** | **77.8 %** | **100 %** | **71.0 %** |

Recall is unchanged (97.4 %). The mask was trusted on all 1162 frames.
Cost: 4.6 ms/frame.

### Tracker level — motion-only (same analysis script)

| Input boxes | IDF1 | MOTA | IDSW | IDs (GT 10) |
|---|---|---|---|---|
| A · detector as is | 36.1 | −25.0 | 44 | 132 |
| **A+F · detector + court mask** | **51.5** | **69.1** | 44 | **54** |
| B · only boxes that match a player (ceiling for any filter) | 64.4 | 94.3 | 42 | 24 |

The filter closes **55 % of the IDF1 gap** and **79 % of the MOTA gap**
between the raw detector and a perfect player-only detector.

### Full tracker — validation with histogram ReID

| | IDF1 | MOTA | IDSW | IDs |
|---|---|---|---|---|
| 001 setup | 22.8 | −24.0 | 186 | 180 |
| + court mask | 34.1 | 66.2 | 189 | 115 |

MOTA improves as expected, but **ID switches do not**, and this tracker does
worse than the motion-only one above (IDSW 186 vs 44). The colour histogram puts
team-mates almost as close as the same player, so with uncalibrated gates the
appearance cost *causes* swaps. This is evidence for hypothesis H4
(calibrate ReID), not a problem of the filter.

### Official run (ResNet-50, same setup as 001)

```bash
hooptrack track --set detector.type=cached detector.cache_dir=outputs/detections/yolo26m "data.only=[v_00HRwkvvjtQ_c001]" reid.only_usable_boxes=true filters.court_mask=true output.save_video=true output.dir=outputs/runs/003_court_mask --eval
```

| | IDF1 | MOTA | IDSW | Frag | IDs (GT 10) | False positives | Re-entries | ReID ms/frame | FPS |
|---|---|---|---|---|---|---|---|---|---|
| 001 | 37.8 | −27.7 | 49 | 111 | 123 | 13,725 | 0 / 4 | 2014 | 0.49 |
| 002 | 37.8 | −27.7 | 49 | 111 | 123 | 13,725 | 0 / 4 | 544 | 1.75 |
| **003** | **56.6** | **68.0** | 48 | 109 | **58** | **2,941** | **1 / 4** | **379** | **2.38** |

Raw files: [`track_config.yaml`](track_config.yaml) · [`summary.json`](summary.json) ·
[`timing.csv`](timing.csv) · [`motmetrics.csv`](motmetrics.csv).

- **+18.8 IDF1, +95.7 MOTA**; false positives −79 %, misses unchanged (626 → 624).
- **Identities created halve** (123 → 58) and the memory now recovers a
  returning player for the first time (1 / 4), still with **no stolen identity**.
- The filter costs 17.8 ms/frame on the laptop CPU, but saves more than that in
  ReID (fewer crops): end-to-end FPS goes 1.75 → 2.38.
- **ID switches do not move (49 → 48).** The filter removes *extra* people; it
  does not help to tell two real players apart.
- With the full tracker (ResNet ReID + GMC + memory) IDF1 is 56.6 vs 51.5 for the
  motion-only tracker on the same filtered boxes: once the input is clean,
  appearance starts to pay off — modestly.

<div align="center">
<img src="../../assets/before_after.jpg" width="900" alt="Same frame before and after the court mask: crowd and photographers disappear, ids stay low">
<br><sub>Same frame, same detector. Left: every person gets an id (#63 by frame 451). Right: the court mask keeps players and referees only (#25).</sub>
</div>

## Limitations

- **Referees are not removed**: they stand on the court. They need an
  appearance cue (two team colours + "other") — next experiment.
- **Bench players on the baseline** can fall inside the tolerance.
- **One clip, one arena.** The colour model assumes a single dominant floor
  colour. Courts with large painted areas or very different lighting must be
  checked on the rest of the validation split.

## Conclusion

**Confirmed.** Without any training, the court mask removes 71 % of the false
positives, loses no player, and is the largest single gain so far
(IDF1 37.8 → 56.6, MOTA −27.7 → 68.0). `filters.court_mask` becomes `true`
in `configs/default.yaml`; the code default stays off, so 001 and 002 remain
reproducible from their snapshots.

What is left, in order of evidence:
1. **Referees** — the remaining on-court false positives → team/referee classifier.
2. **ID switches (48)** — unchanged by filtering → isolate what ReID, GMC and
   memory each contribute, then calibrate ReID / add a team gate.
3. **Generalisation** — this is one clip in one arena → run on the full
   validation split before claiming more.
