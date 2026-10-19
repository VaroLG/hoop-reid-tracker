# 004a · Component ablation: what does each part of the tracker add?

| | |
|---|---|
| **Date** | 2026-10-18 / 19 |
| **Builds on** | [003](../003_court_mask_filter/) — "ID switches do not move" |
| **Switches** | none new — existing keys `tracker.gmc`, `reid.type`, `tracker.memory_enabled` |
| **Script** | [`scripts/run_ablation.py`](../../scripts/run_ablation.py) |

## Question

After 003 the remaining problem is identity: 48 ID switches, and in 003 the
full tracker was only 5 IDF1 points better than the motion-only one. Before
adding anything new (team classifier, calibrated ReID), we need to know what
the **existing** parts contribute. The ablation adds one component at a time:

1. motion only — Kalman + IoU, two-stage association (≈ ByteTrack)
2. + camera motion compensation (GMC)
3. + appearance ReID (ResNet-50)
4. + long-term identity memory (= full tracker)
5. full tracker on ground-truth boxes (ceiling of the association logic)

## Part 1 · on the raw detector output (court mask OFF)

> This run was launched while `configs/default.yaml` still had
> `filters.court_mask: false`, so it measures the components on the **001 input**
> (all people, including the crowd). It is kept because it is informative on
> its own; Part 2 repeats it on the filtered boxes.

| Configuration | IDF1 | IDSW ↓ | HOTA | AssA | Frag |
|---|---|---|---|---|---|
| 1 · motion only | 36.1 | 44 | 38.0 | 37.5 | 83 |
| 2 · + GMC | **40.6** | **36** | **42.8** | **48.0** | 83 |
| 3 · + ReID | 39.0 | 51 | 41.2 | 44.7 | 111 |
| 4 · + memory (full, = 001) | 37.8 | 49 | 40.5 | 43.4 | 111 |
| 5 · full, ground-truth boxes | 76.7 | 25 | 79.9 | 66.4 | 36 |

Raw table: [`ablation_part1.csv`](ablation_part1.csv).

### Reading

- **Camera motion compensation clearly helps:** AssA +10.5, 8 fewer ID switches.
  Broadcast cameras pan constantly; warping the Kalman prediction keeps IoU
  matching alive during fast breaks.
- **Appearance, as currently configured, hurts:** adding ResNet ReID costs
  1.6 IDF1 and adds 15 ID switches and 28 fragmentations. The memory on top
  does not recover anything (0 / 4 re-entries) and loses another point.
- **The strongest evidence is row 5.** On perfect boxes the full tracker makes
  25 ID switches, while the plain motion-only tracker on the same perfect boxes
  makes **9** (see [003 diagnosis](../003_court_mask_filter/detection_analysis.md), row C).
  With clean input, the appearance stages *create* switches.
- Likely causes, to be tested: generic ImageNet features put team-mates very
  close; the gates (`appearance_gate 0.35`, `memory_thresh 0.25`) were copied,
  not calibrated; and appearance is trusted in stage 4 even when motion would
  disagree.

## Part 2 · on court-masked detections (the current input)

```bash
python scripts/run_ablation.py --set "data.only=[v_00HRwkvvjtQ_c001]" detector.cache_dir=outputs/detections/yolo26m filters.court_mask=true --out outputs/ablation/004a_court --hota
```

| Configuration | IDF1 | HOTA | AssA | IDSW ↓ | Frag ↓ | IDs (GT 10) | Re-entries | FPS |
|---|---|---|---|---|---|---|---|---|
| 1 · motion only | 51.5 | 49.9 | 37.6 | 44 | 83 | 54 | 0 / 4 | 103 |
| 2 · + GMC | **58.2** | 56.4 | 48.0 | **36** | **83** | **50** | 0 / 4 | **52** |
| 3 · + ReID | **58.3** | **57.2** | **49.7** | 50 | 111 | 64 | 1 / 4 | 3.0 |
| 4 · + memory (full, = 003) | 56.6 | 55.6 | 47.1 | 48 | 109 | 58 | 1 / 4 | 2.8 |
| 5 · full, ground-truth boxes | 76.7 | 79.9 | 66.4 | 25 | 36 | 30 | 0 / 5 | 2.9 |

Raw table: [`ablation_part2.csv`](ablation_part2.csv). FPS excludes detection (cached).

### What kind of ID switches each component causes

Each switch is classified with the two teams recovered by clustering jersey
colours (5 + 5 players):

| Configuration | Cross-team | Same team | Lost → new / old id | Total |
|---|---|---|---|---|
| 1 · motion only | 14 | 11 | 19 | 44 |
| 2 · + GMC | 14 | 6 | 16 | 36 |
| 3 · + ReID | 14 | 7 | **29** | 50 |
| 4 · + memory | 14 | 6 | 28 | 48 |

### Reading

- **GMC is the clear winner:** +6.7 IDF1, +10.4 AssA, −8 ID switches, and the
  tracker still runs at 52 FPS (detection excluded).
- **ReID buys +0.1 IDF1 and +1.7 AssA for a 17× slow-down — and adds 14 switches.**
  All of the extra switches are of one kind: *tracks lost and restarted*
  (16 → 29), matching the jump in fragmentations (83 → 111). The appearance
  gate (`appearance_gate: 0.35`) is rejecting correct matches when a player's
  look changes (blur, pose, partial occlusion), which breaks tracks that motion
  alone would have kept.
- **ReID does not fix cross-team swaps at all (14 in every row).** Generic
  ImageNet features are not separating two jersey colours reliably enough to
  matter — the team gate idea attacks exactly these 14.
- **Memory** recovers 1 / 4 re-entries but costs 1.7 IDF1 overall.
- Row 5 confirms it on perfect boxes: 25 switches for the full tracker vs 9 for
  plain motion matching ([003 diagnosis](../003_court_mask_filter/detection_analysis.md), row C).

## Conclusion

**The best measured tracker today is the simplest one: motion + camera
compensation** — IDF1 58.2, HOTA 56.4, 36 ID switches, 52 FPS, better than the
full tracker (56.6, 55.6, 48 switches, 2.8 FPS).

ReID is not removed — it is the component that should solve re-entries and
team-mates — but as configured it does not earn its cost. Next, in order:

1. **Fix how appearance is used** (005): calibrate `appearance_gate` / `memory_thresh`
   on the *train* split and use appearance as a soft cost instead of a hard veto
   in stage 1. Target: remove the +13 "lost → new id" switches.
2. **Team gate** (006): forbid cross-team matches using jersey-colour clusters.
   Target: the 14 cross-team switches that no component touches today.
3. ReID must then beat row 2 (58.2 IDF1 / 36 IDSW) to stay on by default.
