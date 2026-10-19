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

## Part 2 · on court-masked detections (pending)

```bash
python scripts/run_ablation.py --set "data.only=[v_00HRwkvvjtQ_c001]" detector.cache_dir=outputs/detections/yolo26m filters.court_mask=true --out outputs/ablation/004a_court --hota
```

| Configuration | IDF1 | IDSW ↓ | HOTA | AssA |
|---|---|---|---|---|
| 1 · motion only | _pending_ | | | |
| 2 · + GMC | | | | |
| 3 · + ReID | | | | |
| 4 · + memory (full) | | | | |
| 5 · full, ground-truth boxes | | | | |

## Conclusion (provisional)

**GMC stays. ReID and memory, as configured, do not earn their place yet.**
If Part 2 confirms it, the next experiment is not a new module but fixing the
existing one: calibrate the appearance thresholds on the *train* split
(`scripts/calibrate_reid.py`) and re-run this ablation — then add the team gate.
