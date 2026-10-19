# 005 · Make ReID earn its place

| | |
|---|---|
| **Date** | 2026-10-19 |
| **Builds on** | [004a](../004a_component_ablation/) — ReID added 14 ID switches, all "lost track" |
| **Bar to beat** | motion + GMC, no ReID: **IDF1 58.2 · HOTA 56.4 · 36 IDSW** (current default) |

## Hypotheses

- **005a · Veto, not appearance, breaks tracks.** In stage 1 the appearance gate
  is a hard veto at any IoU. When a player's look changes for a few frames,
  a correct motion match is forbidden, the track is cut and a new id is born
  (lost-track switches 16 → 29 in 004a). Limiting the veto to pairs where
  motion is ambiguous (IoU < 0.5) should remove most of those switches while
  keeping appearance as a tie-breaker.
- **005b · The thresholds were never calibrated.** `appearance_gate: 0.35` and
  `memory_thresh: 0.25` were copied, not measured for ResNet-50 features.
  Measuring same-player vs closest-other-player distances on the **train**
  split gives thresholds that fit this extractor without touching validation data.

## Change

| Switch | Default (= 004a) | 005a |
|---|---|---|
| `tracker.appearance_veto_max_iou` | `.inf` (veto at any IoU) | `0.5` |

Code: `fuse_motion_appearance()` in `src/hooptrack/tracking/matching.py` ·
test: `test_veto_limited_to_ambiguous_motion`.

## Commands

```bash
# 005a — soft appearance (court mask on by default; ReID turned on explicitly)
hooptrack track --set detector.type=cached detector.cache_dir=outputs/detections/yolo26m "data.only=[v_00HRwkvvjtQ_c001]" reid.type=resnet tracker.memory_enabled=false tracker.appearance_veto_max_iou=0.5 output.dir=outputs/runs/005a_soft_reid --eval
hooptrack track --set detector.type=cached detector.cache_dir=outputs/detections/yolo26m "data.only=[v_00HRwkvvjtQ_c001]" reid.type=resnet tracker.memory_enabled=true  tracker.appearance_veto_max_iou=0.5 output.dir=outputs/runs/005a_soft_reid_memory --eval

# HOTA + switch breakdown
hooptrack evaluate --run outputs/runs/005a_soft_reid --hota --set "data.only=[v_00HRwkvvjtQ_c001]"
hooptrack evaluate --run outputs/runs/005a_soft_reid_memory --hota --set "data.only=[v_00HRwkvvjtQ_c001]"
python scripts/idsw_breakdown.py outputs/ablation/004a_court/2_plus_gmc outputs/ablation/004a_court/3_plus_reid outputs/runs/005a_soft_reid outputs/runs/005a_soft_reid_memory --set "data.only=[v_00HRwkvvjtQ_c001]"

# 005b — calibrate on TRAIN (needs: python scripts/download_data.py --splits train)
python scripts/calibrate_reid.py --set reid.type=resnet
```

## Results

| Configuration | IDF1 | HOTA | AssA | IDSW | Lost-track switches | Re-entries | FPS |
|---|---|---|---|---|---|---|---|
| 004a · motion + GMC (bar) | 58.2 | 56.4 | 48.0 | 36 | 16 | 0 / 4 | 52 |
| 004a · + ReID (veto) | 58.3 | 57.2 | 49.7 | 50 | 29 | 1 / 4 | 3.0 |
| 005a · + ReID, soft | _pending_ | | | | | | |
| 005a · + ReID, soft, + memory | _pending_ | | | | | | |
| 005b · calibrated thresholds | _pending_ | | | | | | |

## Conclusion

_Pending._
