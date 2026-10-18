# 002 · ReID only on the boxes the tracker can use

| | |
|---|---|
| **Date** | 2026-10-18 |
| **Builds on** | [001](../001_baseline_yolo26m_coco/) — limitation 2 |
| **Switch** | `reid.only_usable_boxes: true` (default `false` = behaviour of 001) |
| **Code** | `embed_needed()` in `src/hooptrack/pipeline.py` · tests in `tests/test_improvements.py` |

## Hypothesis

In 001 the ReID network ran on ~58 boxes per frame and took 2 s/frame, 57× more
than the tracker itself. But the tracker only uses an embedding when the box
is confident:

| Where appearance is read | Boxes that reach it |
|---|---|
| Stage 1 (fused cost), stage 4 (re-identification), births | score ≥ `high_thresh` (0.5) |
| Updating a track's appearance model | score ≥ `feature_update_min_score` (0.6) |
| Stage 2 (low-score recovery) | IoU only — appearance never read |

Embedding only boxes with `score ≥ min(high_thresh, feature_update_min_score)`
should give **exactly the same tracks** and **~3× faster ReID** (17.6 instead
of 57.8 crops per frame).

## Change

`embed_needed()` crops and embeds only those boxes; the other rows of the
feature matrix are zeros that the tracker never reads. The cut-off is the
*lower* of the two thresholds, so a config that learns appearance below
`high_thresh` stays correct (tested).

## Results

### Validation (histogram ReID, full clip)

Run with the colour-histogram extractor, which needs no PyTorch:

| | IDF1 | MOTA | IDSW | ReID ms/frame |
|---|---|---|---|---|
| `only_usable_boxes: false` (001 behaviour) | 22.8 | −24.0 | 186 | 6.1 |
| `only_usable_boxes: true` | 22.8 | −24.0 | 186 | **2.5** |

The two MOT result files are **byte-identical**. Tracking is unchanged; ReID is 2.4× faster.

### Official run (ResNet-50, same setup as 001)

```bash
hooptrack track --set detector.type=cached detector.cache_dir=outputs/detections/yolo26m "data.only=[v_00HRwkvvjtQ_c001]" reid.only_usable_boxes=true output.dir=outputs/runs/002_reid_only_usable --eval
```

| | IDF1 | MOTA | IDSW | ReID ms/frame | FPS |
|---|---|---|---|---|---|
| 001 | 37.8 | −27.7 | 49 | 2014 | 0.49 |
| **002** | **37.8** | **−27.7** | **49** | **544** | **1.75** |

Every metric is identical (also Frag 111, 123 ids, 14 revivals). Wall-clock
time for the clip dropped from ~41 min to ~12 min. Raw files:
[`track_config.yaml`](track_config.yaml) · [`summary.json`](summary.json) ·
[`timing.csv`](timing.csv).

The speed-up (3.7×) matches the number of crops: 57.8 → 17.6 boxes per frame
is 3.3× fewer, and a ResNet-50 crop costs ~30–35 ms on this CPU either way.

## Conclusion

**Confirmed.** Same tracks, ReID 3.7× faster, end-to-end 3.6× faster.
`only_usable_boxes` becomes `true` in `configs/default.yaml` (the code default
stays `false`, so the 001 snapshot still reproduces the baseline).
