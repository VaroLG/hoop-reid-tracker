# 026 · Porting Deep-EIoU's pieces into our tracker: negative

| | |
|---|---|
| **Date** | 2026-10-31 · **negative**, options kept off by default |
| **Builds on** | 025 (in Deep-EIoU: min-distance fusion −10.3 HOTA if removed, expansion −3.7, last box vs Kalman −1.6/−2.5) |
| **Question** | Do the same pieces help **our** tracker? And how much of the 024 gap was the ReID model? |

## What was added (all behind options, defaults = tracker up to 025)

| option | values | code |
|---|---|---|
| `tracker.fusion` | `weighted` (default) · `min` | `matching.fuse_min`: cost = min(1 − IoU, appearance); appearance only counts when cosine distance < `min_app_gate` (0.5) and 1 − IoU < `min_proximity` (0.5), Deep-EIoU's values in our scale |
| `tracker.assoc_box` | `kalman` (default) · `last` · `last_gmc` | `Track.last_tlbr`: stages 1–4 compare the last DETECTED box (optionally warped by the camera motion) instead of the Kalman prediction. The output box is still the filtered one |
| `reid.embedding_cache` | folder of `.npz` | `reid/cache.py`: reads the OSNet embeddings saved by 024 instead of running the network (unknown boxes are computed). Speed only: ReID 0.2 ms/frame instead of the network |

Tests: `tests/test_deep_eiou_port.py`. Expansion (014 keys): `eiou_expand=0.7 eiou_iterations=2 eiou_expand_low=0.5`.
All runs: 15 val clips, cached detections (011), OSNet 012, `run_court_source_eval.py --variants segmentation`.
No value tuned: the new options use Deep-EIoU's numbers.

## Results (HOTA, 15 val clips)

| run | ReID | expansion | fusion | box | HOTA | AssA | IDF1 | ID sw. |
|---|---|---|---|---|---|---|---|---|
| 011b | ResNet-50 ImageNet | — | weighted | Kalman | 68.8 | 55.1 | 72.7 | 278 |
| **osnet** | OSNet | — | weighted | Kalman | **69.2** | 55.8 | 73.6 | 268 |
| eiou | OSNet | ✓ | weighted | Kalman | 63.0 | 46.7 | 66.2 | 454 |
| min | OSNet | ✓ | min | Kalman | 64.4 | 48.8 | 67.5 | 463 |
| last | OSNet | ✓ | weighted | last | 63.3 | 47.2 | 67.6 | 427 |
| min_last | OSNet | ✓ | min | last | 64.0 | 48.5 | 68.4 | 442 |
| min_last_gmc | OSNet | ✓ | min | last + GMC | 63.2 | 47.1 | 66.9 | 465 |
| Deep-EIoU (024) | OSNet | ✓ | min | last | 74.3 | 63.5 | 79.6 | 859 |

| comparison (per clip) | better | worse | median Δ |
|---|---|---|---|
| osnet vs 011b | 9 | 3 | +0.4 |
| eiou vs osnet | 1 | **14** | −6.6 |
| min vs eiou | 8 | 5 | +0.4 |
| last vs eiou | 6 | 9 | −0.6 |
| min_last vs osnet | 3 | 12 | −5.4 |

Per clip: [`hota.csv`](hota.csv).

## Reading

1. **The ReID model was not the gap.** Our online tracker with OSNet: 69.2 vs 68.8 with ResNet-50 (+0.4). The
   5-point gap to Deep-EIoU (74.3) is the tracker itself (this also corrects 024/025, see their notes).
2. **In our tracker the expansion is harmful** (−6.2, 14 of 15 clips, ID switches 268 → 454). In Deep-EIoU the same
   expansion is worth +3.7. 014 had seen the same on 4 clips.
3. **The min rule and the last box do not rescue it** (+0.4 and −0.6 on top of the expansion); together they stay
   5 points below our tracker without expansion.
4. So the pieces do not transfer one by one: in Deep-EIoU they work **together with the rest of its design**.
   Differences that remain between the two trackers, not tested here: our appearance is NaN for occluded crops (a
   box covered > 20 % by another) and falls back to motion, exactly where expanded boxes of neighbours overlap; our
   appearance veto is disabled above IoU 0.5 (005a), and expanded IoUs of neighbours often exceed it; thresholds
   (high 0.5 vs 0.6, new track 0.6 vs 0.7); lost tracks stay in every stage-1 pass; long-term memory and delayed
   births.

## Conclusion

**Negative, not adopted.** The options stay in the code, off by default (identical output to 025), so the experiment
can be re-run; they can be removed without touching anything else. The best result of the project is still
Deep-EIoU online + our offline step (024, 77.4 HOTA), which cannot be shipped (no licence).
