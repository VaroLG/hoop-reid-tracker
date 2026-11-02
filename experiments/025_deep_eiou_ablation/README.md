# 025 · What makes Deep-EIoU's online tracker better than ours: ablation

| | |
|---|---|
| **Date** | 2026-10-30 · **diagnostic** |
| **Builds on** | 024 (Deep-EIoU online 74.3 HOTA vs our online 011b 68.8, same detections; 011b uses the ImageNet ResNet-50, Deep-EIoU uses OSNet: see the correction in 024) |
| **Question** | Which piece of Deep-EIoU makes the +5.5 HOTA? |

## Method

`scripts/run_deep_eiou.py --ablate <name>`: each run removes ONE piece of Deep-EIoU and keeps their default values for
everything else. Their files are not changed: our script replaces a function of their modules in memory, or changes
one argument. Since no value is tuned, measuring on val is a diagnosis, not a selection. 15 val clips, same
detections and OSNet embeddings as 024, no interpolation.

Two things found while reading their code:

- **Deep-EIoU has no motion model.** Its `STrack.update()` never corrects the Kalman filter (only `re_activate`
  does), and every stage matches the **last detected box** (`last_tlbr`), not a prediction. A lost track stays where
  it was last seen.
- **Their expansion is larger than in the paper and in our 014.** `matching.expand()` grows each side by
  `(2e+1)/2` × the size, so `e = 0.7` gives a box **3.4×** as wide and tall (014: 2.4×), and `e = 0` doubles it.

## Results (HOTA, 15 val clips)

| run | what is removed | HOTA | AssA | IDF1 | Δ HOTA | clips worse / better |
|---|---|---|---|---|---|---|
| **full** | nothing (024) | **74.3** | 63.5 | 79.6 | — | — |
| `no_reid` | appearance: expanded IoU only | 64.0 | 48.0 | 66.8 | **−10.3** | 14 / 1 |
| `no_expand` | expansion: plain IoU in every stage | 70.6 | 57.0 | 72.5 | **−3.7** | 11 / 4 |
| `kalman` | last box → Kalman-predicted box (filter corrected every frame) | 71.8 | 59.2 | 74.9 | **−2.5** | 13 / 2 |
| `kalman_gmc` | same + our camera-motion compensation (as in our tracker) | 72.7 | 60.7 | 76.3 | −1.6 | 12 / 3 |
| `no_low` | second stage with low-score boxes | 72.7 | 61.0 | 77.9 | −1.6 | 10 / 5 |
| `expand_014` | their expansion (3.4×) → ours (2.4×) | 74.5 | 63.7 | 79.3 | +0.2 | 5 / 10 |
| `confirmed_only` | tracks not output on their birth frame | 74.3 | 63.5 | 79.6 | 0.0 | 0 / 4 |
| our online, 011b (ResNet-50 ImageNet, not OSNet) | — | 68.8 | 55.1 | 72.7 | −5.5 | 10 / 5 |

(A first `kalman` run gave 23 HOTA: the prediction was never corrected, because their `update()` does not touch the
filter. The patch now corrects it with every matched box, as a real Kalman tracker does.)

Per clip: [`hota.csv`](hota.csv).

## Reading

1. **How appearance is used is the main piece (−10.3).** Deep-EIoU does not add IoU and appearance with weights, as
   ours does (`cost = 0.5·app + 0.5·IoU`). It takes the **smaller** of the two distances, and the appearance distance
   only counts when the expanded boxes overlap (IoU distance < 0.5) and the cosine distance is small (< 0.5). A pair
   that looks the same and is nearby is matched even if the boxes barely overlap.
2. **Expansion matters (−3.7), its exact size does not** (2.4× ≈ 3.4×). So 014 did not fail for using a smaller box.
3. **The last detected box beats the Kalman prediction** (−2.5, −1.6 with camera compensation, worse on 12–13
   clips). In basketball players stop, turn and change pace all the time: a constant-velocity prediction is often
   wrong, and with a very large expanded box no prediction is needed. Our tracker associates on the Kalman + GMC box.
4. The low-score second stage helps a little (−1.6); ours already has one.
5. Even with the Kalman box and camera compensation, like ours, Deep-EIoU stays at 72.7 vs 68.8. Part of that gap is
   the association rule (point 1), part the ReID model: 011b uses the ImageNet ResNet-50, these runs use OSNet. 026
   measures our tracker with OSNet first, to separate the two.

## Conclusion

To port into our tracker (026), in this order: (a) the min-distance fusion with its two gates, (b) association on the
last detected box, expanded, instead of the Kalman prediction. Our own pieces stay (long-term memory, team gate,
offline stitching). 014 tested the expansion alone, on the predicted box and with the weighted sum: that is why it
was neutral.
