# P05 · Possession with a learned per-frame score and a two-state temporal model

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [P04](../P04_keep_state/) (adopted: 86.1 % of frames right, event F1 0.50, but 213 events predicted for 100) |
| **Question** | P04 combines its cues with hand-set thresholds and flickers. Does a logistic score trained on the labels, decoded by a two-state hidden Markov model with the change rate counted in the labels, do better? |

## Method (fixed before running)

[`hooptrack/ball/possession_temporal.py`](../../../src/hooptrack/ball/possession_temporal.py),
[`scripts/possession/p07_temporal.py`](../../../scripts/possession/p07_temporal.py):

* 8 features per frame: ball present, interpolated, distance to the closest player's hands, inside a reach box,
  players within reach, speed relative to that player, own speed, height on his box (P01's geometry);
* logistic regression (scikit-learn defaults, standardised) on the 3 held-out **train** clips (1,877 frames, ball =
  B08 on those clips, players = ground-truth boxes), P(held);
* Viterbi over held/air; emission = posterior / prior; change probability = labelled changes / frames on train
  (0.0197 per frame). Nothing tuned.
* Test once on the 4 test clips (ball B08, players 038), P03's scoring. The logistic alone (p > 0.5) is reported as
  the ablation of the temporal model.

## Results (4 test clips, 100 labelled events)

| | frames right | events predicted | found | recall | precision | event F1 |
|---|---|---|---|---|---|---|
| *always held* | *78.6 %* | | | | | |
| **P04 (adopted)** | 86.1 % | 213 | 78 | 0.78 | 0.37 | **0.50** |
| logistic per frame (ablation) | **90.6 %** | 182 | 71 | 0.71 | 0.39 | **0.50** |
| **P05** logistic + temporal model | 87.9 % | 58 | 34 | 0.34 | **0.59** | 0.43 |

Frames right per clip, P04 → logistic → P05 ([`per_clip.csv`](per_clip.csv)): NCAA c001 92.8 → 92.8 → 90.4 · FIBA
Changsha 85.7 → 87.1 → 84.8 · FIBA c003 78.5 → 89.9 → 86.8 · FIBA 2010 c012 89.2 → 92.7 → 89.8.

Train, leave-one-clip-out (3 folds, to compare the two decoders without the test clips): frames 86.4 % (logistic) vs
86.1 % (P05); event F1 0.26 vs **0.38**.

## Reading

1. **The temporal model over-smooths.** It predicts 58 events for 100 and misses the short "air" stretches (quick
   passes, hand-offs): frame-level emissions are strongly correlated, and a single switch rate for every kind of
   change cannot tell a 10-frame pass from noise.
2. **The learned score is the useful part**: alone it gets 90.6 % of the frames right, better than P04 on 3 of 4
   clips and equal on the fourth, with the same event F1. But it was not the planned method and on train (leave one
   clip out) it is only equal to P05 on frames and worse on events: adopting it would be a choice made after seeing
   the test clips. **Not adopted; candidate for the next experiment** (e.g. labels on more train clips, then decide
   on train).
3. The weights say what matters: no ball → air; close to the hands, inside the reach box and high on the body → held;
   fast ball → air. Same cues as P01, now weighted by data.

## Conclusion

**Mixed → not adopted.** P04 stays the possession output. Kept: the feature extractor and the decoder (off by
default, own script).

## Command

```powershell
python scripts/possession/p07_temporal.py
```
