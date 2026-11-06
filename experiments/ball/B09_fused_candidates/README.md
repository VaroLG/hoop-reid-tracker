# B09 · WASB's and YOLO's candidates in one tracker

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [B08](../B08_wasb_finetuned/) (adopted: fine-tuned WASB + B05 tracker, F1 0.78) · [B05](../B05_ball_tracker/) |
| **Question** | B08 finds 87 % of the balls in the air or on the floor but 70 % of those over a player and 38 % over two or more. On 41 % of its misses B03's YOLO had a candidate on the right spot. Do the two detectors together find more? |

## Change (one value)

[`scripts/ball/b09_fuse_candidates.py`](../../../scripts/ball/b09_fuse_candidates.py): YOLO's candidates (B03) are added
to WASB's in the same B05 tracker with their confidence × `alpha` (0 = B08). `alpha` chosen on the 3 held-out
**train** clips (best F1, criterion fixed before), tested once on the 4 test clips.

## Results

Train grid ([`train_grid.csv`](train_grid.csv)): F1 0.893 (alpha 0) · **0.894 (0.1)** · 0.885 (0.25) · 0.877 (1.0).

| test, 334 visible balls | found | elsewhere | missed | false alarm | recall | precision | F1 |
|---|---|---|---|---|---|---|---|
| B08 (alpha 0) | 237 | 24 | 73 | 17 | 71.0 % | 85.3 % | 0.775 |
| B09 (alpha 0.1) | 240 | 26 | 68 | 17 | 71.9 % | 84.8 % | 0.778 |

Per clip ([`per_clip.csv`](per_clip.csv)): equal on 2, +2 and +1 balls on the other two.

## Reading

1. **Null result.** The YOLO candidates that sit on the missed balls are weak (median confidence 0.42, then × alpha)
   and lose against the tracker's gap and restart costs; letting them weigh more (alpha ≥ 0.25) brings their still
   look-alikes back (more false alarms on train).
2. **The held-out train clips are easy for WASB** (93 % found vs 71 % on the test clips: same 3 games as its training
   clicks), so they say little about the hard cases. Same caveat as 038's README for the player models.
3. The missing balls are a modelling problem (a ball on a body), not a candidates problem: next is to treat "held"
   as a state of the tracker (follow the holder's hands) instead of more detectors.

## Conclusion

**Negative (null), not adopted.** B08 stays.
