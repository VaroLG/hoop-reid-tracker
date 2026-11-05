# P03 · The same possession rules with a better ball

| | |
|---|---|
| **Date** | 2026-11-05 |
| **Builds on** | [P01](../P01_rules/) (rules) · [P02](../P02_ball_motion/) (perfect-ball ceiling) · [B08](../../ball/B08_wasb_finetuned/) (WASB fine-tuned + tracker: 85 % of the answers right, 73 % for B05c) |
| **Question** | P02 suggested the rules, not the ball, were the limit. With B08's ball and exactly P01's rules, how much does possession change? |

## Method

P01 unchanged ([`scripts/possession/p01_possession.py`](../../../scripts/possession/p01_possession.py), same values), on
the B08 ball tracks (`outputs/ball/tracks_b08`) and the 038 player tracks; scored by
[`scripts/possession/p03_eval_possession.py`](../../../scripts/possession/p03_eval_possession.py) against the same 100
hand-marked catches and releases ([`per_clip.csv`](per_clip.csv)).

## Results (4 test clips)

| P01's rules with… | events found | events right | **event F1** | frames right, where the system has a ball |
|---|---|---|---|---|
| B05c ball (P01) | **80 %** | 27 % | 0.41 | 83 % |
| hand clicks, 1 in 10 interpolated (P02) | 73 % | 33 % | 0.46 | – |
| **B08 ball** | 78 % | **37 %** | **0.50** | **88 %** |

Changes invented: 292 → 213. Per clip, events right: NCAA 30 → 46 %, Changsha 41 → 53 %, FIBA women's WC 18 → 24 %,
FIBA 2010 26 → 32 % (better on all four). Over all frames the held / air agreement is 73.7 % (74.8 % before): B08
answers on fewer frames (no ball = counted as air); where it has a ball, 86–91 % of the frames are right, above the
trivial "always held" (79 %).

## Reading

1. **The ball was a limit after all**: with the same rules, a surer ball removes a quarter of the invented changes and
   lifts the event F1 from 0.41 to 0.50, past the "perfect ball" of P02 (whose clicks, one frame in ten, smooth the
   ball's path between them: a coarser ceiling than it looked).
2. Where the system sees the ball, it knows held / air in ~88 % of the frames; what is left are the frames with no
   ball (counted as air) and still ~2 invented changes per real one, most on the FIBA women's clip.

## Conclusion

**Positive: P01's rules on the B08 ball are the possession reference** (event F1 0.50). Next steps that follow from
this: keep the last state through the frames with no ball, and a temporal model (as in Wei et al., ICCV-W 2015) instead
of hand rules.
