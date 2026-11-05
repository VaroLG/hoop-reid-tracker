# P02 · "Air" from the ball's own motion, and possession with a perfect ball

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Builds on** | [P01](../P01_rules/) / [P01b](../P01b_tuned_gaps/): event F1 0.41, three times too many changes |
| **Questions** | (1) Is a pass or a shot easier to see in the ball's own motion than in its distance to the boxes? (2) How much of the possession error comes from where the ball is, and how much from the rules? |

## Method

* **P02** (`possession_motion` in [`hooptrack/ball/possession.py`](../../../src/hooptrack/ball/possession.py)): "air"
  when the ball moves faster than `v_air` ball diameters per frame (the camera's motion removed: median motion of the
  player boxes) for at least `min_air` frames; one holder per held stretch (the player most often within reach).
  Values chosen on the 3 held-out train clips by event F1 ([`grid_train.csv`](grid_train.csv)): `v_air = 1.0`,
  `min_air = 6`.
* **Oracle ball**: the hand clicks of B02 (one frame in ten) interpolated in between: not available on a new video,
  it measures the ceiling a perfect ball would give.
* [`scripts/possession/p05_motion.py`](../../../scripts/possession/p05_motion.py); test on the 4 test clips with the 038
  player tracks ([`test.csv`](test.csv)).

## Results (4 test clips, 100 hand-marked changes; "always held" gets 78.6 % of the frames right)

| method | ball | frames right | events found | events right | event F1 |
|---|---|---|---|---|---|
| P01 rules | tracked (B05c) | 74.8 % | **80 %** | 27 % | 0.41 |
| P01 rules | oracle (hand clicks) | 78.4 % | 73 % | **33 %** | **0.46** |
| P02 ball motion | tracked | 70.6 % | 27 % | 29 % | 0.28 |
| P02 ball motion | oracle | 70.6 % | 25 % | 39 % | 0.31 |

On train already, every value of P02 found at most 27 % of the marked changes.

## Reading

1. **The ball's motion alone does not show the changes of possession.** Many passes in these clips are short
   (a hand-off, a bounce pass between two close players) and, in the image, slower than a dribble seen from the side,
   and many releases are shots that start from the hands at low speed: speed misses three changes in four.
2. **A perfect ball helps only a little**: P01 with the hand clicks goes from F1 0.41 to 0.46, and per frame only up to
   the trivial "always held". Most of the error is in the **rules**, not in the ball: deciding by hand-written
   thresholds whether a ball near two overlapping players is held, and by whom.
3. P01 stays the best: it finds 80 % of the catches and releases within one frame, but invents about two for each
   real one.

## Conclusion

**Negative (P02), and a diagnosis: possession by hand-written rules plateaus at F1 ~0.45 even with a perfect ball.**
Going further needs a model that **learns** held / air from examples (the ball's position relative to each player,
its speed, the players' poses) with more labelled clips: the 3 held-out train clips (37 changes) are too few to train
one. Possession stays an open item; P01 is documented as the reference.
