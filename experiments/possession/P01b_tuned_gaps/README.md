# P01b · Filling short "air" gaps, values chosen on train

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Builds on** | [P01](../P01_rules/): 80 % of the marked catches / releases found on time, but 292 changes for 100 |
| **Question** | Does filling short air stretches (inside one player's possession, and between two close players) remove the invented changes? |

## Method

Two values of [`hooptrack/ball/possession.py`](../../../src/hooptrack/ball/possession.py): `max_drop` (air between two
stretches of the same player, filled as his) and `max_drop_switch` (air between two different players, filled as held,
half each; new, 0 = off = P01). Chosen on the 3 held-out train clips (catches / releases marked by hand: 37 changes;
ground-truth player boxes) by the best event F1 over a 4 × 4 grid
([`scripts/possession/p04_tune_possession.py`](../../../scripts/possession/p04_tune_possession.py),
[`grid_train.csv`](grid_train.csv)), then tested once on the 4 test clips with the 038 player tracks ([`test.csv`](test.csv)).

## Results

The grid hardly moves anything on train (event F1 0.27–0.31); chosen: `max_drop = 15`, `max_drop_switch = 0`.

| 4 test clips, 100 marked changes | frames right | events found | events right | event F1 |
|---|---|---|---|---|
| P01 | 74.8 % | 80 % | 27.4 % | 0.408 |
| P01b | 75.8 % | 78 % | 27.5 % | 0.406 |
| "always held" | 78.6 % | – | – | – |

## Reading

The invented changes were not where P01's diagnosis put them. Counted on P01's output: **53 direct changes from one
player to another** (each one counted as a release and a catch), 42 of them while the ball was held by one player in
the labels (the holder flipping between two close players, or the player tracks swapping); and **195 changes through
"air"** for ~100 real ones. Filling gaps does not touch either: the "air" decided by distance to the boxes is too
noisy at the frame level.

## Conclusion

**Neutral.** Kept behind its value (off by default). Possession by "closest box within reach" is limited by the
boxes themselves; the next idea decides air from the **ball's own motion** (a pass or a shot is the ball flying fast
and far, a dribble moves with the player), with its values chosen on the same train clips.
