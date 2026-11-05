# P01 · Who has the ball: fixed rules on the tracked ball and players (possession chapter)

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Builds on** | [B05c](../../ball/B05b_static_and_weak/) ball tracks · [038](../../038_closed_set_continuity/) player tracks |
| **Question** | With simple rules, does the system know when a player has the ball and when it is in the air? |
| **Test** | the 4 ball test clips (3,883 frames), every catch and release marked by hand (P02): 100 changes; the ball is held in 79 % of the frames |

## Method

* [`hooptrack/ball/possession.py`](../../../src/hooptrack/ball/possession.py): a player can have the ball if it is in his
  box widened for the arms and raised above the head, if it moves with him (less than 1.5 ball diameters per frame
  relative to him) and he is the closest at hand height; a holder must last 4 frames, drops of ≤ 3 frames are filled,
  and frames without a ball keep the holder for ≤ 10 frames. Values fixed before measuring.
* Labels ([`scripts/possession/p02_label_events.py`](../../../scripts/possession/p02_label_events.py)): `catch` /
  `release` at the frames where the state changes (a dribble is possession); each frame takes the state of the last mark.
* Scoring ([`scripts/possession/p03_eval_possession.py`](../../../scripts/possession/p03_eval_possession.py)): held / air
  per frame, and events found within 5 frames (0.2 s).

## Results ([`per_clip.csv`](per_clip.csv))

| | frames right (held / air) | events found (recall) | events right (precision) | timing of the found events |
|---|---|---|---|---|
| **P01** | **74.8 %** | **80 %** (80 of 100) | 27 % (80 of 292) | median 1 frame |
| "always held" (trivial) | 78.6 % | – | – | – |

Per clip, frames right: NCAA 81 %, Changsha 84 %, FIBA women's WC 63 %, FIBA 2010 73 %.

## Reading

1. **It finds the real changes, on time**: 80 % of the catches and releases, within one frame of the hand mark.
2. **But it invents three times more**: 292 changes for 100. Two patterns (P01's air stretches between two held
   stretches): between the SAME player, 30 stretches, 87 % of them labelled held (a dribble bounce or the ball hidden
   for a moment, up to ~0.4 s); between two DIFFERENT players, 60 stretches, half of them labelled held (the holder
   flips between an attacker and the defender on him). So it says "air" too often: per frame it is below the trivial
   "always held".

## Conclusion

**Negative as a frame-by-frame state; promising as an event detector (recall 80 %, on time).** The rules need two
values the test clips should not choose: how long "air" must last to be a real release, and when a change of holder
is real. Next (P01b): mark the 3 held-out **train** clips the same way, choose those values there (with the
ground-truth player boxes), test once here.
