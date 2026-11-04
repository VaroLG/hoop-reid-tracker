# B05b / B05c · Static look-alikes removed on the court plan, "no ball" for weak segments

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Builds on** | [B05](../B05_ball_tracker/): tracker recall 73 %, precision 66 % (detector alone 58 % / 76 %) |
| **Question** | B05's wrong answers were mostly still objects that look like a ball (an orange circle painted on the court, a scorer's table, an advert, faces in the stands) and positions given when the ball was hidden. Do the court plan and a "no ball" decision fix them? |

## Looking at B05's errors first

All 129 wrong answers of B05 on the test clips were checked by eye (tracker position and the click side by side):
of the 47 answers on frames labelled "no ball", 2–3 were labelling mistakes (the ball is visible) and about 20 are
interpolated positions next to a player, where the hidden ball probably was; the rest, and almost all of the 82
answers in the wrong place, were **still look-alikes** that the tracker followed for seconds (they pass any speed
limit). The labels were left unchanged, so that the test is exactly B04's and B05's.

## Changes (both off by default)

* **B05b · static filter** ([`integrations/ball_static.py`](../../../src/hooptrack/integrations/ball_static.py)): every
  candidate is projected to the court plan with its frame's homography (court chapter). A 0.5 m cell (with its
  neighbours) where candidates were seen in at least `min_frames` frames is background: an object fixed in the arena
  stays on the same spot of the plan however the camera moves; the ball does not. Frames without a homography keep
  their candidates. The ball code and the court code still do not import each other: the adapter is the bridge.
* **B05c · weak segments out** (`BallTrackConfig.min_segment_score` in
  [`ball/track.py`](../../../src/hooptrack/ball/track.py)): a run of linked detections worth less than that in total
  is dropped, and the tracker answers "no ball" there.

**The values were chosen on the 3 held-out train clips**, by the best F1 over a 5 × 5 grid
([`scripts/ball/b05b_tune_ball.py`](../../../scripts/ball/b05b_tune_ball.py), [`grid_train.csv`](grid_train.csv)):
`min_frames = 50` (2 s), `min_segment_score = 8`. Then the test clips were scored once.

## Results on the 4 test clips ([`test.csv`](test.csv); 334 visible balls)

| | found (recall) | precision | F1 | elsewhere | missed | answer without a ball |
|---|---|---|---|---|---|---|
| detector alone (B04) | 193 (58 %) | **76 %** | 0.66 | 45 | 96 | **15** |
| B05 tracker | 245 (73 %) | 66 % | 0.69 | 82 | **7** | 47 |
| B05b + static filter | 260 (78 %) | 71 % | 0.74 | 66 | 8 | 39 |
| **B05c + weak segments out** | **261 (78 %)** | 73 % | **0.75** | **62** | 11 | 36 |

Per clip, F1: B05 → B05c: NCAA 0.84 → **0.87**, Changsha 0.75 → **0.79**, FIBA women's WC 0.48 → **0.62**, FIBA 2010
0.74 → 0.74 (unchanged). Against the detector alone, B05c is better on all four clips.

## Reading

1. **The static filter does what it was built for**: on the FIBA women's World Cup, the clip with the painted orange
   circles and the "LOTERÍAS" table, found balls go from 52 to 65 and wrong places from 48 to 35. It never hurts.
2. Dropping weak segments adds a little precision for a little recall (3 more missed balls).
3. Final ball localisation: **78 % of the visible balls found at 73 % precision**, against 11 % / 29 % for the COCO
   detector at the start of the chapter. Part of the remaining "answers without a ball" are probably the hidden
   ball next to a player, which a click could not label.

## Conclusion

**Positive: B05c is the ball tracker** (static filter 2 s + weak segments out, values from train). Next: possession —
which player has the ball — from the tracked ball and the players' boxes.
