# B05 · One ball through the clip: offline tracking over the detector's candidates

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Builds on** | [B02–B04](../B03_ball_detector/): our detector finds 58 % of the balls at 76 % precision, frame by frame |
| **Question** | Using that there is one ball that can only move so far between frames, can a tracker recover the missed balls and drop the false alarms? |
| **Clips** | the 4 test clips of B04 (detector on every frame; scored on the 390 hand-clicked frames) |

## Method

[`hooptrack/ball/track.py`](../../../src/hooptrack/ball/track.py) (tests in `tests/test_ball_track.py`): every
candidate of every frame (confidence ≥ 0.05, [`scripts/ball/b05_detect_all.py`](../../../scripts/ball/b05_detect_all.py))
is a node worth `log(conf / 0.1)`; the best sequence over the clip is found exactly by dynamic programming. Two nodes
can follow each other if at most 15 frames apart and within 3 ball diameters per frame (a pass reaches ~2 per frame;
camera motion adds); a skipped frame costs 0.05; a segment can end and another start later for 1.5. Inside a segment
the frames in between are interpolated. All values fixed before running, from physics, not tuned.
Scored by [`scripts/ball/b05_track_ball.py`](../../../scripts/ball/b05_track_ball.py) as B04.

## Results ([`per_clip.csv`](per_clip.csv))

| 4 test clips, 334 visible balls | found (recall) | precision | elsewhere | missed | false alarm without a ball |
|---|---|---|---|---|---|
| detector alone (conf 0.25, B04) | 193 (58 %) | **76 %** | 45 | 96 | **15** |
| **detector + tracker** | **245 (73 %)** | 66 % | 82 | **7** | 47 |

Per clip, recall / precision, detector → tracker:

| NCAA c001 | FIBA Changsha | FIBA women's WC c003 | FIBA 2010 c012 |
|---|---|---|---|
| 76 / 92 → **90** / 80 % | 57 / 80 → **78** / 73 % | 44 / 54 → **52** / 44 % | 53 / 88 → **78** / 71 % |

The tracker gives a position in 95–100 % of the frames (17–35 % of them interpolated).

## Reading

1. **More balls found on all four clips** (+52 found, misses 96 → 7): interpolation between detections recovers the
   ball in the hands and in passes.
2. **But it almost never says "no ball"**: it answers in nearly every frame, so frames where the ball is hidden get a
   position anyway (false alarms 15 → 47), and when the path follows a false alarm for a while the answer is in the
   wrong place (45 → 82; worst on the FIBA women's clip, with its yellow kits). Precision drops on all four clips.
3. Balance (F1): 0.66 → 0.69. A gain, but a trade-off, not a clear win. The fix is to let the tracker answer
   "no ball" when a segment has weak support; its values must be chosen on held-out **train** clips (labelled, not
   used for the detector's weights), not on these test clips.

## Conclusion

**Mixed: recall 58 → 73 %, precision 76 → 66 %.** Kept as is (the values were fixed beforehand). Next: B05b, the same
tracker with a "no ball" decision for weak segments, its values chosen on the three held-out train clips.
