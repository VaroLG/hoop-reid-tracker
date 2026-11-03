# 040 · Embeddings adapted to each clip, 5 per team, at most 10 ids

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [038](../038_closed_set_continuity/) closed set of 10 + continuity (adopted) |
| **Question** | 038's errors come from team-mates the ReID model confuses. (a) Can the embeddings be adapted to each clip with no labels? (b) Does a hard "5 per team, never to the other team" rule help? (c) And "at most 10 ids"? |
| **Clips** | all 15 validation clips; one change per run, on top of 038 |

## Changes (all off by default)

* **040a** `--adapt-dim 9` ([`reid/adapt.py`](../../src/hooptrack/reid/adapt.py)): a linear map of the embeddings,
  fitted per clip in closed form (Fisher's discriminant): crops of one piece are one person, pieces on screen
  together are two people. 9 dimensions = classes − 1 for 10 players. **Not blind**: 64 dimensions were looked at
  first on 4 validation clips (team-mate AUC went down), then 9 (up on the 4 clips, +0.01 to +0.03); this choice was
  made after seeing those clips. Distances are rescaled per clip so the 0.3 continuity bonus keeps its meaning.
* **040b** + `--closed-teams`: the 10 seeds are split into the two groups of 5 most similar in kit (zero-shot
  SigLIP torso vectors, all 126 splits tried); every piece takes the closer team and can only go to a player of it.
* **040c** + `--roster-drop`: pieces that fit no player are removed (at most 10 ids per clip).

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | switches / min | new ids |
|---|---|---|---|---|---|---|---|---|
| **038** | **79.7** | 84.9 | **74.9** | **89.6** | **96.6** | 379 | 43.7 | 96 |
| 040a + clip adaptation | 78.5 | 85.2 | 72.3 | 87.3 | 96.6 | 365 | 42.2 | 96 |
| 040b + 5 per team | 77.0 | 85.1 | 69.7 | 85.1 | 96.5 | 476 | 55.4 | 186 |
| 040c + at most 10 ids | 77.1 | 82.0 | 72.5 | 87.3 | 92.6 | **293** | **33.3** | **42** |
| *029, for reference* | *76.9* | *85.7* | *69.1* | *83.9* | *96.6* | *278* | *31.3* | *52* |

Per clip ([`hota.csv`](hota.csv)): 040a is worse than 038 on 8 clips, better on 3, equal on 4.

## Reading

1. **The adaptation separates team-mates a little better on average, and assigns worse.** A better AUC over all
   pairs does not mean better decisions on the hard pairs: the closed-form map, fitted on ~150 pieces, also moves
   the pieces 038 already got right (FIBA c001 −4.9, FIBA 2010 c008 −6.4).
2. **A hard team rule needs right team labels, and they are not.** The kit of a piece cut at a crossing comes from
   crops that are often occluded; when it is wrong, the rule forbids the right player, the piece fits nobody and
   becomes a new id (96 → 186 new ids). Cross-team switches hardly move (206 → 214).
3. **"At most 10 ids" trades boxes for switches**: removing what fits nobody gives the fewest switches of the
   closed-set runs (33 / min, about 029's level, with IDF1 87.3 instead of 83.9), but deletes 4 % of the boxes
   (MOTA 96.6 → 92.6), and some of them are players.

## Conclusion

**Negative: 038 stays the best mode.** Documented and kept behind the three switches. 040c is a product option
(at most 10 ids, fewer switches, fewer boxes), not a better tracker.
