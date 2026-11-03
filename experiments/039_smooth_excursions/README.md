# 039 · Smoothing short excursions after 038 (diagnosis: nothing to smooth)

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [038](../038_closed_set_continuity/) closed set + continuity (adopted) |
| **Question** | 038 makes more identity switches than 029 (379 vs 278) but a higher IDF1. If its errors are single pieces given to the wrong player between two right ones, can a parameter-free rule give them back? |

## Change (off by default)

`--smooth` in [`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py), `smooth_sandwiches` in
[`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py): if both online neighbours of a piece (the pieces
before and after it inside the same online tracklet) belong to the same player B, and B has no box in that piece's
frames, the piece goes to B; repeated until nothing changes. Test in `tests/test_stitch.py`.

## Diagnosis before running it on all clips

On the pieces and embeddings of 038, with the ground truth, the rule was applied to 4 clips (FIBA 2010 c014, the
clip with the median gain of 038; NCAA c007 and c011; FIBA c003): it moves **0, 0, 0 and 1 piece**. The wrong pieces
of 038 on c014 (44 of 233) were classified by their online neighbours:

| wrong piece of 038 (c014) | pieces |
|---|---|
| both neighbours with the **same** wrong player (a whole run of pieces went to the wrong player) | 22 |
| neighbours with different players, one of them the same wrong player | 12 |
| neighbours with two other players | 5 |
| only one neighbour, with the same wrong player | 3 |
| a wrong piece between two right pieces of one player (what the rule fixes) | 0 |

The hypothesis was wrong: 038's errors are not isolated pieces. The continuity bonus makes a wrong decision
**spread along the online tracklet**: when a piece goes to the wrong player, its neighbours follow it, until the
next crossing where appearance pulls them back. The identity does come back (hence the high IDF1), but after a run
of pieces, not after one.

## Conclusion

**Negative by diagnosis (no effect)**; not run as a full experiment, since it changes at most one piece per clip.
Kept behind `--smooth`, off by default. Fixing these errors needs a better decision at the crossing where the run
starts, not a smoothing after it.
