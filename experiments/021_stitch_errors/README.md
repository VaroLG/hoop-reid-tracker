# 021 · What is left after 020, and a team rule in the connector

| | |
|---|---|
| **Date** | 2026-10-28 · diagnosis + **neutral** rule (15 val clips: 81.8 → 81.8) |
| **Builds on** | 020: team cut with SigLIP labels, IDF1 81.8 on the 15 val clips, at the ceiling of oracle team labels (82.0) |
| **Question** | Where do the remaining identity errors come from? Can the team labels also stop the connector from joining two kits? |

## Diagnosis

`outputs/_review/team/diag021.py` (review only) rebuilds 020 with oracle team labels (same output as SigLIP on the
clips that lose points), piece by piece. Every ground-truth box whose final id is wrong (optimal GT ↔ id matching, as in
IDF1) is assigned to one cause:

| cause | wrong boxes | share of matched boxes |
|---|---|---|
| **wrong join**: a clean piece of this player, joined to another player | 7,771 | **6.7 %** |
| **not cut**: inside a piece of someone else, no cut (a swap the team cut cannot see) | 5,351 | 4.6 % |
| **not joined**: a clean piece of this player, never re-joined | 4,777 | 4.1 % |
| **miscut**: inside a piece of someone else, in a tracklet that was cut | 2,413 | 2.1 % |

* **Where the cut lands is not the problem.** Of 53 cuts, 26 follow a real change of player (the other 27 split one
  player and are re-joined, harmless). The cut comes a median of 12 frames (0.5 s) after the real change. Only 3 cuts
  are more than 1 s late.
* **The connector is the problem.** `merges021.py`: 112 merges, **42 wrong**. 32 of them join two team-mates and
  10 join two opponents. The ReID distances of right and wrong merges overlap (median 0.075 vs 0.093, all below 0.13),
  so no threshold separates them.

## Idea: never join two kits

With reliable team labels (020), a piece with a clear team (≥ 3 labelled boxes, ≥ 2/3 of them agreeing, fixed in
advance) gets that team. `stitch_gta(..., teams=...)` treats two groups of different known teams like two groups on
screen together: distance 1, never merged. Pieces without a clear team are free, as before. One change, no new parameter
tuned on val. `--team-merge-rule` in `scripts/stitch_tracklets.py`, test in `tests/test_stitch.py`.

## Results (15 val clips)

| | IDF1 | MOTA | ID switches |
|---|---|---|---|
| 013b: stitching | 79.9 | 96.7 | **214** |
| 020: team cut, SigLIP labels | **81.8** | 96.6 | 267 |
| 021: + team rule in the connector, SigLIP labels | **81.8** | 96.6 | 276 |
| 020 with oracle labels | 82.0 | 96.6 | 276 |
| 021 with oracle labels | 81.9 | 96.6 | 280 |

Per clip, 021 against 020 (SigLIP): 1 better (Tenerife c001 +1.6), 2 worse (NCAA c007 −0.9, c011 −0.4), 12 equal.

## Reading

1. **The rule does what it says, and it does not help.** With oracle labels, wrong merges between opponents drop
   from 10 to 2. Wrong merges between team-mates rise from 32 to 35, and right merges stay at 70: a piece that can no
   longer join an opponent joins the wrong team-mate instead. GTA has no "doubt" rule. As long as some group is closer than
   0.4, the piece goes somewhere.
2. **What is left is the team-mate problem.** Wrong joins between team-mates (6.7 %) and swaps between team-mates that no team
   cut can see (most of the 4.6 % "not cut") are the two largest causes. Both need a signal that tells **team-mates** apart.
   Appearance does not (016–018: AUC ~0.71).

## Conclusion

**Neutral, not adopted.** `--team-merge-rule` stays available, off. 020 stays the offline mode. The next lever is a signal
independent of appearance that separates team-mates: the **jersey number**.
