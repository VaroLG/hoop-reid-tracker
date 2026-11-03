# 036 · A roster of 10: every identity must be one of the 10 players

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [029](../029_stitch_siglip_reid/) stitching (adopted) |
| **Question** | A game has 10 players on the court (5 per team), and every SportsMOT clip has exactly 10 ground-truth identities. 029 ends with 11–14 ids per clip. Can the leftover ids be given back to one of 10 players, with a gallery of embeddings per player? |
| **Clips** | all 15 validation clips |

## Starting point

Taking the 10 longest ids of each 029 clip as "the players", the other ids hold **6.2 %** of the boxes, and **89 %**
of those boxes are real players (the rest: referees and other false detections).

## Change (off by default)

`--roster 5` in [`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py), `assign_roster` in
[`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py), after the 029 stitching:

* **roster**: the 5 longest groups of each team (team = majority SigLIP team label of the group's pieces); each
  keeps a gallery with all its fine-tuned SigLIP embeddings;
* **assign**: every other group goes, in rounds of one Hungarian assignment, to the closest roster player in
  appearance, of its own team, and **never to one on screen at the same time** (two boxes in one frame are two
  people);
* **leftovers** (groups that fit nobody): kept (036a) or dropped (036b, `--roster-drop`: at most 10 ids per clip).

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| 029 | **76.9** | **85.7** | 69.1 | **83.9** | **96.6** | 278 |
| 036a: roster, leftovers kept | 76.9 | 85.7 | 69.1 | 83.9 | 96.6 | 278 |
| 036b: roster, leftovers dropped | 74.1 | 76.9 | 71.3 | 83.4 | 86.6 | 223 |

**036a assigned no group at all**, on any clip: every leftover was on screen at the same time as every roster
player it could go to. 036b therefore only deletes the leftovers: fewer identity switches (223), higher AssA, but
about 10 % of the boxes are lost (MOTA 96.6 → 86.6, DetA 85.7 → 76.9), most of them real players; on two clips (NCAA c005, c008) the SigLIP
team labels put nearly every box in one cluster, so the roster had 5–6 players and the clip lost almost half of
its boxes (HOTA 72 → 50, 72 → 48).

## Diagnosis (why nothing could be assigned)

Run on the same pieces and embeddings, without teams (roster = 10 longest groups):

* still **no assignment** on any clip: the team labels are not the (only) reason;
* for each leftover, the share of its frames that it shares with each roster group: the best case is usually
  **20–50 %**, with every roster group. With at most 10 players on the court, 11 ids on screen together means
  some ids do not hold one player: after the online tracker's swaps, a "player" of the roster carries pieces of
  several players (031–033: ~31 identity switches per minute), so its frames cover moments when another player
  was the one on screen.

The hard rule "never two boxes of one player in one frame" is right, but it can only be applied to identities that
are **pure**. The 029 groups are not.

## Conclusion

**Negative as built (036a: no effect; 036b: worse).** Documented and kept behind `--roster`, off by default.
The constraint itself is valid and stays the target: it has to be applied to pieces that hold one player each
(cut at the crossings, where the swaps happen) instead of to whole stitched groups.

## Commands

```powershell
# same embeddings as 029 (copied into each output folder first)
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --thresh 1.0365 --split-teams --team-source siglip --team-k 2 --roster 5 --out outputs/runs/036a_roster
# 036b: the same with --roster-drop --out outputs/runs/036b_roster_drop
python scripts/eval_hota.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft `
    036a=outputs/runs/036a_roster 036b=outputs/runs/036b_roster_drop
```
