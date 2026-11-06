# 043 · Re-deciding every piece at once (integer program) instead of greedily

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [038](../038_closed_set_continuity/) (adopted) · [042](../042_soft_team_cost/) diagnosis: 73 % of the wrong boxes are clean pieces given to the wrong player, ~2/3 of them because the right player was already taken by an earlier wrong decision |
| **Question** | Is the greedy order the problem? If all pieces are decided jointly, with the same evidence, do the errors go down? |
| **Clips** | all 15 validation clips |

## Change (one switch, off by default)

`--closed-refine` ([`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py),
`refine_closed_set` in [`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py)): start from 038's answer,
describe each of the 10 players by the crops of his current pieces, then solve **exactly** (scipy `milp`, HiGHS):

* minimise Σ x[piece, player] · distance − 0.3 · Σ (two linked online pieces with the same player) + 1.5 · Σ (piece keeps its own id);
* one label per piece; two pieces on screen together never share a player; seeds cannot take their own id.

Players are re-described from the new answer and the program solved again (at most 3 rounds). Same evidence as 038:
same pieces, embeddings and continuity bonus (0.3); the 11th-id price 1.5 was **fixed before running** (above every
usual distance, so a piece keeps its own id only when nobody is free, as in 038). No other value was tried.

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| **038** | **79.7** | 84.9 | **74.9** | **89.6** | 96.6 | 379 |
| 043 global re-assignment | 78.6 | 85.2 | 72.6 | 88.2 | 96.6 | **331** |

Per clip ([`hota.csv`](hota.csv)): better on 4 (NCAA c007 +0.8, c011 **+3.3**, FIBA c004 +1.2, 2010 c008 +0.9), equal
on 3, worse on 8 (FIBA c005 −5.9). 0–38 pieces move per clip.

## Reading

1. **Optimising harder does not help: the evidence is the limit, not the order.** With the same distances, the
   exact optimum is no better than the greedy answer: where the greedy was wrong, the appearance distances
   themselves prefer the wrong player. (On val the SigLIP distance is 0.40 for the same player vs 0.74 for a
   team-mate *in median*; the hard pieces live in the overlap.)
2. Fewer switches (379 → 331) but fewer right boxes: the solver makes the answer more consistent with the
   (wrong) appearance.
3. Together with 039–042: rules, adaptation, kit terms and now exact optimisation all reshuffle the same weak
   evidence. **The next gain must come from new evidence** (a better ReID model, readable jersey numbers, or an
   online tracker that does not need to cut at every crossing), see the research notes.

## Conclusion

**Negative, kept off by default.**

## Commands

```powershell
# embeddings copied from 038 (same pieces)
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --split-crossings --closed-set 10 --continuity 0.3 --closed-refine --out outputs/runs/043_global_refine
python scripts/eval_hota.py --config configs/player.yaml --runs 038=outputs/runs/038_continuity 043=outputs/runs/043_global_refine
```
