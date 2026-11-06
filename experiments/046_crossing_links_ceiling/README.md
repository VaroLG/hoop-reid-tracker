# 046 · Ceiling of a perfect crossing verifier (before spending GPU on SAM2)

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [038](../038_closed_set_continuity/) (adopted) · [042](../042_soft_team_cost/)/[043](../043_global_reassignment/): the remaining errors are assignment errors |
| **Question** | Mask propagation (SAM2 / EdgeTAM through each crossing, as McByte / SAMIDARE do) could tell which of the 2,078 cuts of 038 are the same person on both sides. Before building it: if it were PERFECT, how much would 038 gain? |
| **Clips** | all 15 validation clips. **Uses ground truth** (a ceiling, never a result) |

## Change

* [`scripts/crossing_links_oracle.py`](../../scripts/crossing_links_oracle.py): for every cut of every online tracklet,
  the link (piece before, piece after) when both are mostly the same ground-truth player. 2,078 cuts: **1,955 same
  person (94 %)**, 110 swaps, the rest without a ground-truth player.
* `--link-file` ([`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py), `join_pieces`): the linked pieces
  are joined into one piece (their crops together) before the closed set. The same switch will take the links of a
  real verifier. 100–215 pieces per clip become 20–40.

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | IDSW |
|---|---|---|---|---|---|
| **038** | **79.7** | 84.9 | **74.9** | **89.6** | 379 |
| 038 + perfect links, continuity 0.3 | 76.7 | 85.9 | 68.5 | 83.8 | **277** |
| 038 + perfect links, continuity 0 | 77.9 | 86.1 | 70.5 | 84.6 | 305 |

(Continuity 0 because after joining, the only online links left are the swaps, where the bonus is always wrong.)

## Reading

1. **Even perfect crossing links do not help the closed set as it is.** Fewer, longer pieces mean each wrong decision
   now costs a long stretch of boxes, and the 10 seeds (the frame with 10 pieces on screen, the longest) become less
   reliable: two seed slots end up owned by the same player on 5 of 15 clips (1 of 15 in 038).
2. So a SAM2 crossing verifier, built now, would be spent on a step that is not the bottleneck. **Not built.** The
   weak point is how pieces are matched to the 10 players (seeds and appearance), consistent with 042/043.

## Conclusion

**Negative ceiling → SAM2 verifier postponed.** `--link-file` stays (off by default) for when the assignment changes.

## Commands

```powershell
python scripts/crossing_links_oracle.py --out outputs/runs/046_links_oracle.csv
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --split-crossings --closed-set 10 --link-file outputs/runs/046_links_oracle.csv --out outputs/runs/046_oracle_links_c0
```
