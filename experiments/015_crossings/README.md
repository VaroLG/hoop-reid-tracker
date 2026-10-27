# 015 · Identity swaps in crossings: cut, repair, and what appearance can tell

| | |
|---|---|
| **Date** | 2026-10-26 · **negative**, with a diagnosis that sets the next step |
| **Builds on** | 013b (011b + offline stitching, IDF1 81.3) and 014 (a looser motion match only adds swaps) |
| **Question** | Most remaining identity errors happen in crossings. Can the offline step undo them? |

## Diagnosis (on 011b, no new tracking)

Each ground-truth player is followed through time. A box is *in a crossing* when it covers, or is
covered by, another box by at least 0.2 (the tracker's "clean crop" threshold).

| | |
|---|---|
| identity changes while the player stays visible | 56, **all** within 10 frames of a crossing |
| IDF1 with a perfect connector on today's tracklets (oracle) | 86.1 (013b: 81.3) |
| IDF1 with a perfect connector after cutting every tracklet at the end of each crossing (oracle) | 97.2 |

With a perfect connector, cutting at crossings would remove almost every identity error. The question
is whether a real connector, using appearance, can put the pieces back together.

## 015a · Cut at every crossing, reconnect with GTA's connector

`split_at_crossings` (`hooptrack.tracking.stitch`): walking each tracklet in time, a new segment starts at
the first clean box after a crossing; the crossing frames stay with the segment before. Then
`stitch_gta` joins the segments, as in 013b. Run with `scripts/stitch_tracklets.py --split-crossings`.

| 4 arenas, OVERALL | IDF1 | MOTA | ID switches | pieces per clip → identities |
|---|---|---|---|---|
| 013b (no cut) | **81.3** | **96.1** | **34** | 17–21 → 11–12 |
| 015a cut + connector | 64.3 | 95.3 | 217 | 106–181 → 15–17 |

**Negative.** The connector sees ~140 short pieces per clip and joins them by appearance only. Many pieces of
team-mates get joined. Cutting also throws away the strongest clue: after almost every crossing the
tracker was right.

## 015b · Keep or swap at each crossing exit (prototype, not in the package)

Same cut points, but a local decision instead of a global one. For the players leaving a crossing
together (cut within 5 frames, 2–4 players), compare each player's appearance *before* (every clean crop
of his identity so far) with each piece *after*. Swap only if another assignment beats "keep" by the
margin from the train calibration (0.021). Prototype: `outputs/_review/swap/proto*.py`.

| 4 arenas, OVERALL | IDF1 | ID switches |
|---|---|---|
| 011b | 74.6 | 49 |
| 015b swap repair | 75.7 | 63 |
| 015b + stitching | 82.4 | 49 |
| 013b (stitching only) | 81.3 | 34 |

The +1.1 IDF1 is **not real progress**. Checked against the ground truth, only **1 of the 8 swaps** was right.
The gain comes from where the wrong decisions happened to fall, and the ID switches go up.

## What the crossings really look like

Every cut point was labelled with the ground truth (majority player of the piece before and after):

| cut points (end of a crossing) | count |
|---|---|
| same player before and after | 472 |
| **different player** (a swap, or a track that continued on someone else) | **21** (~1,100 boxes) |
| of which the other player was also cut (a true swap between two tracks) | 8 |
| of which only one track continued (the other track died in the crossing) | 13 |

Then we checked whether appearance can tell the 21 apart from the 472. We used the mean OSNet distance
between everything a track saw before the crossing and the piece after it:

| | distance |
|---|---|
| same player, median / 95th pct | 0.081 / 0.163 |
| different player, range | 0.055 – 0.159 |
| **AUC** (1.0 = perfect, 0.5 = chance) | **0.55** |

**Appearance is at chance here.** After a crossing, a player looks as different from himself as from
the players he crossed. Crossings happen mostly between players close together, often team-mates
wearing the same kit, and our OSNet (rank-1 0.71) cannot separate team-mates.

> **Correction (016).** The team-mate benchmark measures the same decision with ground-truth tracks and clean
> crops. There, OSNet is right at 97 % of team-mate crossings (AUC 0.785), so the model is not at chance. The 0.55
> above was measured on the tracker's pieces (impure history, few clean crops). Even 97 % is not enough, because
> real swaps are rare (4 % of crossings): see [016](../016_teammate_reid_benchmark/).

## Conclusion

Negative. No rule on top of this ReID model can fix swaps in crossings: neither cutting
(015a, −17 IDF1) nor a careful local swap check (015b, 1 of 8 right). The swaps are rare (21 in 4 clips,
4 % of crossings) and the remaining gap is not in the association rules. It is in **telling team-mates apart**.
`split_at_crossings` stays in the code (`--split-crossings`, off) with its tests. The swap-repair
prototype is not added to the package.

**Next:** a better player identity signal that does not depend on the kit. Measure it first on a
team-mate verification benchmark (same player vs team-mate crops, AUC), then plug it in. The candidates are jersey
numbers and ReID trained with team-mates as hard negatives.
