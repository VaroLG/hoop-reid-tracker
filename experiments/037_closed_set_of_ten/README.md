# 037 · A closed set of 10 players, built from pieces cut at the crossings

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [036](../036_roster_of_ten/) (the 10-player rule on stitched groups: nothing could be assigned) · [015](../015_crossings/) cut at crossings · [028](../028_siglip_reid_finetune/) fine-tuned SigLIP |
| **Question** | If the 10-player rule is applied to pieces that hold one player each (cut where the swaps happen), can a closed-set assignment by appearance replace the stitching and fix swaps as well as fragments? |
| **Clips** | all 15 validation clips |

## Change (off by default)

`--split-crossings --closed-set 10` in [`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py),
`assign_closed_set` in [`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py), instead of the connector:

1. the online run (011b) is cut at the end of every crossing (015): 106–234 pieces per clip;
2. **seeds**: the frame with 10 pieces on screen together whose pieces are longest gives the 10 players (10
   people in one frame are 10 different people; no team labels, they failed on two clips in 036); each player
   keeps a gallery of fine-tuned SigLIP embeddings;
3. **assignment**: every other piece goes to one of the 10, never to a player who already has a box in one of its
   frames, by appearance distance to the gallery, without threshold;
4. pieces that fit nobody (on screen with all 10) keep their own id.

Two versions of step 3:

* **037 (v1)**: rounds of one Hungarian assignment between 10 pieces and the 10 players. **A bug of design**: the
  Hungarian gives each player at most one piece per round, so 10 pieces that are *not* on screen together were
  still forced onto 10 different players.
* **037b**: one piece at a time, the most confident pair (piece, player) of all first; the gallery of that player
  grows and his distances are recomputed.

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | IDSW | switches / min | to a team-mate | to an opponent |
|---|---|---|---|---|---|---|---|---|
| 029 | **76.9** | **85.7** | **69.1** | **83.9** | **278** | **31.3** | **34** | **176** |
| 037 v1 (Hungarian rounds) | 44.6 | 84.4 | 23.6 | 48.8 | 1,435 | 169.9 | 479 | 661 |
| 037b (most confident first) | 74.5 | 84.8 | 65.4 | 81.7 | 612 | 71.6 | 173 | 248 |

HOTA per clip ([`hota.csv`](hota.csv)), 029 → 037b:

| NCAA | c001 | c003 | c005 | c007 | c008 | c011 |
|---|---|---|---|---|---|---|
| | 85.0 → **86.4** | 90.0 → 89.9 | 72.1 → **80.6** | 62.7 → **77.9** | 71.5 → **78.1** | 67.2 → **75.4** |

| FIBA / FIBA 2010 | Changsha c001 | c001 | c002 | c003 | c004 | c005 | 2010 c008 | c012 | c014 |
|---|---|---|---|---|---|---|---|---|---|
| | 87.5 → 75.8 | 75.1 → 72.0 | 74.3 → 62.6 | 78.9 → 64.3 | 79.7 → 72.1 | 76.6 → 67.7 | 66.2 → 56.0 | 70.1 → 67.9 | 70.7 → 67.3 |

## Diagnosis (ground truth, pieces and embeddings of these runs)

* The pieces are pure: 97–99.6 % of each piece's boxes belong to one player.
* The seeds are right: 10 different players on 7 of the first 8 clips checked (9 on the other).
* Share of pieces given to the right player with 037b, same 8 clips: **75–98 %** per clip (NCAA c001 98 %, c003 98 %, c008
  92 %; FIBA c001 75 %). With v1 it was 26 % (3 clips); to the nearest seed with no rule at all, 71 % (Changsha).

## Reading

1. **On the NCAA clips (one game, two very different kits), the closed set is the best result of the project**:
   +1.4 to +15.2 HOTA on 5 of 6 clips. There the appearance model tells the players apart well, and the rule (10
   players, never two boxes of one player in one frame) turns that into identities that last the whole clip.
2. **On the FIBA clips it is clearly worse.** Cutting at every crossing throws away the online tracker's motion
   continuity, which keeps the right identity through ~96 % of the crossings; each piece is then decided by
   appearance alone, and when team-mates look alike (FIBA women's World Cup, FIBA 2010) one wrong piece in five
   gives two switches each. 029 keeps that continuity and only repairs where it is unsure.
3. On the whole set it loses (74.5 vs 76.9, worse on 9 clips of 15), so by the rules of this project it is **not
   adopted**. The per-clip split is too clear to ignore: a criterion that says, without ground truth, when the
   appearance is reliable enough for the closed set (for example how far each piece's best player is ahead of the
   second) could pick per clip; it would have to be calibrated on the train split, not here.

## Conclusion

**Negative on the whole set, strongly positive on one game.** Documented, kept behind `--closed-set`, off by
default; 029 stays the default. 037 v1 is kept (`batch` argument) to reproduce the design bug.

## Commands

```powershell
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --split-crossings --closed-set 10 --out outputs/runs/037b_closed_set
python scripts/eval_hota.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft 037b=outputs/runs/037b_closed_set
python scripts/switch_report.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft 037b=outputs/runs/037b_closed_set
```
