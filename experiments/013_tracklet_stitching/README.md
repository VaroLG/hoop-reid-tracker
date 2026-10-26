# 013 · Joining a player's fragments after the video (offline tracklet stitching)

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Builds on** | [012](../012_reid_osnet/): OSNet fine-tuned on SportsMOT tells players apart twice as well (rank-1 0.42 → 0.71), but the online tracker barely uses it. Its memory revives nobody, and 17–21 identities are created for 10 players |
| **Question** | Seeing the whole video at once, can appearance join the fragments of each player without ever joining two players? |

## Idea

After tracking, every id is a *tracklet*. Two tracklets of the same player never
appear in the same frame, and each one can be described by many clean crops, not
by the single, often occluded crop the online memory has to decide on. This is
GTA (Global Tracklet Association) on top of Deep-EIoU.

Rules (`src/hooptrack/tracking/stitch.py`):

1. **Never together**: two groups with a box in the same frame are two players.
2. **Look alike**: median cosine distance over all crop pairs < `thresh`.
3. **No doubt**: the pair must beat its *competitors* by `margin`. A competitor is a
   tracklet on screen at the same time as the candidate, typically a team-mate.
   Tracklets that never coexist are not competitors: a player lost three times
   leaves three fragments in a row.

Greedy from the most similar pair; the merged group keeps the earliest id.

Thresholds come from the ReID calibration on the **train** split
(`outputs/calibration/osnet.json`), never from the validation clips:
`thresh` = hard-negative p5 (0.077), `margin` = hard-negative p5 − p1 (0.021).

## How to run

Post-processes the 011b run (best online configuration, ResNet in the tracker) with OSNet for
the stitching. No re-tracking. Embeddings are cached, so different thresholds are instant:

```powershell
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth `
    --out outputs/runs/013_stitch
```

Tests: `tests/test_stitch.py`.

## Results

011b run post-processed (ResNet in the tracker, OSNet only for the stitching), thresholds from the
train calibration (`thresh` 0.077, `margin` 0.021). Took ~10 minutes for the 4 clips on the laptop CPU.

| 4 arenas, OVERALL | IDF1 | MOTA | ID switches | Frag | Identities per clip |
|---|---|---|---|---|---|
| 011b (online tracker) | 74.6 | 96.0 | 49 | 95 | 17–21 |
| **011b + stitching (013)** | **77.2** | **96.1** | **43** | 95 | 16–19 |

| Clip | IDF1 before → after | ID switches | ids | merges / refused |
|---|---|---|---|---|
| NCAA c005 | 68.9 → 72.5 | 11 → 10 | 18 → 16 | 2 / 0 |
| Changsha c001 | 85.6 → **90.6** | 8 → 5 | 21 → 16 | 5 / 0 |
| Tenerife c002 | 69.1 → 69.1 | 10 → 10 | 17 → 16 | 1 / 0 |
| Istanbul c012 | 69.5 → 70.6 | 20 → 18 | 21 → 19 | 2 / 7 |

**Were the merges right?** Each tracklet was labelled with the ground-truth player it covers most
(`outputs/_review/mergecheck.py`):

| | count |
|---|---|
| merges of two fragments of the same player | 6 |
| merges involving a tracklet that is not a ground-truth player (a stray track, 2–35 frames) | 3 |
| **wrong merge: two different players** (Changsha, distance 0.070, close to the 0.077 threshold) | **1** |
| refused as ambiguous, would have been right | 3 |
| refused as ambiguous, would have been wrong or not a player | 4 |

The one wrong merge costs less than the six right ones gain. The doubt rule did its job in
Istanbul, where most of the would-be merges were team-mates. MOTA does not move: stitching only
renames ids, it never adds or removes a box.

**Positive.** +2.6 IDF1 and −6 ID switches with no new tracking run, using the ReID model that the
online tracker could not exploit (012). The remaining identity errors are mostly swaps inside
crossings while both players are visible, which stitching cannot see: a swap leaves two tracklets
that each mix two players. Next: an appearance check when an occlusion ends (swap repair), which
would also give stitching cleaner tracklets.
