# 013 · Joining a player's fragments after the video (offline tracklet stitching)

| | |
|---|---|
| **Date** | 2026-10-26 · best result: 013b |
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

## External reference: the original GTA code

[gta-link](https://github.com/sjc042/gta-link) (MIT), run unchanged on the same 011b results with **our** OSNet
(`generate_tracklets.py` then `refine_tracklets.py`), so that only the association algorithm differs.
GTA's own ReID checkpoint was not used: it is 10× slower on CPU and it is not clear whether it saw the
SportsMOT validation games. GTA embeds every box of every frame (~25,000 crops, 13 min on a 2-core CPU
with denormal flushing, see below). Its parameters are the published SportsMOT defaults
(`eps 0.6, min_samples 10, max_k 3, min_len 100, spatial_factor 1.0, merge_dist_thres 0.4`), not tuned here.

| 4 arenas, OVERALL | IDF1 | MOTA | ID switches | Identities per clip | IDF1 NCAA / Tenerife / Istanbul / Changsha |
|---|---|---|---|---|---|
| 011b online | 74.6 | 96.0 | 49 | 17–21 | 68.9 / 69.1 / 69.5 / 85.6 |
| 013 ours (conservative connector) | 77.2 | 96.1 | 43 | 16–19 | 72.5 / 69.1 / 70.6 / 90.6 |
| GTA connector, our threshold (0.077) | 76.6 | 96.0 | 46 | 16–19 | 72.5 / 69.1 / 70.9 / 88.6 |
| **GTA connector, published defaults** | **80.5** | **96.1** | **36** | **11–12** | 77.2 / 69.1 / 74.2 / **95.1** |
| GTA splitter + connector, defaults | 80.4 | 96.1 | 38 | 11–12 | 76.5 / 69.1 / 74.2 / 95.1 |

(GTA's script always runs the connector, so a "splitter only" row is not possible without editing it.)

**Reading.**

1. **Our stitcher is too conservative.** With the same ReID model, GTA's connector with its published settings
   brings the identities down to **11–12 for 10 players** (IDF1 +5.9 over the online tracker, +3.3 over ours).
   GTA does not refuse doubtful merges. Instead it (a) forbids a merge when the gap between where one tracklet
   ends and the other starts is larger than the players' range of motion in the clip, and (b) merges
   hierarchically with a loose appearance threshold. In our clips its spatial rule is a better safeguard than
   our "doubt" rule, which also refused right merges.
2. **The splitter adds nothing here** (80.4 vs 80.5). With our tracker, a swap rarely leaves one long tracklet
   with two clear appearance clusters.
3. Purity check: the share of an id's boxes that belong to its majority player goes from 0.863 to 0.837. There
   are a few wrong merges, but far fewer than the right ones.
4. Tenerife does not change in any variant. Its fragments are not separated by gaps the connector can bridge.

**Next:** reproduce GTA's connector rules (spatial range, mean distance, hierarchical merging) inside
`hooptrack.tracking.stitch` as an option, check that it gives the same numbers, and make it the offline mode.

## OSNet on CPU: denormal floats

The fine-tuned OSNet was 8× slower than ResNet-50 on the laptop (012: 5.4–6 s per frame). The cause is
subnormal floats: activations so close to zero that the CPU takes a slow path. `torch.set_flush_denormal(True)`
(now in `OSNetExtractor`) gives 0.35 → 0.056 s per crop with identical embeddings (max difference 0.0).

## 013b · GTA's connector rebuilt in the repo (`stitch_gta`, default mode)

`hooptrack.tracking.stitch.stitch_gta` rebuilds the connector rules, documented in the code: mean
cosine distance over all crop pairs, distance 1 for groups on screen together, hierarchical merging
with recomputed distances, and the spatial jump check. `scripts/stitch_tracklets.py --mode gta` (now the
default) uses GTA's published settings: merge distance 0.4, spatial factor 1.0.

**Reproduction check.** Fed with the embeddings computed by the original code (every box), it gives
exactly the original numbers on every clip: IDF1 80.5, MOTA 96.1, 36 ID switches, 11–12 identities.
On these clips the spatial check never refused a merge (factor 1.0 = the whole range the players cover).
The gain comes from the time rule plus the loose, mean-based appearance threshold.

**With our crop sampling** (12 clean crops per tracklet instead of every box: ~10× fewer crops, about
20 s per clip on a 2-core CPU):

| 4 arenas, OVERALL | IDF1 | MOTA | ID switches | Identities per clip |
|---|---|---|---|---|
| 011b online | 74.6 | 96.0 | 49 | 17–21 |
| original GTA (every box) | 80.5 | 96.1 | 36 | 11–12 |
| **`stitch_gta`, 12 clean crops per tracklet** | **81.3** | **96.1** | **34** | **11–12** |

Per clip IDF1 (011b → 013b): NCAA 68.9 → 77.2 · Tenerife 69.1 → 73.9 · Istanbul 69.5 → 72.9 · Changsha 85.6 → 95.1.

Clean crops (no other box covering them) describe a player better than every box, half of which mix two
players in crossings. That is also why Tenerife improves here and not with the original code.
Confirmed on the laptop on the original frames, with identical numbers (`outputs/runs/013b_stitch_gta`).

**Conclusion.** Offline tracklet stitching with GTA's connector is the best result of the project:
IDF1 74.6 → **81.3**, ID switches 49 → 34, 11–12 identities for 10 players, with no new tracking run and
about a minute of CPU per clip. It is the offline mode: `scripts/stitch_tracklets.py` after `hooptrack track`.
