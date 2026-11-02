# 019 · Giving a stolen id back: cut tracklets where they change team

| | |
|---|---|
| **Date** | 2026-10-27 · +2.2 IDF1 on the 4 clips, **no gain on all 15 val clips** (79.9 → 79.7) |
| **Builds on** | 013b (offline stitching, IDF1 81.3) and the swap diagnosis of 015–018 |
| **Question** | When a crossing gives a player's id to someone else, can the offline step give it back? |

## Diagnosis: the id is not lost, it is stolen

After stitching (013b), each ground-truth player was followed through time and every lasting id change was
classified (`outputs/_review/swap/after_crossing.py`):

| what happens to the player's old id | changes |
|---|---|
| **it goes on, on someone else** | **46** (39 never come back) |
| it stays lost | 5 |
| it ended and stitching did not re-join it | 1 |

Stitching joins fragments, and it already removes them (22 → 1 against 011b). It can never give an id back
while that id is still on screen, on the player who took it. **36 of the 46 thieves play for the other team.**
That is the case appearance handles well: OSNet is right at 100 % of opponent crossings (016, 5 s windows),
against 97 % for team-mates.

## Idea

A tracklet that is team A for seconds and team B afterwards contains two players. 006 tried the team as an
**online** rule, frame by frame, and failed: labels are noisy and hidden inside crossings. **Offline** the whole
tracklet is known. One wrong label is a blip, and a stolen id is a long run of the other team.

1. Label the team of the clean boxes of every tracklet.
2. `split_at_team_changes` (`hooptrack.tracking.stitch`): cut a tracklet only where at least 1 s of one team is
   followed by at least 1 s of another. The cut goes at the end of the last crossing before the change (the
   crossing frames stay with the old player).
3. Stitch as in 013b. Each piece goes back to its player.

A wrong cut is cheap: it leaves two pieces of the same player that never overlap, and the connector re-joins them.
A wrong *swap* (015) was expensive, which is why this works where 015 did not.

## Ceiling (ground-truth teams)

Team labels from the ground truth, everything else real (`outputs/_review/swap/team_ceiling.py`):

| 4 arenas | IDF1 | ID switches | IDF1 NCAA / Tenerife / Istanbul / Changsha |
|---|---|---|---|
| 013b (no cut) | 81.3 | 34 | 77.2 / 73.9 / 72.9 / 95.1 |
| perfect connector, no cut (old ceiling) | 86.1 | 31 | 81.9 / 81.7 / 80.1 / 96.2 |
| **team cut + stitch_gta** | **89.1** | **22** | 81.5 / 90.7 / 81.0 / 98.7 |
| team cut + perfect connector | 94.7 | 13 | 90.4 / 95.0 / 92.9 / 98.7 |

15–17 cuts in the 4 clips. Same result with 10, 25 or 50 boxes in a row (90.0 / 89.1 / 89.1). Adding "never
merge two teams" to the connector changes nothing, because OSNet already keeps opponents apart.

## Where the team labels come from

The ceiling needs team labels that are right for seconds in a row. Two sources were measured against the
ground-truth teams on the clean boxes of 011b:

| clip | 006 colour classifier | OSNet, tracklets clustered (k = 3) |
|---|---|---|
| NCAA c005 | 74.6 % | 90.4 % |
| Tenerife c002 | 67.1 % | 96.9 % |
| Istanbul c012 | 87.6 % | 96.3 % |
| Changsha c001 | 92.1 % | 99.3 % |

The colour classifier of 006 (3 pixel fractions of the torso: white, black, vivid) was validated on one NCAA clip
(3 % errors). On these arenas it mixes kits. **OSNet** (`--team-source osnet`, default): the embedding of every clean
box one frame in 5; the tracklet means are clustered into 3 groups (two teams and "other", the group with the
fewest boxes, as in 006); each box takes the nearest group. Clustering tracklets, not boxes, gives every
player one vote.

*Caveat.* The label source was chosen after comparing both on these 4 evaluation clips. Nothing numeric was
tuned (1 s minimum run, k = 3 and the 006 rule for "other" are fixed), but the choice itself was made here.
The 15 val clips are the check.

## Results (sandbox, 4 clips, run 011b, OSNet 012 for labels and stitching)

```powershell
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth `
    --split-teams --out outputs/runs/019_team_split
```

| 4 arenas | IDF1 | MOTA | ID switches | team cuts | IDF1 NCAA / Tenerife / Istanbul / Changsha |
|---|---|---|---|---|---|
| 013b (no cut) | 81.3 | 96.1 | **34** | 0 | 77.2 / 73.9 / 72.9 / 95.1 |
| 019, colour labels (006) | 80.5 | 96.0 | 51 | 20 | 73.7 / 70.0 / 75.7 / 95.8 |
| **019, OSNet labels** | **83.3** | 95.9 | 54 | 15 | **83.8** / 73.9 / **75.7** / 95.0 |
| ceiling (ground-truth labels) | 89.1 | 96.4 | 22 | 15–17 | 81.5 / 90.7 / 81.0 / 98.7 |

**Laptop (original frames, `outputs/runs/019_team_split`):** IDF1 **83.5**, MOTA 96.0, 52 ID switches, IDF1 per clip
83.8 / 75.4 / 74.8 / 95.0, 10–14 identities per clip. The sandbox runs on re-encoded frames, which explains the small
differences (Tenerife 73.9 → 75.4, Istanbul 75.7 → 74.8).

## All 15 val clips (laptop)

The 11 other val clips were tracked with the 011b configuration for the first time, then stitched with and
without the team cut. Nothing was designed on these 11.

| 15 clips, OVERALL | IDF1 | MOTA | ID switches |
|---|---|---|---|
| 011b online | 72.7 | 96.7 | 267 |
| **013b: stitching** | **79.9** | **96.7** | **214** |
| 019: team cut + stitching | 79.7 | 96.6 | 289 |

| clip (★ = one of the 4 used so far) | team cuts | IDF1 013b → 019 |
|---|---|---|
| NCAA c001 | 6 | 88.8 → 88.8 |
| NCAA c003 | 0 | 94.5 → 94.5 |
| ★ NCAA c005 | 4 | 77.2 → **83.8** |
| NCAA c007 | 4 | 79.2 → **68.6** |
| NCAA c008 | 8 | 74.6 → **79.3** |
| NCAA c011 | 5 | 70.3 → **64.2** |
| ★ Changsha c001 | 2 | 95.1 → 95.0 |
| Tenerife c001 | 4 | 79.3 → 80.7 |
| ★ Tenerife c002 | 5 | 73.9 → 75.4 |
| Tenerife c003 | 10 | 76.7 → 74.1 |
| Tenerife c004 | 5 | 86.6 → 84.9 |
| Tenerife c005 | 12 | 76.5 → 79.1 |
| Istanbul c008 | 3 | 75.1 → 73.8 |
| ★ Istanbul c012 | 4 | 72.9 → 74.8 |
| Istanbul c014 | 7 | 66.3 → 66.7 |

On the 11 new clips, 019 changes IDF1 by −1.2 on average (4 better, 5 worse, 2 equal). Two clips lose 6 and 11
points. **The 4-clip gain does not hold.** That is what the caveat above warned about: the label source was picked on
those 4 clips. Two facts survive. Wrong cuts can be harmless (NCAA c001: 6 cuts, IDF1 unchanged, the connector
re-joined them). But when a cut piece is re-joined to the wrong player, it costs a lot (NCAA c007).

## Conclusion

**Not adopted.** The diagnosis stands: ids are stolen in crossings, mostly by opponents, and with perfect team labels
the cut would be worth +7.8 IDF1 on the 4 clips. But the unsupervised team labels (OSNet, tracklets clustered) are
not reliable enough across clips: same overall IDF1 as 013b (79.7 vs 79.9) and 75 more ID switches. 013b stays the
offline mode. `--split-teams` stays available, off. Next, if continued: measure team-label quality and the ceiling on
all 15 clips (and choose any labeller on train clips), and check why the pieces are re-joined wrongly in NCAA c007/c011.

## Reading

1. **+2.2 IDF1 over 013b** with OSNet labels (laptop: 81.3 → 83.5), the best result of the project. NCAA gains 6.6 and Istanbul 2.8.
   Tenerife and Changsha do not move.
2. **ID switches go up (34 → 54).** IDF1 counts how long each player keeps the right identity. ID switches count
   events. A cut at the end of a crossing puts the frames *inside* the crossing on the wrong side when the swap
   happened earlier. That makes short wrong stretches, each counted as switches. The ceiling has the same cut
   rule and only 22, so most of the extra switches probably come from the labels: a cut a little off, or a cut that should
   not be there.
3. **Tenerife is where the ceiling gains most (+16.8) and the real version gains nothing.** The cut rule is the
   same, so the difference is in where the labels put the cuts. Comparing them with the ground-truth cuts
   (`outputs/_review/swap/cuts_check.py`, done so far for the colour labels) is the next check.
4. With the colour classifier, the same idea is worse than no cut. The method is only as good as its team labels.

## Next

* ~~Run it on the laptop to confirm~~ (83.5, done). Then all 15 val clips.
* Better team labels, the gap between 83.3 and 89.1, measured on the TRAIN clips (ground-truth teams there), not here.
* Place the cut at the frame where the swap happened inside the crossing, not at its end (fewer ID switches).

Tests: `tests/test_stitch.py` (`split_at_team_changes`).
