# 020 · Team labels that hold in new arenas: SigLIP on the torso, two clusters

| | |
|---|---|
| **Date** | 2026-10-28 · **positive**: all 15 val clips IDF1 79.9 → **81.8**, at the ground-truth-label ceiling of this sampling (82.0) |
| **Builds on** | 019: cutting tracklets where their team changes works with ground-truth teams, but with OSNet labels it gives nothing on the 15 val clips (79.9 → 79.7). The label source had been chosen on the 4 evaluation clips |
| **Question** | Which unsupervised team labeller is reliable across arenas, when it is **chosen on the train clips** and checked once on val? |

## Protocol: choose on train, report on val

019's caveat was that its labeller was picked on the clips used to report it. Here nothing is decided on val.

* **Choice: train split** (15 clips, 3 games). `scripts/team_label_benchmark.py` labels the clean ground-truth boxes
  (one frame in 5, as 019) with each candidate and measures three things against the reference teams
  (`hooptrack.eval.gt_teams`; sanity: no frame has more than 5 players in one team on any clip):
  * **accuracy** of the labelled boxes;
  * **false cuts**: 019's cut rule applied to the ground-truth tracks. A track is one player, so every cut is wrong;
  * **swap recall**: for every crossing between opponents (857 on train), a fake tracklet follows player A for 5 s,
    then player B for 5 s, which is exactly an id stolen in a crossing. Recall = share of them the cut rule finds.
    Without this, a labeller that never cuts would look perfect.
* **Check: val split** (15 clips, 4 other games). The chosen labeller runs once in the pipeline (011b + team cut +
  `stitch_gta`) next to 013b and 019. An oracle (`--team-source gt`: each clean box takes its ground-truth player's team)
  gives the ceiling of this sampling.

**One trap avoided.** OSNet 012 was trained on the train players. On train it clusters *players*, not teams
(accuracy 0.58, near chance), so its train numbers say nothing about val. It was kept as a reference only.

## Candidates

The same recipe for all (`hooptrack.tracking.team_labels.cluster_team_labels`): the mean vector of each tracklet votes once,
k-means on the tracklet means, and each box takes the nearest centre. Only the box descriptor and k change.

| descriptor | what it looks at | trained on our data? |
|---|---|---|
| `lab` | median Lab colour of the torso (the same colour the reference teams use, so its accuracy is flattering) | no |
| `colour006` | white / black / vivid pixel fractions of the torso (006) | no |
| `osnet` | OSNet 012 on the whole box (019) | **yes**, the train players |
| `osnet_torso` | OSNet 012 on the torso only | **yes** |
| `siglip_torso` | SigLIP base (image tower, web-scale pre-training) on the torso only | no |

Options tried: k = 3 (019: two teams + "other") or 2, rejecting ambiguous boxes (distance ratio 0.7 / 0.8), and UMAP to 3-D
before k-means (the "SigLIP + UMAP + K-means" team-clustering recipe).

## Results on train (where the choice is made)

All 15 train clips. No rejection, no UMAP ([`train_summary.csv`](train_summary.csv) for every variant):

| descriptor | k | accuracy | worst clip | boxes put in "other" | false cuts | swap recall |
|---|---|---|---|---|---|---|
| colour006 | 3 | 0.700 | 0.568 | 22 % | 174 | 0.625 |
| colour006 | 2 | 0.887 | 0.758 | – | 63 | 0.804 |
| lab | 2 | 0.895 | 0.830 | – | 54 | 0.768 |
| osnet (019) | 3 | 0.561 | 0.469 | 9 % | 1 | 0.425 |
| osnet_torso | 2 | 0.893 | 0.762 | – | 18 | 0.700 |
| siglip_torso | 3 | 0.792 | 0.706 | 19 % | 71 | 0.751 |
| **siglip_torso** | **2** | **0.975** | **0.959** | – | **2** | **0.951** |

Rejection lowers the false cuts of the colour labellers but loses as many true swaps. UMAP never helps SigLIP (k = 2:
recall 0.951 → 0.931, false cuts 2 → 13). **Chosen: SigLIP on the torso, k = 2, no rejection, no UMAP.** It is the best on every
measure, on every clip (accuracy 0.959–1.000 per clip; 2 false cuts in total, both in one clip).

## Results on val (checked once)

Same 011b run, same stitching (`stitch_gta`, OSNet 012, 12 crops per tracklet). Only the team labels change.
Sandbox, original frames: 013b and 019 reproduce the laptop numbers exactly (79.9 and 79.7).

| 15 val clips | IDF1 | MOTA | ID switches | team cuts |
|---|---|---|---|---|
| 011b online | 72.7 | 96.7 | 267 | – |
| 013b: stitching | 79.9 | 96.7 | **214** | 0 |
| 019: team cut, OSNet k = 3 | 79.7 | 96.6 | 289 | – |
| 020: team cut, OSNet k = 2 | 79.0 | 96.6 | 277 | 60 |
| **020: team cut, SigLIP torso k = 2** | **81.8** | 96.6 | 267 | 47 |
| ceiling: team cut, ground-truth teams (same boxes) | 82.0 | 96.6 | 276 | 53 |

| clip (★ = one of the 4 used before) | 013b | 019 | **020 SigLIP** | ceiling |
|---|---|---|---|---|
| NCAA c001 | 88.8 | 88.8 | **92.7** | 92.7 |
| NCAA c003 | 94.5 | 94.5 | 94.5 | 94.5 |
| ★ NCAA c005 | 77.2 | **83.8** | 77.2 | 78.8 |
| NCAA c007 | **79.2** | 68.6 | 73.9 | 73.9 |
| NCAA c008 | 74.6 | **79.3** | 74.6 | 78.2 |
| NCAA c011 | **70.3** | 64.2 | 67.2 | 67.2 |
| ★ Changsha c001 | **95.1** | 95.0 | 95.0 | 95.0 |
| Tenerife c001 | 79.3 | 80.7 | 80.7 | 80.6 |
| ★ Tenerife c002 | 73.9 | 75.4 | 75.5 | 75.5 |
| Tenerife c003 | 76.7 | 74.1 | **84.3** | 81.1 |
| Tenerife c004 | **86.6** | 84.9 | 85.5 | 90.1 |
| Tenerife c005 | 76.5 | 79.1 | **82.9** | 82.9 |
| Istanbul c008 | **75.1** | 73.8 | 73.8 | 73.8 |
| ★ Istanbul c012 | 72.9 | 74.8 | **77.2** | 77.2 |
| Istanbul c014 | 66.3 | 66.7 | **71.8** | 70.1 |

Against 013b: 7 clips better, 5 worse, 3 equal. On the 4 evaluation clips: 81.3 (013b) → 82.6 (019 there: 83.5).

**Cost**: SigLIP runs only offline, on the clean boxes of one frame in 5 (13,051 boxes over the 15 clips): 0.18 s per box on
2 sandbox CPU cores, about 2–3 minutes per clip. The online tracker does not change.

## Reading

1. **The labels were the problem, and they are now as good as the oracle.** With the same boxes, SigLIP gives 81.8 and
   ground-truth teams 82.0. The 0.2 left is what better labels could still give with this sampling.
2. **k = 3 was a mistake once the detector saw only players (011).** The third cluster no longer finds referees: it
   takes 16–22 % of real players, whose tracks then "change team" for no reason. With k = 2 every descriptor improves on train.
3. **SigLIP describes the kit, OSNet describes the player.** OSNet was trained to tell players apart. Its tracklet means
   cluster by person, worst of all on the players it was trained on. SigLIP was never fitted to our clips and describes the kit
   only (the torso crop hides the face, legs and background). That is also why it should transfer to new arenas: nothing in it
   comes from the 3 training games.
4. **The losses now come from the cut, not the labels.** NCAA c007 and c011 lose 5.3 and 3.1 points *even with ground-truth teams*.
   SigLIP and the oracle produce exactly the same output files on both clips (c011: 6 cuts vs 4, the 2 extra ones re-joined),
   so the cuts are the right ones. What goes wrong is after the cut: some pieces are re-joined to the wrong player, or the cut
   lands a few frames off and leaves short wrong stretches (ID switches 214 → 267 overall).
5. **The 019 ceiling (89.1 on 4 clips) used denser labels**: every matched box of every frame, not clean boxes one frame in 5.
   The same 4 clips give 83.0 with the oracle on 020's sampling. Where the cut is placed, and how the pieces are re-joined,
   is now the limit.

## Conclusion

**Adopted as the offline mode**: team cut with SigLIP torso labels, k = 2, then `stitch_gta`. All 15 val clips, IDF1
79.9 → 81.8, chosen on train and never tuned on val. `--split-teams` stays off by default in the script; the switches below select 020.

## How to run

```powershell
pip install transformers==5.21.0
hf download google/siglip-base-patch16-224 --local-dir weights/siglip-base-patch16-224

# choice on train (features cached in outputs/team_bench/train/<clip>/)
python scripts/team_label_benchmark.py --split train --sources lab colour006 osnet osnet_torso siglip_torso

# the pipeline (011b run, 15 val clips)
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth `
    --split-teams --team-source siglip --team-k 2 --out outputs/runs/020_siglip_k2_15

# ceiling of this sampling (oracle labels)
python scripts/stitch_tracklets.py ... --split-teams --team-source gt --out outputs/runs/020_gt_15
```

`--team-umap 3` needs `pip install umap-learn` (not adopted).

Code: `hooptrack.tracking.team_labels` (recipe, documented), `hooptrack.reid.siglip` (SigLIP crops), new `--team-source`
options in `scripts/stitch_tracklets.py` (019's default path is unchanged). Tests: `tests/test_team_labels.py`.

## Next

* **Place the cut where the swap happened**, not at the end of the crossing, and label more boxes near crossings (the gap to the
  denser oracle).
* **NCAA c007 / c011**: follow each cut piece and see why it is re-joined to the wrong player (same failure with oracle labels).
* SigLIP on new, non-SportsMOT clips (no ground truth) to check that the kit clustering holds in other broadcasts.
