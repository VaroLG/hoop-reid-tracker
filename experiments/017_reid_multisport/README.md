# 017 · ReID trained on more varied people (basketball + football + volleyball)

| | |
|---|---|
| **Date** | 2026-10-27 · neutral: better retrieval, same team-mate separation, stitching 80.2 vs 81.3 |
| **Builds on** | 012 (OSNet fine-tuned on 3 basketball games) and 016 (team-mate benchmark: AUC mate 0.706, keep < swap at 97 % of team-mate crossings, and that is still not enough) |
| **Question** | With the same recipe, does training on more varied people teach OSNet to tell team-mates apart better in games it has never seen? |

> **Data and licences.** Training data: basketball, football and volleyball crops of SportsMOT (CC BY-NC 4.0); starting model: OSNet (torchreid, MIT); weights not redistributed. All sources: [`docs/data_models_licences.md`](../../docs/data_models_licences.md).

## Idea

Three basketball games are about 30 different people. A model can learn *those* players instead of learning
*how to tell players apart*. The 012 sampler already makes team-mates the hardest negatives: each batch holds
the players of one clip, so changing the sampling again would teach nothing new. What can change is the
**variety**. Most cues that separate team-mates (shoes, hair, skin, build, how a kit is worn) do not depend on
the sport. The SportsMOT **train** clips of football and volleyball add many more people wearing identical kits.

**One change:** the training crops. Model, losses (identity + batch-hard triplet), P×K sampler (one clip per
batch), 60 epochs, Adam 3e-4 and the augmentations are all identical to 012. The test stays basketball val (the 4
unseen games).

**Judged by** the team-mate benchmark (016), on val, against the 012 model. `--split train` gives the
same numbers on basketball train. A model that improves on train but not on val is memorising, not
generalising. The tracker is run only if the benchmark improves.

## How to run

1. Download the football and volleyball **train** clips (the train archive is ~6.7 GB and is deleted afterwards;
   only those sequences are kept, basketball is not touched):
   ```powershell
   python scripts/download_data.py --splits train --sport football volleyball
   ```
2. Build the crops and the zip, then upload `data/reid_players_multisport.zip` to `MyDrive/hooptrack/`:
   ```powershell
   python scripts/build_reid_dataset.py --train-sports basketball football volleyball --out data/reid_players_multisport
   ```
3. In [`notebooks/colab_train_reid_osnet.ipynb`](../../notebooks/colab_train_reid_osnet.ipynb), set
   `DATASET = "reid_players_multisport"` (section 2) and run everything. Each epoch takes about 3× longer than in 012, and
   the run resumes from Drive if Colab disconnects.
4. Put `osnet_x1_0_multisport.pth` in `weights/` and run the benchmark on both models:
   ```powershell
   python scripts/reid_teammate_benchmark.py --config configs/player.yaml `
       --model osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth osnet_multisport:osnet:weights/osnet_x1_0_multisport.pth
   python scripts/reid_teammate_benchmark.py --config configs/player.yaml --split train `
       --model osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth osnet_multisport:osnet:weights/osnet_x1_0_multisport.pth
   ```

## Results

Training data: 102,063 crops, **639 identities**, 45 clips (012: 36,311 crops, 150 identities, 15 clips).
Same test: 23,797 crops of the 15 basketball val clips. Colab T4, 60 epochs.

**Retrieval among all the players of a clip** (`eval_017.json`, 3,000 queries, the 012 protocol):

| model | rank-1 | mAP |
|---|---|---|
| OSNet 012 (basketball) | 0.710 | 0.409 |
| **OSNet 017 (multi-sport)** | **0.745** | **0.448** |

**Team-mate benchmark (016), all 15 val clips:**

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates (crop) | AUC mate (2 s piece) | rank-1 among team-mates (piece) |
|---|---|---|---|---|---|
| ResNet-50 ImageNet (the online tracker's ReID) | 0.558 | 0.674 | 0.492 | 0.568 | 0.497 |
| colour histogram | 0.574 | 0.841 | 0.483 | 0.595 | 0.447 |
| OSNet 012 (basketball) | 0.713 | 0.839 | 0.757 | **0.756** | 0.737 |
| **OSNet 017 (multi-sport)** | **0.715** | **0.851** | **0.791** | 0.737 | **0.799** |

| around crossings | team-mate crossings | AUC | keep < swap | opponent crossings | AUC | keep < swap |
|---|---|---|---|---|---|---|
| ResNet-50 ImageNet | 258 | 0.600 | 0.767 | 445 | 0.747 | 0.966 |
| colour histogram | 258 | 0.624 | 0.748 | 445 | 0.909 | 0.991 |
| OSNet 012 | 258 | 0.795 | **0.969** | 445 | 0.908 | 0.998 |
| **OSNet 017** | 258 | **0.810** | 0.965 | 445 | **0.918** | **1.000** |

Per clip, `AUC mate (crop)` 012 → 017: Istanbul 0.72–0.75 → 0.78–0.80 and Changsha 0.71 → 0.78 improve.
NCAA is mixed (0.67–0.81 both). Tenerife drops slightly (0.66–0.71 → 0.64–0.71). Mean over clips 0.726 → 0.730.

## Reading

1. **More varied people improve ranking, not separation.** The nearest crop is more often the right player:
   rank-1 +3.5 among all players, +3.4 among team-mates, +6.2 at piece level. But the probability that a
   same-player pair is closer than a team-mate pair (AUC) does not move: 0.713 → 0.715.
2. **The generalisation worry was justified, both ways.** The extra sports do not make the model memorise:
   the unseen games improve on average. The gain depends on the arena, though, and Tenerife gets slightly worse.
3. **Crossings stay at ~97 %.** No model reaches the > 99 % that a swap check would need (016). The
   data variety alone does not close the team-mate gap.
4. **Side finding: the online tracker's ReID is the weak link.** ResNet-50 ImageNet takes the right decision at only
   77 % of team-mate crossings (OSNet: 97 %). It is barely better than a colour histogram at telling
   team-mates apart (AUC 0.558 vs 0.574). In 012, OSNet did not help online because the tracker uses little appearance in
   overlaps. The model itself is clearly better.

## Conclusion

### In the pipeline: offline stitching (013b) with each model

Same 011b run, same `stitch_gta` settings (merge distance 0.4, spatial factor 1.0, 12 clean crops per tracklet).
Only the ReID model used to describe the tracklets changes.

| 4 arenas | IDF1 | MOTA | ID switches | IDF1 NCAA / Tenerife / Istanbul / Changsha |
|---|---|---|---|---|
| 011b online | 74.6 | 96.0 | 49 | 68.9 / 69.1 / 69.5 / 85.6 |
| **013b: stitching with OSNet 012** | **81.3** | **96.1** | **34** | **77.2 / 73.9** / 72.9 / 95.1 |
| 017: stitching with OSNet 017 | 80.2 | 96.1 | 36 | 74.8 / 70.0 / **74.2** / 95.1 |
| 012 online (OSNet in the tracker) + stitching with OSNet 012 | 80.8 | 96.1 | **32** | **77.2** / 73.2 / **75.7** / 92.1 |

The last row puts OSNet in the online tracker too (run 012) before stitching: fewer ID switches (32) and
better Istanbul, but Changsha loses 3 points, so it ties with 013b overall. Same pattern as the benchmark: Istanbul improves, Tenerife and NCAA get worse, Changsha is unchanged. With
only 6–10 merges per clip, one different merge moves a clip's IDF1 by several points, so a 1-point difference
overall is within what a couple of decisions can change.

## Conclusion

**Neutral to slightly negative in the pipeline.** OSNet 017 retrieves better (rank-1 +3.5, team-mate rank-1 +3.4)
but it does not separate team-mates better (AUC 0.713 → 0.715, crossings ~97 %), and in the offline stitching it
gives 80.2 IDF1 against 81.3 with OSNet 012. **OSNet 012 stays the pipeline's ReID model.** The weights
`osnet_x1_0_multisport.pth` and the data options (`--train-sports`, `DATASET` in the notebook) stay available.

What this rules out: the team-mate problem is not mainly a lack of training people. With 4× more identities,
the separation between team-mates is unchanged.
