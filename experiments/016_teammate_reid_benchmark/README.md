# 016 · Team-mate ReID benchmark: can the appearance model tell team-mates apart?

| | |
|---|---|
| **Date** | 2026-10-27 · 4 evaluation clips (sandbox) and all 15 val clips (laptop) |
| **Builds on** | 012 (OSNet fine-tuned, rank-1 0.71 among all players) and 015 (swaps in crossings, appearance looked like a coin on the tracker's pieces) |
| **Question** | How well does each ReID model separate a player from his **team-mates**, at random moments and right around a crossing? |

## Why a benchmark first

Every idea to fix the swaps (retrain ReID with team-mate negatives, part-based models, per-game adaptation,
jersey numbers) changes the appearance model. Running the tracker takes ~40 minutes per configuration. This
benchmark takes minutes and measures the model alone: crops come from the **ground truth**, so detector and
tracker errors do not enter.

## Protocol

`scripts/reid_teammate_benchmark.py`, metrics in `hooptrack.eval.reid_bench` (tests: `tests/test_reid_bench.py`).

* **Crops**: one frame in 5, ground-truth boxes ≥ 50 px tall, **clean only** (no other box covering 20 % or more).
* **Teams**: jersey-colour clustering of the ground-truth players (`hooptrack.eval.gt_teams`, as in 004).
* **Random pairs** (≤ 60 crops per player, spread over the clip, pairs ≥ 2 s apart):
  * `AUC mate`: probability that two crops of the same player are closer than two crops of team-mates.
    0.5 = coin, 1.0 = perfect. **The number to improve.**
  * `AUC opp`: the same against opponents (other kit: should be easy).
  * `rank-1 among team-mates`: the nearest crop of the query's own team (~5 players) is the same player. Chance ≈ 0.2.
  * The same at **piece** level: 2-second pieces of a player, mean distance over all crop pairs (what `stitch_gta` compares).
* **Crossings** (two ground-truth players overlapping ≥ 20 %): clean crops in the 2 s before and the 2 s after.
  `AUC`: each player after vs himself before (positive) and vs the other player before (negative).
  `keep < swap`: share of crossings where "nobody swapped" is the closer explanation, i.e. where a swap check would be right.

Split **val** = the 4 unseen games where tracking is measured. Models are compared here, nothing is tuned here.

## Results (4 evaluation clips, sandbox)

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates (crop) | AUC mate (2 s piece) | rank-1 among team-mates (piece) |
|---|---|---|---|---|---|
| colour histogram | 0.578 | 0.837 | 0.467 | 0.598 | 0.422 |
| **OSNet fine-tuned (012)** | **0.706** | **0.852** | **0.740** | **0.747** | **0.707** |

Around crossings:

| model | team-mate crossings | AUC | keep < swap | opponent crossings | AUC | keep < swap |
|---|---|---|---|---|---|---|
| colour histogram | 69 | 0.629 | 0.783 | 115 | 0.864 | 0.991 |
| **OSNet fine-tuned** | 69 | **0.785** | **0.971** | 115 | **0.923** | **0.991** |

(ResNet-50 ImageNet, the tracker's ReID, could not be measured here: its weights cannot be downloaded in the
sandbox. Command below.)

## All 15 val clips (laptop)

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates (crop) | team-mate crossings: AUC | keep < swap |
|---|---|---|---|---|---|
| ResNet-50 ImageNet (the online tracker's ReID) | 0.558 | 0.674 | 0.492 | 0.600 | 0.767 |
| colour histogram | 0.574 | 0.841 | 0.483 | 0.624 | 0.748 |
| OSNet fine-tuned (012) | 0.713 | 0.839 | 0.757 | 0.795 | 0.969 |

258 team-mate and 445 opponent crossings. The 4-clip numbers hold on all 15 clips. The ReID model the online
tracker uses (ResNet) is barely better than colour at telling team-mates apart. Newer models are compared in
[017](../017_reid_multisport/).

## Reading

1. **Team-mates are the hard case**, as expected: with OSNet, AUC 0.71 against team-mates vs 0.85 against opponents.
   Among 5 team-mates the right player is the nearest one 74 % of the time (chance 20 %).
2. **With clean ground-truth crops, OSNet takes the right decision at 97 % of team-mate crossings.** This
   corrects the reading of 015 (AUC 0.55 on the tracker's pieces). The model is not at chance. What the tracker
   gives the swap check is worse than what the model can use: pieces that already mix two players, few clean
   crops, a long and impure history.
3. **Why 97 % is still not enough: the base rate.** In 015, only 21 of 493 crossings ended with a different
   player (4 %). A check that is wrong 3 % of the time on the ~470 crossings with no swap makes ~14 false
   swaps, about as many as the real ones it can fix. That is what 015b showed (1 of 8 swaps right). A swap check pays
   off only when false swaps become much rarer than real ones (accuracy on non-swaps above ~99 %),
   either with a better model or with very confident decisions only.

## How to run (laptop, all 15 val clips)

```powershell
python scripts/reid_teammate_benchmark.py --config configs/player.yaml `
    --model resnet50_imagenet:resnet osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth histogram:histogram
```

Embeddings are cached in `outputs/reid_bench/<model>/`. Adding a new model later only computes that model.
`--split train` gives the same numbers on the games OSNet was trained on (optimistic, for reference only).
