# 018 · Part-based ReID: one vector per body stripe

| | |
|---|---|
| **Date** | 2026-10-27 · **negative**: worse on team-mates, opponents and retrieval |
| **Builds on** | 016–017: OSNet separates a player from his team-mates with AUC 0.71, and neither more training people (017) nor association rules (015) move it. Swaps in crossings need > 99 % right decisions; OSNet gives 97 % |
| **Question** | Does keeping the body regions apart (and seeing them in more detail) teach the model the small cues that tell team-mates apart? |

## Idea

OSNet (012) turns the crop into ONE vector by averaging its last feature map over all 16 × 8 positions.
Team-mates share the kit, and the kit fills most of those positions, so small local cues (shoes, socks,
hair, skin, knee pads) are diluted in the average. **PCB** (Part-based Convolutional Baseline, Sun et al.
2018) keeps them apart. The feature map is cut into horizontal stripes, each stripe is averaged on its own, and
during training **each stripe has to recognise the player by itself**. A stripe that does not contain the
jersey cannot rely on it: the bottom stripe has to learn shoes and socks, the top one hair and skin.

`hooptrack.reid.parts.OSNetParts` (documented in the code):

* backbone: OSNet-x1.0 with ImageNet weights, as in 012;
* branches: the whole player (global average, as before) + 4 horizontal stripes;
* BNNeck per branch; one identity classifier per branch; the loss is the mean cross-entropy over branches plus
  the same batch-hard triplet as 012, applied to the final embedding;
* embedding: the 5 branch vectors, each L2-normalised, concatenated and scaled. The cosine similarity of two
  embeddings is then exactly the **mean of the 5 branch similarities**. Every region gets one vote, so the
  jersey gets one vote out of five. The rest of the pipeline uses it unchanged (cosine distance, 2,560-D).

Inference: `reid.type=osnet_parts reid.weights=<file>`. The number of stripes and the input size are stored in
the checkpoint. Tests: `tests/test_reid_parts.py`.

## Two runs, one change each

| run | change | data | compare with |
|---|---|---|---|
| **018a** | stripes (model) | 012 crops, 128 × 256 (`reid_players.zip`, already on Drive) | 012 |
| ~~018b~~ | + larger crops (192 × 384) | `reid_players_384.zip` | 018a. **Dropped before running**, see below |

**Why 018b was dropped.** Larger crops only add detail if the original boxes are larger than the crop.
They are not. Ground-truth player heights in SportsMOT basketball, boxes ≥ 50 px:

| split | p10 / median / p90 height | taller than 256 px | taller than 384 px |
|---|---|---|---|
| train | 66 / 108 / 197 px | 0.6 % | 0 % |
| val | 118 / 150 / 197 px | 0.9 % | 0 % |

Almost every player is already *enlarged* to 256 px. At 384 px he would be enlarged more, with no new
information, at 3× the CPU cost. The only real effect would be a finer feature map (24 × 12 instead of 16 × 8).
If 018a shows that local detail matters, the standard way to get a finer map without upscaling is the
"last stride = 1" trick of ReID models. The `--size` option of `build_reid_dataset.py` stays.

Everything else is the 012 recipe: basketball train (3 games), one clip per batch, 60 epochs, Adam 3e-4, the same
augmentations, no validation and no best-epoch selection. The training data stays basketball because the
multi-sport data of 017 did not help.

Cost on the laptop CPU (measured): 51 ms per crop at 128 × 256, 154 ms at 192 × 384 (OSNet 012: ~56 ms).
That is fine for the offline stitching (~12 crops per tracklet) and the benchmark.

## How to run

**018a**: in [`notebooks/colab_train_reid_osnet.ipynb`](../../notebooks/colab_train_reid_osnet.ipynb), section 2:
`DATASET = "reid_players"`, `ARCH = "osnet_parts"`. Run everything. The cell after section 2 must print
`experiment 018a | model osnet_parts`. Download `runs/reid_osnet_x1_0_parts/osnet_x1_0_parts.pth` to `weights/`
and `eval_018a.json` here.

**018b**: build the larger crops, upload `data/reid_players_384.zip` to `MyDrive/hooptrack/`, then set
`DATASET = "reid_players_384"`, `ARCH = "osnet_parts"`:

```powershell
python scripts/build_reid_dataset.py --size 192 384 --out data/reid_players_384
```

**Benchmark (016)**: on val, then on train to check for memorising:

```powershell
python scripts/reid_teammate_benchmark.py --config configs/player.yaml `
    --model osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth osnet_parts:osnet_parts:weights/osnet_x1_0_parts.pth
```

The pipeline (stitching, 013b) is run only if the benchmark improves the team-mate numbers.

## Results (018a)

Colab T4, 60 epochs, the 012 data (36,311 crops, 150 identities, 128 × 256).

**Retrieval** (`eval_018a.json`, 3,000 queries, the 012 protocol):

| model | rank-1 | mAP |
|---|---|---|
| **OSNet 012 (one vector)** | **0.710** | **0.409** |
| OSNet + 4 stripes (018a) | 0.691 | 0.356 |

**Team-mate benchmark (016)**, all 15 val clips:

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates (crop) | AUC mate (2 s piece) | rank-1 among team-mates (piece) |
|---|---|---|---|---|---|
| **OSNet 012** | **0.713** | **0.839** | 0.757 | **0.756** | **0.737** |
| OSNet + 4 stripes (018a) | 0.668 | 0.739 | **0.761** | 0.701 | 0.729 |

| around crossings | team-mate crossings | AUC | keep < swap | opponent crossings | AUC | keep < swap |
|---|---|---|---|---|---|---|
| **OSNet 012** | 258 | **0.795** | **0.969** | 445 | **0.908** | **0.998** |
| OSNet + 4 stripes (018a) | 258 | 0.720 | 0.957 | 445 | 0.783 | 0.989 |

**Each branch on its own** (4 evaluation clips, `outputs/_review/parts/branches.py`, the embedding cut into its 5 blocks):

| branch | AUC mate | AUC opp |
|---|---|---|
| OSNet 012 (one vector) | **0.705** | **0.852** |
| part model: global | 0.662 | 0.752 |
| stripe 1 (head, shoulders) | 0.665 | 0.745 |
| stripe 2 (torso) | 0.655 | 0.757 |
| stripe 3 (hips, thighs) | 0.665 | 0.769 |
| stripe 4 (legs, feet) | 0.675 | 0.753 |
| all branches (the embedding) | 0.669 | 0.762 |

## Reading

1. **Worse everywhere that matters.** Team-mates AUC 0.713 → 0.668, crossings 96.9 % → 95.7 %. Only the
   rank-1 among team-mates stays level. The expected trade (opponents worse, team-mates better) did not happen:
   both get worse.
2. **The stripes did not specialise.** Every stripe separates team-mates about as badly as the others
   (0.655–0.675). The legs stripe, which should learn shoes and socks, is only 0.01 better than the torso. With
   OSNet, each position of the last feature map already "sees" most of the body (large receptive field), so a
   stripe is not really local. It is a known weakness of PCB on deep, compact backbones.
3. **The shared backbone got worse.** Even the global branch of the part model (0.662) is below OSNet 012 (0.705).
   Asking five classifiers to recognise 150 players from partial views, with only 36k crops, spreads the same
   data over more objectives.
4. **The detail may simply not be in the pixels.** The median player is 150 px tall in val (108 in train), so a
   shoe is a handful of pixels. Three ReID changes in a row (017 data, 018 parts, and the crop size check
   that ruled out 018b) leave team-mate separation at AUC ~0.71. At broadcast resolution, team-mates are
   told apart by little more than what OSNet 012 already uses.

## Conclusion

Negative. The part-based model is worse than OSNet 012 for team-mates, opponents and retrieval. **OSNet 012
stays the pipeline's ReID model.** `reid.type=osnet_parts`, `hooptrack.reid.parts` and the notebook option
stay available (tests: `tests/test_reid_parts.py`). Not tried, because the evidence does not point there:
last stride = 1, more stripes, part-specific losses.
