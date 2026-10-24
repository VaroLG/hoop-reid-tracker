# 012 · A ReID model that tells team-mates apart (OSNet fine-tuned on SportsMOT)

| | |
|---|---|
| **Date** | 2026-10-24 (in progress) |
| **Builds on** | [011b](../011_player_detector/): players-only detector + C06 court mask, IDF1 74.6, MOTA 96.0 on 4 validation arenas. The detections are now right, but identities are not: 17–21 identities for 10 players, and swaps in crossings |
| **Question** | Does an appearance model trained on players (instead of ImageNet ResNet-50) keep identities through crossings and re-entries? |

## Idea

The tracker's appearance vector comes from ResNet-50 trained on ImageNet: for it, two
team-mates are "a person in a white shirt". Deep-EIoU, the reference tracker on SportsMOT,
uses OSNet fine-tuned on SportsMOT player crops (+3.9 HOTA in its ablation). OSNet is
built for re-identification (features at several scales at once) and is 11× smaller than
ResNet-50, so it is also faster on CPU.

Training uses an identity loss (cross-entropy over the training players) plus a batch-hard
triplet loss. Batches are 16 players × 4 crops **from the same clip**, so the hardest
negatives are team-mates.

## Data and evaluation

`scripts/build_reid_dataset.py` → `data/reid_players.zip`, crops resized to 128×256:

| split | clips | frames | role |
|---|---|---|---|
| train | 15 SportsMOT train clips (3 games) | 1 in 3 | learning |
| test | 15 SportsMOT val clips (4 other games) | 1 in 5 | final numbers only |

There is no validation split and no best-epoch selection. The recipe (60 epochs, Adam 3e-4,
step decay) is fixed in advance from Deep-EIoU, so nothing is tuned on the test arenas.

Test protocol = the tracker's situation: a query crop is searched among the crops of **the
same clip** (team-mates included) taken **≥ 2 s away**. Adjacent frames would make the task
trivial, which is the weakness of the older `scripts/export_reid_crops.py` split. Reported:
rank-1, mAP, and the distance distributions (same player vs other player). The tracker's
thresholds (`appearance_gate`, `memory_thresh`) were set for ResNet-50 and are recalibrated
from these distributions.

## How to run

1. `python scripts/build_reid_dataset.py`, then upload `data/reid_players.zip` to `MyDrive/hooptrack/`.
2. Run [`notebooks/colab_train_reid_osnet.ipynb`](../../notebooks/colab_train_reid_osnet.ipynb) in Colab (T4, ~1 h, resumable).
3. Put `osnet_x1_0_sportsmot.pth` in `weights/`, run `pip install torchreid gdown`, then measure it in the tracker with the 011b configuration, changing only the ReID model:

```powershell
python scripts/run_court_source_eval.py --config configs/player.yaml `
    --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth `
    --out outputs/runs/012_reid_osnet --variants segmentation `
    --clips v_00HRwkvvjtQ_c005 v_5ekaksddqrc_c002 v_BgwzTUxJaeU_c012 v_4r8QL_wglzQ_c001
```

## Results

Pending.
