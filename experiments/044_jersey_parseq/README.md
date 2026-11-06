# 044 · Jersey numbers with a pose-guided torso crop and a pretrained text reader (PARSeq, zero-shot)

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [033](../033_jersey_number_detector/)/[041](../041_jersey_reader_sportsmot/): whole-box classifiers, ~30 % of pieces numbered, contradictory · [042](../042_soft_team_cost/): 37 % of 038's errors are team-mates, where appearance cannot help |
| **Question** | The published recipe (Koshkina & Elder 2024; SoccerNet 2025 entries): torso crop from pose keypoints + PARSeq + vote over the crops. Zero-shot, does it read our numbers well enough to pass 041's gate? |
| **Decided on** | the 3 held-out **train** clips, 30 players with a typed number (`data/jersey_labels/train_numbers.csv`), 12 tallest clean crops each. Criterion fixed before: most players numbered with accuracy ≥ 90 %. |

## Method

[`scripts/jersey_parseq.py`](../../scripts/jersey_parseq.py): YOLO11n-pose on each crop → box between shoulders and
hips (fixed window if no keypoints) → PARSeq (torch.hub, pretrained) → reads of 1–2 digits → vote per player.

## Results (train, 30 players)

| variant | best setting with accuracy ≥ 90 % | players numbered |
|---|---|---|
| 044a vote = Σ confidence / crops ([`train_grid_044a.csv`](train_grid_044a.csv)) | min score 0.4 | 3 (10 %) |
| 044b + only digits may be read ([`train_grid_044b.csv`](train_grid_044b.csv)) | min score 0.4 | 4 (13 %) |
| 044c vote over confident reads only (conf ≥ 0.8, ≥ 2 such reads; same cached reads as 044a) | | **11 (37 %)** at 91 % |

The torso crop works: keypoints found on 88–93 % of the crops, and when the number faces the camera it fills the
crop ([`assets/torsos_pose.png`](assets/torsos_pose.png)). Per player, the right number usually appears in a few crops
with confidence 0.93–1.00 (35, 44, 11, 30, 23…); the wrong reads are mostly a mid-confidence "1" on side views
(stripes, arms) and team names above the number. Summing confidences lets the many weak wrong reads win, hence 044c.
Forcing digits (044b) turns letters into confident wrong digits.

## Reading

1. **Zero-shot, about a third of the players get a reliable number over a whole clip** (44c, 37 % at 91 %). 044c's
   rule was chosen on train after looking at the reads; it is still a train decision.
2. **The gate is per piece, and 038's pieces are short** (most errors are on 1–3 s pieces, with 1–4 clean crops):
   at the piece level coverage can only be lower than 37 %, far from the 50 % the gate asks. **Not run on val**: the
   gate cannot be passed and the test clips stay untouched for a better reader.
3. What a better reader would need is what Koshkina & Elder did: fine-tune PARSeq on in-domain numbers (from many
   games: 041 showed that 3 games teach the players, not the digits) plus a legibility classifier.

## Conclusion

**Negative (zero-shot)**, val not used. Kept: the script (reads cached), the switch `--digits-only`, the crops.

## Commands

```powershell
pip install timm pytorch-lightning nltk
python scripts/jersey_parseq.py --split train --players gt --out outputs/jersey/044_train
python scripts/jersey_parseq.py --split train --players gt --digits-only --save-crops 60 --out outputs/jersey/044b_train
```
