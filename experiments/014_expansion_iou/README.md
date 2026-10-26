# 014 · Expansion IoU: matching players whose boxes no longer overlap

| | |
|---|---|
| **Date** | 2026-10-26 · **negative result**, EIoU stays off |
| **Builds on** | 011b (best online tracker) + 013b (offline stitching, IDF1 81.3). In the identity-break analysis of 012, 18 breaks were "new id after being lost ≤ 60 frames": the Kalman prediction and the next detection of the same player did not overlap enough |
| **Question** | Does Deep-EIoU's Expansion IoU help our tracker, which keeps its Kalman filter, as it helped ByteTrack (+4.7 HOTA in the Deep-EIoU paper)? |

## Idea

IoU between the predicted box of a track and a new detection is the main motion cue. A basketball
player changes direction and speed within a couple of frames, so the prediction can miss: IoU drops to
0 and the track is lost, even though the detection is right next to it. **Expansion IoU** (Huang et
al., WACV-W 2024) compares the boxes grown by `e` times their size on each side
(`hooptrack.tracking.matching.eiou_matrix`). Close-but-not-overlapping pairs now overlap, far pairs still
do not, and the nearer candidate still scores higher.

Settings, from Deep-EIoU's code (not tuned here):

| | Deep-EIoU | here (`tracker.*`) |
|---|---|---|
| stage 1, high-score boxes | e = 0.7, then the leftovers again with e = 0.8 | `eiou_expand: 0.7`, `eiou_iterations: 2`, `eiou_step: 0.1` |
| low-score boxes, unconfirmed tracks | e = 0.5 | `eiou_expand_low: 0.5` (stages 2 and 3) |
| motion model | no Kalman (last box) | **Kalman kept**: only the IoU changes, one change at a time |

Defaults stay 0 / 1, which gives exactly the previous tracker (checked: identical output file). Tests: `tests/test_eiou.py`.

## How to run

Only the four `eiou` keys change compared with 011b. Detections are cached; ~40 min on the laptop.
Then the same offline stitching as 013b, to compare the full pipeline:

```powershell
python scripts/run_court_source_eval.py --config configs/player.yaml `
    --set tracker.eiou_expand=0.7 tracker.eiou_iterations=2 tracker.eiou_expand_low=0.5 `
    --out outputs/runs/014_eiou --variants segmentation `
    --clips v_00HRwkvvjtQ_c005 v_5ekaksddqrc_c002 v_BgwzTUxJaeU_c012 v_4r8QL_wglzQ_c001
python scripts/stitch_tracklets.py --run outputs/runs/014_eiou/segmentation `
    --config configs/player.yaml --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth `
    --out outputs/runs/014_eiou_stitched
```

## Results

Same run as 011b except the three `eiou` keys (checked: the saved `config.yaml` files differ only there).
Then the same offline stitching as 013b.

| 4 arenas, OVERALL | IDF1 | MOTA | ID switches | FP | Misses | Identities per clip |
|---|---|---|---|---|---|---|
| 011b online | **74.6** | **96.0** | **49** | 397 | 551 | 17–21 |
| 014 online (EIoU) | 68.6 | 95.8 | 81 | 475 | 495 | 15–19 |
| 013b = 011b + stitching | **81.3** | **96.1** | **34** | | | 11–12 |
| 014 + stitching | 72.5 | 95.8 | 76 | | | 11 |

| Clip | IDF1 011b → 014 | ID switches 011b → 014 |
|---|---|---|
| NCAA c005 | 68.9 → 63.6 | 11 → 20 |
| Tenerife c002 | 69.1 → 71.2 | 10 → 11 |
| Istanbul c012 | 69.5 → 59.2 | 20 → 40 |
| Changsha c001 | 85.6 → 76.6 | 8 → 10 |

**What kind of identity errors?** Each ground-truth player is followed through time and every change of
predicted id is classified (same script as the 012 diagnosis, `outputs/_review/eiou/idbreaks014.py`):

| identity break | 011b | 014 (EIoU) |
|---|---|---|
| new / other id while visible (gap ≤ 2 frames): swap or split in view | 30 | **56** |
| back to an id the player had before (the other half of a swap) | 31 | **57** |
| new id after being lost ≤ 60 frames, **the case EIoU targets** | 18 | 20 |
| new id after being lost > 60 frames | 13 | 11 |

**Reading.**

1. **EIoU did not fix what it was meant to fix.** Short losses stay at 18–20. Those players are not
   lost because the prediction lands *next to* the detection. Most often the detection itself is missing
   for some frames (occlusion), and the Kalman prediction already bridges small offsets.
2. **It almost doubles the swaps inside crossings.** In basketball, team-mates and opponents stand
   less than a body width apart. With e = 0.7 a box grows to 2.4× its width, so neighbours overlap
   above our IoU gate and become valid candidates. Motion no longer separates them, and appearance has
   to decide on overlapping crops, where our tracker uses little appearance (012). The swaps also
   bring more false positives (397 → 475) and fewer misses: fewer tracks die, but some survive on the
   wrong player.
3. **Why it works in Deep-EIoU and not here.** Deep-EIoU drops Kalman and puts a strong sports-trained
   ReID in the cost of the *same* stage, with its own gates. Its expansion compensates for having no motion
   model. On top of Kalman, with our gates, the extra reach only adds wrong candidates. Its table 5
   (ByteTrack + EIoU) is on all SportsMOT sports; basketball is the most crowded one.
4. **Stitching cannot repair it**: a swap leaves tracklets that mix two players (72.5 vs 81.3 IDF1).

**Conclusion.** Negative result. Expansion IoU on our Kalman tracker hurts basketball tracking
(IDF1 −6.0 online, −8.8 after stitching). The switches stay in the code, off by default. Not tuned
further: lowering `e` on the validation clips would be tuning on our test set. The evidence points
the same way as 012 and 013: the remaining errors are **swaps while players are visible and close**,
so the next useful step is an appearance check when two tracks separate after a crossing, not a looser
motion match.
