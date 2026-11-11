# 038 · Closed set of 10 players + the online tracker's continuity

> **In the final pipeline:** the identities: pieces cut at crossings, exactly 10 players, online continuity 0.3. See [the final selection](../../README.md#the-final-selection).

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [037](../037_closed_set_of_ten/) closed set of 10 (better on NCAA, worse on FIBA) · [032](../032_crossing_check_diagnosis/) the tracker keeps the right identity through ~96 % of the crossings |
| **Question** | 037b decides every piece by appearance alone and throws away the online tracker's continuity. Does adding it back keep the NCAA gains and remove the FIBA losses? |
| **Clips** | all 15 validation clips |

## Why not a per-clip selector

The idea after 037 was a criterion, calibrated on train, saying per clip when to use the closed set. It cannot be
calibrated fairly: the detector (011) and the SigLIP ReID (028) were both trained on the train clips, where the
appearance looks almost perfect (median distance same player 0.17 vs team-mate 0.97; on val 0.40 vs 0.74). A
criterion learned there would pick the closed set everywhere; one learned on val would be tuned on the reported
clips. So the two methods are combined instead of chosen.

## Change (one switch, off by default)

`--continuity` in [`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py) (with `--closed-set 10`),
`assign_closed_set(links=..., continuity=...)` in [`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py):
every piece knows the pieces just before and after it inside the same **online** tracklet (cut at a crossing). If
one of them already belongs to a player, the piece's distance to that player is lowered by `continuity`: the piece
stays with the online identity unless its appearance prefers another player by more than that. **0.3, fixed before
running** (about half the gap between same-player and other-player distances); no other value was tried.
`--continuity 0` is exactly 037b. Same pieces and embeddings as 037b.

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | switches / min |
|---|---|---|---|---|---|---|---|
| 029 (adopted until now) | 76.9 | 85.7 | 69.1 | 83.9 | 96.6 | **278** | **31.3** |
| 037b closed set | 74.5 | 84.8 | 65.4 | 81.7 | 96.4 | 612 | 71.6 |
| **038 closed set + continuity** | **79.7** | 84.9 | **74.9** | **89.6** | 96.6 | 379 | 43.7 |

HOTA per clip, 029 → 038 ([`hota.csv`](hota.csv)): **better on 11 of 15 clips**, equal on 1, worse on 3.

| | c001 | c003 | c005 | c007 | c008 | c011 |
|---|---|---|---|---|---|---|
| NCAA | 85.0 → 86.4 | 90.0 → 90.0 | 72.1 → **83.3** | 62.7 → **73.4** | 71.5 → **79.4** | 67.2 → 66.9 |

| | Changsha c001 | c001 | c002 | c003 | c004 | c005 | 2010 c008 | c012 | c014 |
|---|---|---|---|---|---|---|---|---|---|
| FIBA | 87.5 → 88.3 | 75.1 → 76.6 | 74.3 → **79.6** | 78.9 → 77.0 | 79.7 → **84.4** | 76.6 → **80.0** | 66.2 → 65.0 | 70.1 → **73.8** | 70.7 → **73.2** |

Ids per clip: 11–26 (10 players + the pieces that fit nobody: on screen with all 10, mostly referees and false
detections; 1–16 per clip).

## Reading

1. **The best result of the project, on most clips**: HOTA +2.8, IDF1 +5.7 over 029, with the gain on 11 of 15
   clips and in all four games. The two pieces of information are complementary: continuity keeps the identity
   where the tracker was right, appearance plus the 10-player rule corrects it where it was not.
2. **More switches, but shorter errors.** The switch count goes up (278 → 379 IDSW; per kind 029 → 038: new 52 → 96,
   team-mate 34 → 64, opponent 176 → 206), while IDF1 (share of boxes with the right identity) rises a lot. The
   likely reading: 029's errors last (a swapped identity stays swapped until the end), 038's errors are short pieces
   given to the wrong player and the identity comes back after them. Each such excursion costs two switches but
   few boxes.
3. 037b's large NCAA gains are partly kept (c005, c007, c008), c011 loses its gain: there the online continuity was
   wrong and now wins over appearance.

## Conclusion

**Positive: adopted as the best offline mode** (HOTA 79.7, IDF1 89.6). Kept behind `--closed-set 10 --continuity
0.3`, so 029 is still reproducible. Open: the pieces that fit nobody (referees, false detections) keep their own
id; and the short wrong excursions could be smoothed (a piece between two pieces of the same player).

## Commands

```powershell
# same embeddings as 037b (copied first)
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --split-crossings --closed-set 10 --continuity 0.3 --out outputs/runs/038_continuity
python scripts/eval_hota.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft 038=outputs/runs/038_continuity
python scripts/switch_report.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft 038=outputs/runs/038_continuity
```
