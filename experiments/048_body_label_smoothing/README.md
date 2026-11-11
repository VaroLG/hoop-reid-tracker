# 048 · Keep a body's id through a short excursion (P → Q → P on one body shows P)

> **In the final pipeline:** the post-process applied to 038. See [the final selection](../../README.md#the-final-selection).

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [038](../038_closed_set_continuity/) (adopted) · [047](../047_swapped_excursions/): the excursions are not in the online tracklets |
| **Question** | Working on the final output only (no ground truth): when the boxes of one body go id P → Q → P, can the Q stretch be shown as P, so the video and the metrics both lose the excursion? |
| **Clips** | 15 val clips; `--max-seconds 2` fixed before running from 047's measurement (made on val, said here) |

## Change

[`scripts/smooth_body_labels.py`](../../scripts/smooth_body_labels.py), a post-process of any run folder:

1. **bodies**: boxes linked one to one over time when they overlap a lot (IoU ≥ 0.5 with the body's last box seen in
   the last 5 frames), whatever their ids;
2. along a body, a run of id Q (≤ 2 s) between runs of the same id P goes back to P if no other box shows P then, or
   swaps back with the mirror excursion; never two boxes with one id in a frame.

The output is a normal run folder, evaluated as any other: the video drawn from it and its metrics agree.

## Results (15 val clips)

| | HOTA | AssA | IDF1 | IDSW |
|---|---|---|---|---|
| 038 | 79.7 | 74.9 | 89.6 | 379 |
| 038 + 048 | **79.8** | **75.0** | 89.6 | **364** |

15 excursions sent back in 15 clips (0 swapped pairs). HOTA per clip ([`hota.csv`](hota.csv)): +0.2 to +0.4 on 3
clips, −0.07 on 1, unchanged on 11.

## Reading

1. **Right where it acts, but it rarely can.** The id leaves a box chain in only 661 of ~117,000 frame-to-frame steps,
   and an excursion is almost never P → Q → P on ONE box chain: in the crowd where the id changes, the box chain itself
   ends (boxes merge, split or are missed), so "the same body before and after the crowd" is not known from the
   boxes. Knowing it is the tracking problem itself.
2. The ids that come back after a crowd come back through appearance (038's closed set re-identifies the player);
   the stretch in between is exactly where neither boxes nor appearance are sure.

## Conclusion

**Small positive (+0.1 HOTA, −15 switches, never worse by more than 0.07)**: kept as an optional post-process; the
showcase video can be drawn from it, since its metrics are the ones reported here.

## Commands

```powershell
python scripts/smooth_body_labels.py --run outputs/runs/038_continuity --out outputs/runs/048_body_smooth
python scripts/eval_hota.py --config configs/player.yaml --runs 038=outputs/runs/038_continuity 048=outputs/runs/048_body_smooth
```
