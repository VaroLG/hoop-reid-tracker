# 030 · Identity check at the exit of every crossing (online, fine-tuned SigLIP)

| | |
|---|---|
| **Date** | 2026-11-01 · **negative**: HOTA 68.8 → 67.8 online, 76.9 → 75.7 with the offline step; the failure mode is not the one the check looks for |
| **Builds on** | 029 (`scripts/switch_report.py`: 92 % of the identity switches happen within 1 s of an overlap, two thirds between opponents, made by the online tracker and kept by the offline step), 028 (fine-tuned SigLIP: right at 98.8 % of team-mate and 100 % of opponent crossings in the 016 benchmark), 015 (a swap check with OSNet failed because most crossings end with no swap) |
| **Question** | If the tracker checks, when two players separate, whether each one still looks like himself before the crossing, how many switches disappear, and how many false swaps does it create? |

## Method (fixed before running)

`hooptrack.tracking.crossing_check` (docstring: step by step), called after every `ReIDTracker.update`
(`crossing_check:` section of the config, off by default = behaviour up to 029). Per crossing: up to 8 clean crops
of each player in the 1.6 s before the overlap (only stored as images), 3 clean crops of each after they separate
(0.6 s), fine-tuned SigLIP on those crops only, `keep = d(A_before, A_after) + d(B_before, B_after)` vs
`swap = d(A_before, B_after) + d(B_before, A_after)`; swap the two identities (id, appearance, gallery, team votes)
if `keep - swap > margin`. Crossings with a third player, too few crops, or longer than 5 s are not decided.

**Margin, calibrated on train** (`scripts/calibrate_crossing_margin.py`): ground-truth crossings of the 15 train
clips, same windows as online (1.6 s before, 0.6 s after, one crop every 5 frames); the margin that maximises
expected swaps fixed minus swaps created, with the swap rate per crossing measured in 015 (~4 %).

**Only one change from 011b:** the crossing check (same detections, same tracker, same ResNet-50 for the tracker's
own appearance). Then the 029 offline step on top, to compare with 029.

**Measured with** IDSW and switches per minute by kind (`switch_report.py`: fragments / team-mates / opponents,
at crossings or not), plus HOTA / IDF1, and the time per frame. Decision: adopted if switches per minute fall on
most clips without a loss in HOTA, for the online run and for the full pipeline. The effect on team-mate
crossings is reported separately (the risky case of 015).

Tests: `tests/test_crossing_check.py` (a synthetic crossing: a swap by the tracker is undone; no swap, nothing done).

## Calibration: degenerate (and a bug in the rule)

On train, the fine-tuned SigLIP separates the 262 ground-truth crossings perfectly (it was trained on these
players): every margin from -0.2 to +0.3 fixes 100 % of simulated swaps with 0 % false swaps
(`crossing_margin_train.json`). The rule took the first margin of the tie, **-0.2**, which means swapping even when
"keep" is better by up to 0.2. A tie should never pick a negative margin: with these models, train cannot
calibrate this margin, and the natural value is 0 (swap only when swap is the better explanation).

## Results (15 val clips, `hota.csv`, `switches_per_clip.csv`)

| run | HOTA | AssA | IDF1 | IDSW | switches / min |
|---|---|---|---|---|---|
| 011b | 68.8 | 55.1 | 72.7 | 278 | 31.9 |
| 030 (011b + check, margin -0.2) | 67.8 | 53.5 | 71.3 | 302 | 34.8 |
| 029 | 76.9 | 69.1 | 83.9 | 278 | 31.3 |
| 030 + 029 offline | 75.7 | 67.0 | 82.5 | 298 | 33.7 |

Crossings logged (`<run>/<clip>/crossings/*.csv`): 1,096. **Decided: 179 (16 %)**; not decided 857 (a third
player in the crossing, or fewer than 2 clean crops before it); 60 dropped after 5 s (one of the two not visible).
Of the 179 decided, 12 swaps: 8 where "keep" was better (the negative margin), 4 where the model saw a swap.

## Why: the swaps happen in the crossings the check refused to decide

`switch_report` rows (`030_switches.csv`), switches to a team-mate's or an opponent's id:

* **They are reciprocal.** In 011b, 134 of 166 (81 %) have the mirror switch (player 1 takes id B, player 2 takes
  id A) within 1 s, 96 in the same frame. So "keep vs swap" between two tracks is the right question.
  (A first version of this analysis said "none is reciprocal": a pandas bug, `row.clip` is the `Series.clip`
  method, not the column, so every comparison was false. Corrected the same day, before any decision was based on it.)
* **They happen where the check did not decide.** Each switch of the 030 run matched to the logged crossing of
  the same ids (±1 s): 105 in crossings `not_decided` only, 27 `not_decided` + `timeout`, 13 `timeout`, 25 next to a
  (mostly wrong) `swap`, only 8 in crossings decided `keep`, 5 with no crossing logged. The swaps come from crowded
  crossings (a third player, players with few clean crops before) that the check dropped on purpose.

## Conclusion

**Negative, not adopted** (`crossing_check.enabled: false`). The question (keep vs swap) was right; the scope was
not: the check only decided clean two-player crossings, and the swaps happen in crowded ones. With margin 0 instead
of -0.2 it would have made 4 swaps, too few to matter. Next (031): decide crossings of any number of players (an
assignment between the looks after and the looks before) and record why a crossing is not decided.
