# 032 · Diagnosis: where does the tracker lose what the ReID model knows at crossings?

| | |
|---|---|
| **Date** | 2026-11-02 · **diagnosis done**: the model is right where it can look (99.4 %); 83 % of the swaps happen where it cannot (no clean view) |
| **Builds on** | 016/028 (fine-tuned SigLIP right at 98.8 % of team-mate and 100 % of opponent crossings, ground-truth crops), 030/031 (the same decision inside the tracker: ~40 % of its corrections right) |
| **Question** | Which difference between the benchmark and the tracker makes the decision fail? |

## Method (diagnostic only, the tracker is not changed)

`scripts/diagnose_crossing_check.py`. Every ground-truth crossing of the 15 val clips, the same decision
(keep vs swap, right when keep < swap: ground-truth identities never swap), with inputs closer and closer to the
tracker's, one change at a time:

| level | look before | look after |
|---|---|---|
| L1 | ground-truth crops (clean, >= 50 px, 1 in 5 frames), **1.6 s** before | ground-truth crops, **0.6 s** after (the online windows; the benchmark used 2 s / 2 s) |
| L2 | the **detections** matched to those ground-truth boxes (IoU >= 0.5) | same, detections |
| L3 | the crops of the **id the online run (011b) had on the player** just before the crossing, clean among the run's boxes; crops whose ground-truth owner is another player are counted as contaminated | as L2 |

The level where the accuracy falls points at the cause: shorter windows (L1), detector boxes (L2), or the tracker's
own memory of a player being someone else (L3).

```powershell
python scripts/diagnose_crossing_check.py --config configs/player.yaml `
    --model siglip_ft:siglip:weights/siglip_reid_sportsmot `
    --online outputs/runs/011b_no_referee_filter/segmentation --out outputs/runs/032_diagnosis.csv
```

## Results (15 val clips, 1,821 ground-truth crossings: 734 team-mates, 1,087 opponents; `diagnosis.csv`)

| level | crossings decided | right | team-mates | opponents |
|---|---|---|---|---|
| L1 ground truth, tracker windows (1.6 s / 0.6 s) | 338 | 99.4 % | 98.2 % | 100 % |
| L2 detector boxes | 338 | 99.4 % | 98.2 % | 100 % |
| L3 the tracker's look before | 338 | 99.4 % | 98.2 % | 100 % |

(321 crossings decided at all three levels, 99.4 % each.) L3: the look before contains another player's crops in
3.3 % of the crossings; those are still decided right (11 of 11).

**None of the three suspected causes explains 031.** Short windows, detector boxes and the tracker's own memory
leave the decision as good as in the benchmark.

## Where the swaps are

The 166 switches of 011b to a team-mate's or an opponent's id (`030_switches.csv`), matched to these crossings:

| | switches |
|---|---|
| tracker boxes of the two ids overlapped in the second before (the check can see the crossing) | 157 |
| in a ground-truth crossing that is **decidable** (>= 2 clean crops before and after) | **25** |
| in a ground-truth crossing that is **not decidable** | **138** |
| no ground-truth crossing nearby | 3 |

Only 18.6 % of all crossings are decidable, and **83 % of the swaps happen in the others**: crowded moments where
no player has a clean view (no crop free of another box) in the 1.6 s before or the 0.6 s after. The model is
right 99 % of the time where it can look; the swaps happen where it cannot.

## Conclusion

A check at the exit of a crossing, with the views available around it, can at most address ~15 % of the swaps
(25 of 166), whatever the model. This closes the 030/031 line. The identity of a player who went through a crowd
can only be recovered **later**, when he is seen clearly again, by something that does not depend on the moments
around the crossing: a long-range link between pieces with a strong anchor. Jersey numbers (033) are that kind of
anchor; the offline stitching (029) is where such a link lives.
