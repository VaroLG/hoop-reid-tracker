# 047 · Sending back the ids that leave a player and come back (diagnosis: they are not where the rule looks)

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [038](../038_closed_set_continuity/) (adopted) · [039](../039_smooth_excursions/) (sandwich rule: nothing to smooth) |
| **Question** | In the honest demo video an id often leaves a player for 1–2 s in a crowd and comes back. Can a rule without ground truth remove those excursions, so the output (and the metrics) never show them? |
| **Clips** | 15 val clips. The rule has no parameter; it was designed after watching val clip c001, said here |

## Measured first (038, 15 val clips)

Of the 366 identity switches, **136 are a player leaving his id and getting it back later** (89 within 2 s, 28 in
2–5 s, 19 later); about as many are the switches back. **~94 never come back** (permanent changes, new ids).

## Change (off by default)

`--smooth-swaps` ([`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py),
`smooth_swapped_excursions` in [`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py)): inside each online
tracklet, a run of pieces labelled Q between pieces labelled P goes back to P if P is free then, or swaps back with
the mirror run (P between Q's in another online tracklet) when that is what keeps P busy. 039's rule, at the level of
runs and pairs.

## Result

It finds **4 excursions in 15 clips** (0 on 11 clips): nothing measurable. Where the excursions really are (4 clips,
49 excursions, the boxes of the player during the excursion):

| what happens during the excursion | excursions |
|---|---|
| the tracker's output is **right**: in a crowd the player's box overlaps another and the evaluation matches him to the other box for a moment (a counted switch, no visible change of label on the player) | 21 |
| the **online tracker** followed the other player for a while and came back, inside one piece (no crossing cut there) | 20 |
| a clean piece labelled with another player (what the rule targets) | 8 |

## Reading

1. The excursions are mostly **not label decisions**: about half are crowd boxes the evaluation matches back and
   forth, and most of the rest happen inside the online tracker, before any labelling. A post-processing rule on
   labels cannot see them.
2. **Hiding them only in the video would make the video disagree with the metrics.** A smoothing that is right
   belongs in the pipeline and then shows up in the numbers; one that needs the ground truth to know which id is
   right cannot be used at all.

## Conclusion

**Negative (4 moves in 15 clips), kept off.** The showcase video stays the tracker's real output.

## Command

```powershell
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --split-crossings --closed-set 10 --continuity 0.3 --smooth-swaps --out outputs/runs/047_smooth_swaps
```
