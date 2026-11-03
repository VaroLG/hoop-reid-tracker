# 035 · Stitching with a speed limit on the court

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [029](../029_stitch_siglip_reid/) stitching (adopted) · [034](../034_court_crossings/) court on all 15 validation clips |
| **Question** | Can the court reject the stitching joins that are physically impossible, where the pixel rule cannot? |
| **Clips** | all 15 validation clips |

## Why

The stitching refuses a join when the jump between two fragments is larger than the range the players cover in
the clip, in **pixels** (GTA's rule). That bound is very loose, and pixels mean little once the camera has panned or
zoomed between two fragments. On the court plan (034) a position is a position: if one fragment ends at one place and
the next begins too far away for the time between them, they are two players.

## Change (one switch, off by default)

`--max-speed` in [`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py)
(`GTAConfig.max_speed` in [`tracking/stitch.py`](../../src/hooptrack/tracking/stitch.py)): at every change of fragment
inside a candidate group, the last known feet of one fragment and the first known feet of the next (metres, from
`integrations.topdown.feet_on_court`) must be at most `max_speed × seconds + slack` apart, otherwise the join is
refused for good. Frames without a court are not checked. Values fixed before running, not tuned: **9 m/s** (players
in the 15 clips: 5.4 m/s at the 95th percentile, 034) and **1 m** slack (box position error at crossings: p90 0.27 m).
The stitching code receives positions only: it still never sees a homography. Tests in `tests/test_stitch.py`.

Same online run, same embeddings, same threshold as 029; only the rule is added.

## Results (15 val clips)

| | HOTA | AssA | IDF1 | IDSW | switches / min | fragments |
|---|---|---|---|---|---|---|
| 029 | **76.9** | **69.1** | **83.9** | 278 | 31.3 | 52 |
| 035: + speed rule | 76.8 | 68.9 | 83.5 | 281 | 31.7 | 54 |

The rule refused 15 joins on 8 clips (149 joins made) ([`joins.csv`](joins.csv)). HOTA per clip
([`hota.csv`](hota.csv)): better on 4 clips (+0.1 to +0.8), worse on 2 (FIBA c001 −3.2, FIBA 2010 c012 −1.9), equal
on 9.

## Reading

1. **The rule rarely acts**: the joins the stitching makes are almost always physically possible. The wrong
   joins that are left (two team-mates, two similar kits) are wrong in appearance, not in speed: most likely the two
   players are near each other when one fragment ends and the other begins (as in 034's crossings).
2. **When it acts, it is not reliable**: fragments end and begin at crossings, exactly where the box bottom is not
   the feet (034: the feet of the player behind are hidden two thirds of the time), so a right join can look like an
   impossible run and be refused (FIBA c001).

## Conclusion

**Negative (neutral to slightly worse).** Documented and kept behind `--max-speed`, off by default; 029 stays the
best run. With 034, this closes the court as a source of identity: it gives where the players are, not who they are.

## Commands

```powershell
# same embeddings as 029 (copied, not recomputed)
Get-ChildItem outputs/runs/029_siglip_ft -Directory | ForEach-Object {
    New-Item -ItemType Directory -Force "outputs/runs/035_speed/$($_.Name)" | Out-Null
    Copy-Item "$($_.FullName)/*.npz" "outputs/runs/035_speed/$($_.Name)/" }
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --thresh 1.0365 --split-teams --team-source siglip --team-k 2 `
    --max-speed 9 --speed-slack 1.0 --out outputs/runs/035_speed
python scripts/eval_hota.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft 035=outputs/runs/035_speed
python scripts/switch_report.py --config configs/player.yaml --runs 029=outputs/runs/029_siglip_ft 035=outputs/runs/035_speed
```
