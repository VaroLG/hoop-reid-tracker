# 042 · Where 038's wrong identities come from, and a soft kit term

| | |
|---|---|
| **Date** | 2026-11-06 |
| **Builds on** | [038](../038_closed_set_continuity/) closed set of 10 + continuity (adopted) · [040b](../040_clip_adaptation_rules/) hard "never to the other team" rule (negative) |
| **Question** | (a) Which kind of error is left in 038? (b) 040b failed because its kit labels were wrong; with better kit labels and a soft term instead of a hard rule, do the cross-team errors go away? |
| **Clips** | all 15 validation clips |

## (a) Diagnosis: error breakdown of 038

[`scripts/identity_error_breakdown.py`](../../scripts/identity_error_breakdown.py) (evaluation only): every predicted
box matched to a ground-truth player (IoU ≥ 0.5) is wrong when its final id belongs mostly to another player; each
wrong box is traced back to its piece (online tracklet cut at crossings, as 038 does).

| kind of wrong box (038, 15 clips) | boxes | share | piece length (most) |
|---|---|---|---|
| **impure piece**: the online tracker swapped inside a piece | 2,251 | 27 % | 3–10 s |
| **pure piece given to a team-mate** | 3,094 | 37 % | 1–3 s |
| **pure piece given to an opponent** | 3,024 | 36 % | 3–10 s |
| *all wrong boxes* | *8,369 of 116,726* | *7.2 %* | |

([`errors.csv`](errors.csv)). Three quarters of the errors are **assignment** errors on pieces that are clean. On 5
clips we also checked why: for ~2/3 of those boxes the right player was already busy in the piece's frames, i.e. an
earlier wrong decision blocked the right one (038 never revisits a decision). This motivates 043.

Kit labels: Lab torso colour of the clean boxes (one frame in 5, 2-means per clip) gives the kit of the right
ground-truth team for 93 % of the pieces (96 % of the 3–10 s pieces); the zero-shot SigLIP torso vectors used by
040b: 83 %, and **at chance (56–66 %) on all six NCAA clips**. (The ground-truth kits come from the same torso
colour, so the Lab figure is optimistic.) 040b's hard rule was fed labels at chance on a third of the clips.

## (b) Change (one switch, off by default)

`--closed-team-penalty λ` ([`scripts/stitch_tracklets.py`](../../scripts/stitch_tracklets.py),
`assign_closed_set(team_prob=..., team_penalty=...)`): P(kit) per piece = (n₁ + 1) / (n + 2) over its clean boxes
(a piece with few clean boxes stays near 0.5); a piece's distance to a player is raised by λ × P(piece is not of that
player's kit). **λ = 0.5, fixed before running** (above the ~0.34 gap between same-player and team-mate distances,
so a clearly other-kit piece is pushed away). 042: seeds split 5/5 by likelihood; 042b (`--closed-team-own`, decided
**after** seeing 042 on val): each seed keeps its own kit. With λ = 0 the output is identical to 038 (checked).

## Results (15 val clips)

| | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| **038** | **79.7** | 84.9 | **74.9** | **89.6** | 96.6 | **379** |
| 042 soft kit term | 77.8 | 84.9 | 71.3 | 86.3 | 96.5 | 395 |
| 042b + seeds keep their own kit | 77.8 | 84.9 | 71.4 | 86.4 | 96.5 | 393 |

Per clip ([`hota.csv`](hota.csv)): better on 3 (NCAA c008, c011, FIBA 2010 c008), equal on 3, worse on 9 (FIBA
c002 −6.9, c003 −8.4, 2010 c014 −9.0).

Error breakdown, 038 → 042: impure 2,251 → 2,193, **opponent 3,024 → 3,188**, team-mate 3,094 → **5,715**.

## Reading

1. **The cross-team errors are not a kit problem.** Even with kit labels right on 96 % of the long pieces and a
   penalty larger than the team-mate gap, the opponent errors do not go down. They come from the cascade of (a): the
   right player is busy, so the piece goes wherever it is allowed.
2. **Any extra additive term reorders the greedy.** The most-confident-first order compares costs across pieces;
   shifting them (here by up to λ/2 for uncertain pieces) changes which pieces are decided first and the team-mate
   errors almost double. A term that is right on average still hurts a greedy procedure.
3. 040b's failure is now explained (kit labels at chance on NCAA), but fixing the labels is not enough.

## Conclusion

**Negative, kept off by default.** The diagnosis is the useful part: 73 % of the remaining errors are assignment
errors on clean pieces → 043 tries a global (non-greedy) assignment.

## Commands

```powershell
python scripts/identity_error_breakdown.py --runs 038=outputs/runs/038_continuity `
    --online outputs/runs/011b_no_referee_filter/segmentation --out experiments/042_soft_team_cost/errors.csv
# embeddings copied from 038 (same pieces)
python scripts/stitch_tracklets.py --run outputs/runs/011b_no_referee_filter/segmentation `
    --config configs/player.yaml --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot `
    --split-crossings --closed-set 10 --continuity 0.3 --closed-team-penalty 0.5 --out outputs/runs/042_soft_teams
#   042b: add --closed-team-own, --out outputs/runs/042b_soft_teams_own
python scripts/eval_hota.py --config configs/player.yaml --runs 038=outputs/runs/038_continuity `
    042=outputs/runs/042_soft_teams 042b=outputs/runs/042b_soft_teams_own
```
