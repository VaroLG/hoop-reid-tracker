# 029 · The fine-tuned SigLIP / CLIP of 028 in the offline stitching

| | |
|---|---|
| **Date** | 2026-11-01 · **positive, adopted**: HOTA 75.3 → **76.9**, IDF1 81.8 → **83.9**, better on 10 of 15 clips (SigLIP) |
| **Builds on** | 028 (fine-tuned SigLIP: AUC mate 0.860 vs OSNet 0.713, better on all 15 val clips), 020 (best pipeline: 011b online + team cut with SigLIP labels + GTA stitching with OSNet, IDF1 81.8, HOTA 75.3) |
| **Question** | Does the better ReID model, used where it is affordable (offline stitching), join fewer wrong pieces? |

## Method (fixed before running)

**One change from 020:** the ReID model used by `stitch_gta` to compare pieces (`reid.type` / `reid.weights`), and its
threshold. Everything else as 020: same online run (011b), same team cut (SigLIP zero-shot k = 2 labels, 1 s), 12
clean crops per piece, GTA mode.

**Threshold, calibrated on train** (`scripts/calibrate_stitch_threshold.py`): GTA's 0.4 is the operating point with
OSNet. Each model has its own distance scale, so the threshold is transferred on the **train** clips: the new model
gets the threshold at which it joins the same share of team-mate pieces by mistake as OSNet at 0.4 (fallback, if
OSNet makes no such error on train: the same share of right joins). Both models were trained on those clips, which
biases both the same way; written down as a caveat. Val is only used to report.

**Both** fine-tuned towers are stitched (SigLIP and CLIP), so the choice between them is not made on val.

**Decision rule:** adopted if HOTA and IDF1 improve over 020 on the 15 val clips, with the gain on most clips.

**Videos:** `scripts/render_compare.py` draws 020 and 029 side by side on a clip, identity switches (against the
ground truth) in red, with running counters.

## Calibration on train (`stitch_threshold_train.json`)

| model | threshold | team-mate pieces joined by mistake | same-player pieces joined | median distance: same / team-mate / opponent |
|---|---|---|---|---|
| OSNet 012 (reference, GTA's value) | 0.400 | 81.2 % | 100 % | — |
| SigLIP fine-tuned | **1.037** | 81.2 % | 100 % | 0.170 / 0.968 / 0.983 |
| CLIP fine-tuned | **1.043** | 81.2 % | 100 % | 0.136 / 0.955 / 0.964 |

15 train clips, 5,241 same-player, 20,259 team-mate and 25,635 opponent pairs of 2-second pieces.

**What this shows about 020:** with OSNet, GTA's 0.4 lets through 81 % of the team-mate pairs: on OSNet's scale (same
player ~0.04, others ~0.16) the threshold almost never says no. The decisions are made by the rest of GTA: pieces are
merged closest first, two pieces on screen together never merge, a jump must be spatially plausible, and (020) two
pieces of different teams never merge. The calibrated thresholds keep that same permissive operating point, so in
029 **the gain, if any, comes from a better ORDER of the candidate joins** (which pair is closest), not from a
stricter cut. A stricter threshold is a separate question, not tested here.

## A bug found by the first run (fixed before any result was read)

The first stitching run merged **every piece of a clip into one id** (`ids_after: 1`, two boxes with the same id in
the same frame). Cause: `stitch_gta` gave forbidden pairs (on screen together, different teams, different jersey
numbers, refused spatial jump) a distance of **1.0**, which only works while the threshold is below 1. The calibrated
thresholds (1.037, 1.043) are above it, so forbidden pairs passed. Fix: forbidden pairs are at +inf
(`stitch.FORBIDDEN`); for any threshold below 1 the output is identical to before. Test:
`test_gta_forbidden_pairs_stay_forbidden_above_threshold_one`. The results below are from the re-run.

## Results (15 val clips, `hota.csv`)

| run | ReID for stitching | threshold | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|---|---|---|
| 011b (online only) | — | — | 68.8 | 86.0 | 55.1 | 72.7 | 96.7 | 278 |
| 020 | OSNet 012 | 0.40 | 75.3 | 85.7 | 66.1 | 81.8 | 96.6 | 283 |
| **029 SigLIP** | SigLIP fine-tuned (028) | 1.037 | **76.9** | 85.7 | **69.1** | **83.9** | 96.6 | 278 |
| 029 CLIP | CLIP fine-tuned (028) | 1.043 | 76.5 | 85.7 | 68.3 | 83.1 | 96.6 | 280 |

| per clip vs 020 | HOTA better / worse / same | median Δ HOTA | IDF1 better / worse / same |
|---|---|---|---|
| **029 SigLIP** | **10 / 2 / 3** | **+1.8** | 10 / 1 / 4 |
| 029 CLIP | 10 / 1 / 4 | +0.8 | 9 / 1 / 5 |

Identities per clip after stitching: 11–14 (020: 10–14), the same number of merges (152 / 153 vs 152). The
largest gains: Tenerife c001 70.1 → 75.1 and Istanbul c014 66.0 → 70.7; the losses: NCAA c007 64.8 → 62.7 (SigLIP;
CLIP 67.6) and Tenerife c003 77.2 → 77.0 (CLIP). The 3–4 clips with no change are clips where the same joins are
made.

## Reading

1. **The better ReID model pays off where it is used.** Same online run, same team cut, same number of merges:
   only WHICH pieces are joined changes, and AssA rises 66.1 → 69.1. This is the 028 benchmark result (team-mates
   told apart better) turning into fewer wrong joins.
2. **Both towers help** (10 clips better each); SigLIP more (+1.8 vs +0.8 median). Both were stitched so that the
   comparison is visible; SigLIP is adopted because it also led on the benchmark (028), but the difference between the
   two is small and measured on these same clips.
3. **Best shippable result of the project:** 76.9 HOTA / 83.9 IDF1 with code and models that are ours. Only Deep-EIoU
   online + our offline (024, 77.4 / 83.8, not shippable: no licence) is in the same range; with SigLIP stitching on
   top of it the two gains might add up (not tested).
4. **Cost**: SigLIP runs offline only, on ~12 crops per piece: no change to the online tracker's speed.
