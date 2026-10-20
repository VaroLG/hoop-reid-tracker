# Chapter 2 · Court geometry

The tracking chapter (`experiments/001`–`008`) asks *who is who*. This chapter
asks *where is the court*: a homography per frame between the broadcast image
and a court plan. It is a separate chapter with its own code, scripts, tests
and numbering (C01, C02, …), because it is useful on its own and must not
tangle the tracker.

| Lives in | Contents |
|---|---|
| `src/hooptrack/court/` | templates per league, homography estimation, line refinement, anchors, propagation through video |
| `scripts/court/` | dataset download, audit, conversion, training, evaluation, homography propagation, court-mask evaluation, overlay video |
| `tests/court/` | geometry tests + a test that keeps the chapter independent of the tracker |
| `experiments/court/` | one folder per step, same rules as chapter 1 (hypothesis, numbers, conclusion) |
| `data/court_keypoints/` | datasets (not committed) |

**Dependency rule.** `hooptrack.court` never imports the tracker, and the tracker
never imports `hooptrack.court`. When the court model is ready, a single adapter
(`hooptrack/integrations/`) will expose it to the pipeline behind a config switch,
and its effect on tracking will be measured as a normal tracking experiment.
`tests/court/test_separation.py` fails if either rule is broken.

## Why

- The colour court mask (003) loses up to 10 % of players in another arena (008).
  Projecting the real court rectangle works in any arena.
- A top-down map of players (positions in metres, distances, speeds, heatmaps)
  needs the same homography.

## Steps

| # | Step | Status |
|---|---|---|
| [C01](C01_dataset_audit/) | Public dataset: audit, leak-free split, court template verified on 850 images | ✅ |
| [C02](C02_keypoint_detector/) | Court-keypoint detector (YOLO26n-pose, CPU, 3 h): 0.27 m on NBA test frames — **does not transfer** to SportsMOT | ✅ partly |
| [C03](C03_line_refinement/) | Templates per league (NBA, NCAA, FIBA) + line-based refinement and agreement score: fixes close starts, cannot rescue bad ones | ✅ partly |
| [C04](C04_anchor_homographies/) | Hand anchors on 3 frames + propagation (optical flow + lines): homography for **every frame** of the NCAA val clip, two independent chains agree to 5–20 cm; court mask removes 82.5 % of non-players vs 71.9 % (colour) | ✅ |
| [C05](C05_automatic_homographies/) | **No annotation**: candidates from straight lines + physics filter, confirmed by two chains, league picked automatically. NCAA: 6 cm from the hand result; Golden State (NBA): 99.8 % players kept vs 97.3 %, 2× more non-players removed. **Limit (found in 009): on unseen FIBA arenas it can be confidently wrong** — two wrong candidates can confirm each other | ✅ partly |
| C06 | Learned line / keypoint detector trained on C05's automatic homographies (free labels, from the clips where they are right), for courts with faint lines and filled areas | — |
| C07 | Top-down map: players' feet (pose ankles) projected to metres | — |
| → tracking | [009](../009_court_mask_homography/): court mask from the homography inside the tracker, 4 validation clips — helps only where the homography is right; default unchanged | ✅ negative/partial |
