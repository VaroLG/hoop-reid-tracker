# Chapter 2 · Court geometry

The tracking chapter (`experiments/001`–`008`) asks *who is who*. This chapter
asks *where is the court*: a homography per frame between the broadcast image
and a court plan. It is a separate chapter with its own code, scripts, tests
and numbering (C01, C02, …), because it is useful on its own and must not
tangle the tracker.

| Lives in | Contents |
|---|---|
| `src/hooptrack/court/` | templates per league, keypoint detector wrapper, homography estimation |
| `scripts/court/` | dataset download, audit, conversion, training, evaluation |
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
| C02 | Train a court-keypoint detector (YOLO26-pose) on the public data; test it on SportsMOT frames | ⏳ |
| C03 | One template per league (NBA, NCAA, FIBA); pick the league of each video from the data | — |
| C04 | Pseudo-label SportsMOT train (geometric consistency + review), retrain, measure on held-out frames | — |
| C05 | Homography for every frame: detector + camera-motion propagation + quality checks | — |
| C06 | Top-down map: players' feet (pose ankles) projected to metres | — |
| → tracking | Court mask from the homography, measured on the full validation split (tracking experiment) | — |
