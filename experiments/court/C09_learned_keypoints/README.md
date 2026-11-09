# C09 · Learned court keypoints: a homography for any court, with no annotation

| | |
|---|---|
| **Date** | 2026-10-24 → 2026-10-26 |
| **Builds on** | [C02](../C02_keypoint_detector/) keypoint detector · [C05](../C05_automatic_homographies/) automatic homographies · [C06](../C06_court_segmentation/) synthetic courts · [C07](../C07_clip_camera/) clip camera · [C08](../C08_topdown_map/) top-down map |
| **Question** | Can one model find the court keypoints on courts it has never seen, so that the user only says *which kind of court* it is (or gives its measurements) and every frame gets a homography? |
| **Test set** | 6 SportsMOT **validation** clips (never trained on): 2 NCAA, 2 FIBA, 2 FIBA 2010 (old court), 41 frames |

> **Data and licences.** Training data: DeepSportRadar (CC BY-NC-SA 4.0), court keypoints v22 (CC BY 4.0), SportsMOT (CC BY-NC 4.0) train clips, synthetic courts with random measurements; starting model: YOLO26n-pose (AGPL-3.0); weights not redistributed. All sources: [`docs/data_models_licences.md`](../../../docs/data_models_licences.md).

## Approach

Two separate pieces:

* **The model** (YOLO26n-pose, 33 named keypoints) says *which* point it sees: "lane corner on
  the baseline", "free-throw line end"... in pixels. It must not depend on the court's measurements.
* **The court type** says *where* each named point is, in metres: a preset (`NBA`, `FIBA`,
  `FIBA_PRE2010`, `NCAA`) or custom measurements in a YAML file
  ([`configs/courts/`](../../../configs/courts/)). It is chosen by the user; it sets the scale of
  every position, distance and speed.

Keypoints + court type → RANSAC homography → polished on the painted lines → C07 checks that one
camera explains the whole clip. Measurements are stored in each `homographies.npz`, so a custom
court reaches the mask, the map and the speeds.

## Test set (hand-made)

`scripts/court/annotate_anchors.py --manifest test_set.json`: 3–5 frames per clip, 5–9 visible line
crossings each, fitted and polished on the lines. Annotating revealed that the Istanbul clips
(USA–RUS) are the **2010 World Championship**, played on the pre-October-2010 FIBA court
(trapezoid lane 6.0 → 3.6 m, 3-pt at 6.25 m): new preset `FIBA_PRE2010`.

The anchors are **only** a test: they never feed training or inference.

## Training data (labels from a homography)

| source | images | |
|---|---|---|
| v22 | 850 | NBA 2025 broadcasts (train / val / test) |
| deepsport | 728 | 15 French LNB arenas, FIBA, calibrated (split by arena) |
| sportsmot | 687 | SportsMOT **train** clips accepted by C07 (3 FIBA, 8 NBA; no NCAA) |
| synth | 2000 | first run: synthetic courts on the 3 fixed templates (NBA, FIBA, NCAA) |
| **synth_random** | **4000** | second run: every synthetic court with **random measurements** (court 24–29.5 × 13.5–16 m, lane 3.4–6.4 m, 20 % trapezoids, random free-throw, 3-pt radius and corner) |

No real NCAA frame is ever used: NCAA is the test of a court type never seen in a real image.
Training on Colab T4 ([notebook](../../../notebooks/colab_train_court_keypoints.ipynb)); on the laptop CPU, 3 h gave 2 epochs.

## Results

Error = median distance between the method's court and the hand-made one on the visible court;
**wrong** = more than 1 m. Court type given for C09 rows.

| method | covered | median | < 0.5 m | wrong |
|---|---|---|---|---|
| C05 + C07 (lines only, before C09) | 23/41 | 0.05 m | 100 % | 0 |
| C09, CPU (2 epochs) | 22/41 | 0.96 m | 5 % | 10 |
| C09, fixed templates, court type auto | 38/41 | 0.65 m | 45 % | 7 |
| C09, fixed templates + C07 | 13/41 | 0.21 m | 85 % | 0 |
| **C09, random measurements + C07** | **37/41** | **0.28 m** | **92 %** | **1** |

| clip | C05 + C07 | C09 fixed templates | **C09 random measurements** |
|---|---|---|---|
| NCAA c001 | 23/24 · 0.05 m | 0.67 m · rejected by C07 | **20/24 · 0.30 m** |
| NCAA c005 | 0/3 | 0.72 m · rejected by C07 | **3/3 · 0.34 m** |
| FIBA Changsha | 0/3 | 0.06 m | 0.11 m |
| FIBA Tenerife | 0/3 | 0.21 m | **0.11 m** |
| FIBA 2010 c008 | 0/5 | 0.36 m (4/5) | 0.32 m (5/5), 1 wrong |
| FIBA 2010 c012 | 0/3 | 0.46 m | **0.31 m** |

Files: [`eval_test.json`](eval_test.json), [`eval_test_randdims.json`](eval_test_randdims.json);
examples: [`examples_by_court.jpg`](examples_by_court.jpg) (fixed templates),
[`examples_by_court_randdims.jpg`](examples_by_court_randdims.jpg) (random measurements).

## Reading

1. **Random measurements are what generalises.** With fixed templates the model learned the
   *proportions* of NBA/FIBA courts and pulled NCAA points towards them (0.7 m, rejected).
   With random measurements and still no real NCAA image, NCAA drops to 0.3 m and is accepted.
2. **The court type must come from the user.** Chosen automatically (best line agreement), it
   was wrong on 4 of 6 clips and caused 7 of the 8 wrong courts. A wrong type still looks
   plausible in the image, but every distance in metres is distorted.
3. **C07 does not catch a wrong court type**: a wrong template can still be explained by one
   consistent camera. It does catch broken frames (c008 f241).
4. **Use `H_ptz`.** A raw fit can be right on the visible court and wild outside it (c005
   f266: the mask spills into the stands); the clip-camera homography is not. The map already
   used it; the tracker's homography court mask now does too (`filters.homography.use_ptz`).
5. C05 is still the most precise where it works (NCAA c001, 5 cm), but it found no FIBA court.

## End to end: the court in time

[`c09_pipeline.ps1`](../../../scripts/court/c09_pipeline.ps1) on the 6 clips (court type given, nothing
annotated): C07 accepts all 6, tracking IDF1 75.3 / MOTA 96.2. But the maps flickered: each frame's
court is computed on its own, so on the hard clips **12–24 % of the frames had no court** (almost all
in gaps of ≤ 5 frames) and the court **trembled 16–23 cm from frame to frame** (p95 0.6–0.9 m, one
6.4 m jump), which the map turns into players vanishing and fake speed.

The camera head moves smoothly, and C07 already reduces each frame to its pan / tilt / roll / zoom.
`clipcam.smooth_ptz` filters those 4 numbers in time: drop frames far from the running median,
interpolate gaps of up to 15 frames, Savitzky–Golay over 15 frames. Saved as `H_smooth` next to
`H_ptz`; the map and the homography court mask use it (`camera_smooth`).

| against the hand truth (250 frames, `--ncaa-step 5`) | covered | median | < 0.5 m | wrong |
|---|---|---|---|---|
| H_ptz, frame by frame | 84 % | 0.37 m | 78 % | 1 |
| **filtered (window 15)** | **90 %** | **0.34 m** | **85 %** | **0** |

Windows 9 and 25 give the same (0.34 / 0.32 m). Frames without a court on the map: NCAA c001 20 → 11 %,
c008 24 → 6 %, c012 21 → 4 %; worst frame-to-frame jump 6.4 → 0.4 m.

| clip | mean speed, before → after | p95 | worst id p95 |
|---|---|---|---|
| NCAA c001 | 2.45 → 2.02 m/s | 6.1 → 5.2 | 8.9 → 6.2 |
| NCAA c005 | 2.88 → 2.42 | 6.6 → 5.6 | 12.7 → 8.1 |
| FIBA Changsha | 2.30 → 2.10 | 5.8 → 5.1 | 10.1 → 7.9 |
| FIBA Tenerife | 2.09 → 2.02 | 5.2 → 5.0 | 12.9 → 7.6 |
| FIBA 2010 c008 | 3.80 → 2.99 | 8.2 → 6.5 | 18.8 → 11.6 |
| FIBA 2010 c012 | 3.07 → 2.61 | 7.3 → 5.6 | 45.4 → 18.8 |

Speeds of identities with ≥ 200 map frames (median over them). The remaining peaks on the old court
are single identities: most likely identity swaps of the tracker, not the court.

### Completing and polishing with the image itself

Watching the maps, two things were still wrong: ~10 % of frames without a court (midcourt views
and close-ups under the basket, where the detector finds few named points; holes up to 45 frames),
and courts off the painted lines, worst in fast pans and zooms. Two causes: C07's single camera
centre (a zoom lens moves its optical centre, 0.25–0.3 m bias on NCAA / Istanbul), and a temporal
filter, which assumes the camera moves slowly and lags behind a fast pan.

[`c09_complete_court.py`](../../../scripts/court/c09_complete_court.py) uses the image instead:

1. **holes**: the court is carried into them frame by frame with the floor's optical flow from both
   sides, re-aligned on the lines at each step (C04's tracker), cut when lost, and handed over
   gradually from one side to the other (picking one side per frame made the court jump 1.8 m);
2. **polish** every frame on its painted lines;
3. **C07 check** against the clip camera;
4. **flow smoothing** (`propagate.flow_smooth`): each frame's court is the median of its own and
   its ±7 neighbours' polished courts, each carried to it by the measured image motion. No lag
   (the motion is measured, not assumed smooth), isolated wrong frames dropped by the median.

Tried and dropped on c008: letting the polished frame overrule the filter where its lines agree
better (helps fast zooms but the switching makes the court jump 2 m), and a plain Savitzky–Golay on
the polished courts (lags like the camera filter).

| against the hand truth (250 frames) | covered | median | < 0.5 m | wrong |
|---|---|---|---|---|
| H_ptz, frame by frame | 84 % | 0.37 m | 78 % | 1 |
| camera filtered in time (window 15) | 90 % | 0.34 m | 85 % | 0 |
| polished, frame by frame | 100 % | 0.22 m | 94 % | 2 |
| **polished + flow smoothing** | **100 %** | **0.20 m** | **98 %** | **0** |

Coverage on the whole clips: 96.5–100 % (holes at the very start / end of a clip have one side
only and are filled only while the tracker keeps the lines). c008 in the fast pan (f340–356):
line agreement 0.47 (camera filter) → 0.59 (polished frame by frame: 0.66). File:
[`eval_test_temporal.json`](eval_test_temporal.json).

Time per clip, laptop CPU (court steps only; tracking and the map video are separate):

| clip | frames | keypoints + homography | C07 | completion + polish | total |
|---|---|---|---|---|---|
| NCAA c001 | 1162 | 620 s | 18 s | 523 s* | 19 min |
| NCAA c005 | 594 | 273 s | 20 s | 138 s | 7 min |
| FIBA Changsha | 886 | 419 s | 13 s | 221 s | 11 min |
| FIBA Tenerife | 1215 | 542 s | 14 s | 243 s | 13 min |
| FIBA 2010 c008 | 500 | 201 s | 8 s | 103 s | 5 min |
| FIBA 2010 c012 | 620 | 291 s | 9 s | 130 s | 7 min |

About 0.45 s per frame for the keypoints and 0.2 s for the completion (*c001 ran while other jobs
shared the CPU, and its 133 hole frames are tracked as well as polished).

## Conclusion

**Positive.** One model, trained with no real image of the court type, gives homographies on
NCAA, FIBA and the old FIBA court with 0.1–0.3 m median error and C07 verification; the user
only gives the court type. Next: a cross-check between the C06 segmentation mask and the
homography mask (catches wrong court types, which C07 cannot), and more test clips.

## Commands

```powershell
# data and training
python scripts/court/c09_build_keypoints.py synth_random --n 4000 --sources deepsport sportsmot synth_random
python scripts/court/c09_pack_colab.py --out data/court_kp_random.zip      # then the Colab notebook
# any clip, end to end (homographies, C07, completion + polish, tracking, 2D map): only the court type is given
powershell -File scripts/court/c09_pipeline.ps1 -Clips v_5ekaksddqrc_c003=FIBA v_xxx=configs/courts/high_school_nfhs.yaml
# evaluation against the hand-made test set
python scripts/court/eval_court_test.py --method c09=outputs/court_c09_randdims c09+c07=outputs/court_verified/c09_randdims
```
