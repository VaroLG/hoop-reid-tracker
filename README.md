<div align="center">

# 🏀 hoop-reid-tracker

**Identity-preserving multi-player tracking for broadcast basketball, with the ball and who has it**

YOLO26 detection · a tracker written from scratch · player re-identification · exactly 10 identities · ball tracking ·
possession · court geometry<br>
developed **experiment by experiment**, every change measured on [SportsMOT](https://github.com/MCG-NJU/SportsMOT) (ICCV 2023)

[![CI](https://github.com/VaroLG/hoop-reid-tracker/actions/workflows/ci.yml/badge.svg)](https://github.com/VaroLG/hoop-reid-tracker/actions)
![python](https://img.shields.io/badge/python-3.11%E2%80%933.14-blue)
![detector](https://img.shields.io/badge/detector-YOLO26-purple)
![license](https://img.shields.io/badge/code-MIT-green)
![status](https://img.shields.io/badge/status-v1.0-brightgreen)

<img src="assets/final/showcase_c001.gif" width="720" alt="Final pipeline on an NCAA broadcast: players drawn in their team's kit colour with a tag, the ball with a trail, the player who has it highlighted, a possession panel and a court minimap">

<sub><b>The final pipeline</b> on NCAA 2015 (<code>v_00HRwkvvjtQ_c001</code>, a validation game never used for training): players
by team (038 + 048), ball (B08), possession (P04), court minimap (C09). Every identity drawn is the tracker's own output,
nothing is corrected by hand. This is <b>one of the cleanest clips</b>: the table below gives its numbers next to the
numbers over all 15 validation clips, and the next video shows where it still goes wrong. First 26 s shown: two piles
under the basket, the camera following the play and the players on the court plan (bottom right).
▶ <a href="assets/final/showcase_c001.mp4">full clip (46 s)</a></sub>

</div>

---

## Final results

**Players** (SportsMOT validation, basketball; "all 15 clips" = 4 games, 8.4 minutes, metrics recomputed over all clips
together):

| | HOTA | IDF1 | MOTA | ID switches | false positives | missed player boxes | ids for 10 players |
|---|---|---|---|---|---|---|---|
| clip of the video (NCAA c001, 46 s) | 86.4 | 96.6 | 98.2 | 11 (14 / min) | 41 (0.4 %) | 153 of 11,276 (1.4 %) | 11 |
| **all 15 validation clips** | **79.8** | **89.6** | **96.6** | **364 (43 / min)** | 1,509 (1.3 %) | 2,205 of 118,825 (1.9 %) | 11–26 per clip |

* **IDF1**: share of the boxes that carry the right player's identity over the whole clip. **HOTA**: detection and
  association together (the SportsMOT benchmark's main metric). **MOTA**: 1 − (false positives + misses + ID switches)
  / player boxes.
* **ID switches**: a player's box changes identity (CLEAR-MOT counting). **False positives**: boxes on nobody (referees,
  the bench). **Missed player boxes**: players with no box (mostly hidden in a pile).

**Ball and possession** (the 4 hand-labelled test clips; the NCAA clip in brackets):

| ball (B08) | found | right when it answers | F1 | | possession (P04) | frames right | catches / releases found | event F1 |
|---|---|---|---|---|---|---|---|---|
| | 71 % (83 %) | 85 % (92 %) | 0.78 (0.87) | | | 86 % (93 %) | 78 % (90 %) | 0.50 (0.61) |

Ball "found" = within one ball diameter of a hand click; possession = held / in the air against 100 hand-marked
catches and releases (a trivial "always held" gets 79 % of the frames).

### Where it still goes wrong: the same video with boxes

<div align="center">
<img src="assets/final/showcase_vs_boxes_c001.gif" width="900" alt="Side by side: left the showcase video, right the same frames with a box per player in the kit colour seen in that frame, the tag with the tracker's raw id, and red boxes marked ID SWITCH where an identity changes">
</div>

<sub>Left: the showcase drawing. Right: the same output drawn to show its errors (<code>--mode honest</code>): corner brackets
in the kit colour seen <i>in that frame</i> (thin when another box covers the player), the tag with the team label and the
tracker's raw id, and every identity switch measured against the ground truth framed in red with "SWITCH". The
switches happen in piles, where one box covers two players or slides from one to the other; most are short and the id
comes back once the players separate, which is why the showcase looks clean. Why they are hard to remove:
<a href="#16--what-limits-identity-measured-042046">section 16</a>. ▶ <a href="assets/final/showcase_vs_boxes_c001.mp4">full clip, side by side</a></sub>

## The court: mask, homography and map (chapter 2)

A second chapter of the project finds the **court in every frame**. It gives the tracker its court mask (only boxes
whose feet are on the court are kept: no crowd, bench or photographers) and every player a position **in metres** on
the court plan (the minimap of the video above, distances, speeds). It has its own code (`src/hooptrack/court/`),
scripts (`scripts/court/`), tests and experiments ([`experiments/court/`](experiments/court/)); the tracker and the
court code never import each other, the only bridge is `src/hooptrack/integrations/`.

| piece | how | measured on validation arenas never trained on |
|---|---|---|
| **Court mask** (C06) | YOLO26n-seg trained on 15 French arenas, NBA broadcasts, SportsMOT train and synthetic courts with random floors, paint and cameras | IoU **0.82–0.94** with the true court; never catastrophically wrong. Against the floor-colour mask on 4 arenas: false positives 6,897 → 3,989, missed players 4,169 → 1,732, IDF1 45.0 → 53.1 (section 3) |
| **Court keypoints → homography** (C09) | YOLO26n-pose finds 33 named court points; the **court type** (NBA, FIBA, FIBA before 2010, NCAA, or custom measurements) says where they are in metres; RANSAC, then polished on the painted lines | **0.28 m median** error against hand-made courts, **92 %** of the frames under 0.5 m, 37 of 41 test frames covered, 1 wrong; NCAA never seen in a real training image |
| **One camera per clip** (C07) | a homography that no fixed broadcast camera can produce is rejected | wrong courts go to review instead of into the map |
| **Court plan** (C08) | each player's feet projected with the homography of the frame | positions **5.7 cm** (median) from the hand-made court on the NCAA clip |

<div align="center">
<img src="experiments/court/C09_learned_keypoints/examples_by_court_randdims.jpg" width="760" alt="Six frames from three kinds of court (NCAA, FIBA, FIBA 2010 with the old trapezoid lane): the court found by the learned keypoints drawn in green with its lines in yellow">
<br><sub>The court found on the three kinds of court of the validation games, with no annotation: NCAA, FIBA, and the 2010
World Championship played on the old FIBA court (trapezoid lane), found while annotating the test set.
▶ <a href="experiments/court/C09_learned_keypoints/">C09</a></sub>
<br><br>
<img src="experiments/court/C08_topdown_map/ncaa_map.gif" width="480" alt="NCAA broadcast with tracked players and the court lines, and below the court plan with each identity's position and trail">
<br><sub>The tracks on the court plan, in metres (C08). Made with an earlier tracker configuration (COCO detector); the final
pipeline draws the same plan as the minimap of the video above. ▶ <a href="experiments/court/C08_topdown_map/">C08</a></sub>
</div>

Lessons: a court found from lines alone is very precise where it works (5 cm) but can be confidently wrong on a new
arena (C05); training the keypoints on synthetic courts with **random** measurements, not the three official
templates, is what makes them generalise (0.65 → 0.28 m); and the court type must come from a person (chosen
automatically it was wrong on 4 of 6 clips, and a wrong type still looks plausible while every distance is off).
For identity the court is an **output**, not a fix: 62 % of the swaps happen between players less than 0.5 m apart
(section 14).

- [x] **C01** Public court-keypoint dataset audited (850 NBA frames, CC BY 4.0): court template verified to ~1–2 px, split leakage and clamped labels found and fixed
- [x] **C02** Court-keypoint detector (YOLO26n-pose, CPU) — 0.27 m on NBA test frames; does not transfer to SportsMOT (documented)
- [x] **C03** Court templates per league (NBA / NCAA / FIBA) + line-based refinement — fixes close starts only
- [x] **C04** Hand anchors on 3 frames + propagation → homography for all 1,162 frames of the NCAA clip, validated by two independent chains (5–20 cm)
- [x] **C05** No annotation: court homographies from straight lines + physics filter, confirmed by two chains, league picked automatically; on unseen FIBA arenas it can be confidently wrong (found in 009)
- [x] **C06** Learned court-area segmentation, trained on many arenas + synthetic courts: IoU 0.82–0.94 on unseen arenas, never catastrophically wrong; the court mask of the best configuration
- [x] **C07** One camera per clip: a homography no fixed broadcast camera can produce is rejected; clips without a trustworthy camera go to review instead of a confident wrong court
- [x] **C08** Top-down map in the pipeline (`hooptrack map`): tracks in metres, distances and speeds — 5.7 cm from the hand-made court on NCAA
- [x] **C09** Learned court keypoints (YOLO26n-pose, 33 named points) trained on synthetic courts with random measurements + the court type given by the user: **0.28 m median error, 92 % of the frames < 0.5 m**, every validation clip covered

## What this project is

Keeping track of **who is who** in a basketball broadcast is harder than street
pedestrian tracking: team-mates wear identical jerseys, players occlude each
other constantly, the camera pans with the ball and players leave the frame
and come back. On top of that, a generic person detector also sees the crowd,
the bench and the referees.

This repository is an end-to-end pipeline — detection → filtering →
re-identification → tracking → evaluation — whose goal is **never to give a
player someone else's identity, and to give it back when he returns.**

It is also a record of *how* it was built: each improvement starts from a
measured weakness of the previous run, is added behind a config switch, and is
kept only if the numbers say so. Nothing is deleted. Every result can be
reproduced, including the ideas that failed.

## Results, step by step

All numbers on SportsMOT **validation** clips (never used for training or tuning), laptop CPU.
IDF1 measures how well identities are kept, MOTA how many errors there are (false positives, misses,
ID switches). "OVERALL" means the metrics recomputed over all clips together, not an average.

### 1 · One clip: the tracker built from scratch (001–007)

NCAA clip `v_00HRwkvvjtQ_c001` (1162 frames, **10 players**), YOLO26m with COCO weights (no training).

| # | Change | IDF1 ↑ | MOTA ↑ | ID switches ↓ | Identities | False positives ↓ | FPS ↑ |
|---|---|---|---|---|---|---|---|
| [001](experiments/001_baseline_yolo26m_coco/) | Baseline: YOLO26m (COCO) + tracker, no tuning | 37.8 | −27.7 | 49 | 123 | 13,725 | 0.49 |
| [002](experiments/002_reid_only_usable_boxes/) | Run ReID only on boxes the tracker can use | 37.8 | −27.7 | 49 | 123 | 13,725 | 1.75 |
| [003](experiments/003_court_mask_filter/) | **Court mask** (floor colour): drop people whose feet are off the court | 56.6 | 68.0 | 48 | 58 | 2,941 | 2.38 |
| [004a](experiments/004a_component_ablation/) | Ablation: court mask + motion + camera compensation, no ReID | 58.2 | 69.1 | 36 | 50 | 2,909 | **52** ¹ |
| [005](experiments/005_make_reid_earn_its_place/) | **ReID as a soft cost, not a veto** (+ identity memory) | 60.5 | 68.0 | 34 | 42 | 2,941 | 2.9 |
| [006](experiments/006_team_classifier/) | **Team / referee colour classifier**: hide referees | 66.9 | 90.1 | 34 | 42 | 390 | 2.5 |
| [007](experiments/007_referee_leftovers/) | Offline mode: remove referee tracks after the whole video ² | **67.7** | **92.2** | **33** | **42** | **142** | 2.4 |

<sub>¹ detection cached; without ReID the tracker itself runs at 52 FPS on CPU. ² analytics mode (`--set teams.offline_suppress_other=true`).</sub>

- **+29.9 IDF1 and +119.9 MOTA without training anything.** The baseline's problem was *who* was detected:
  half of the confident boxes were spectators, bench and photographers (court mask), most of the rest referees
  (colour classifier).
- **ReID had to earn its place**: as a hard veto it made things worse (004a); as a soft cost, calibrated on
  the train split, it beat the motion-only tracker (005).
- **A good idea that failed**: forbidding matches between the two teams fixed none of the cross-team swaps — they
  all happen inside occlusions, where the hidden jersey cannot be seen (006).

<div align="center">
<img src="assets/compare_baseline_best.gif" width="960" alt="Side by side: the baseline boxes spectators, bench and referees with ever-growing ids; the 007 configuration boxes only the ten players">
<br><sub>Left: baseline (001). Right: 007 — court mask, soft ReID with memory, referee filter. False positives 13,725 → 142.
▶ <a href="assets/compare_baseline_best.mp4">side-by-side clip</a> · step by step: <a href="experiments/003_court_mask_filter/">003</a> · <a href="experiments/006_team_classifier/">006</a> · <a href="experiments/007_referee_leftovers/">007</a></sub>
</div>

### 2 · The honest test: other arenas

Everything above was tuned on **one** clip. Measured on clips from the other validation arenas
(three FIBA games: Tenerife 2018, Istanbul 2010, Changsha 2015), the same configuration falls apart. The
floor-colour mask lets the bench and crowd through in some arenas and cuts real players in others; the
referee classifier switches itself off when it cannot find three clear colour groups.

| 4 clips, one per validation arena | IDF1 | MOTA | ID switches | False positives | Missed players |
|---|---|---|---|---|---|
| Configuration of section 1 (006, live) | 45.0 | 55.6 | 107 | 6,897 | 4,169 |

One clip per arena from here on: NCAA `c005`, Tenerife `c002`, Istanbul `c012`, Changsha `c001`,
with `scripts/run_court_source_eval.py` (detections cached once, one folder per run, per-arena tables).

### 3 · A court mask that works in any arena (C06 → 010)

Rules about the floor colour cannot cover every arena, so the court area is **learned**: YOLO26n-seg trained
on 15 French arenas (DeepSportRadar), NBA broadcasts, SportsMOT train clips and synthetic courts with random
floors, paint and cameras (domain randomisation). On the validation arenas it overlaps the true court by
IoU 0.82–0.94 and, unlike the court homography (C05), it is never catastrophically wrong.
▶ [C06](experiments/court/C06_court_segmentation/)

| 4 arenas | IDF1 | MOTA | ID switches | False positives | Missed players |
|---|---|---|---|---|---|
| Floor-colour mask (003) | 45.0 | 55.6 | 107 | 6,897 | 4,169 |
| **Learned court mask (C06)** | **53.1** | **76.8** | 107 | **3,989** | **1,732** |

<sub>Tenerife: MOTA 54.8 → 90.3 · Istanbul: 5.2 → 48.9. ID switches do not change: the mask decides <i>who</i> enters the tracker, not how identities are assigned.</sub>

<div align="center">
<img src="assets/010/compare_colour_vs_c06_tenerife.gif" width="800" alt="Side by side on the Tenerife arena: the colour mask lets bench people through; the learned court mask follows the court lines">
<br><sub>FIBA Tenerife, same detector and tracker. Left: floor-colour mask. Right: learned court mask (green polygon).
▶ <a href="assets/010/compare_colour_vs_c06_tenerife.mp4">20-second clip</a></sub>
</div>

### 4 · A detector that sees players, not people (011)

The COCO detector's class is *person*: everyone is detected, and sections 1–3 remove the non-players with
rules. The strong trackers on SportsMOT do the opposite: they fine-tune the detector. SportsMOT labels only
the ten players, so referees, bench and crowd are background during training, and the detector learns the
difference from the whole image. YOLO26m, 40 epochs on a Colab T4, trained on 3 games
([notebook](notebooks/colab_train_player_detector.ipynb)). On the 4 unseen games: false positives per frame
**8.4 → 0.17**, missed players per frame 0.81 → 0.19. ▶ [011](experiments/011_player_detector/)

| 4 arenas | IDF1 | MOTA | ID switches | False positives | Missed players |
|---|---|---|---|---|---|
| COCO detector + C06 mask (section 3) | 53.1 | 76.8 | 107 | 3,989 | 1,732 |
| Players-only detector + C06 mask | 73.7 | 93.6 | 50 | 324 | 1,226 |
| **… without the referee classifier (011b)** | **74.6** | **96.0** | **49** | 397 | **551** |

- **False positives ÷10, ID switches halved** without touching the tracker: with fewer people in the image,
  each track has fewer wrong candidates.
- **The referee classifier became harmful.** It hides the smallest of three colour groups; with no referees
  left in the detections, it was hiding real players (NCAA: 787 → 139 missed players without it). It stays
  behind its switch, off in the best configuration ([`configs/player.yaml`](configs/player.yaml)).
- The C06 mask still helps on top of the new detector (+1.9 IDF1): the colour mask now mostly cuts real players.

<div align="center">
<img src="assets/011/compare_coco_vs_player_istanbul.gif" width="800" alt="Side by side on the Istanbul arena: the COCO detector tracks referees and staff on the court; the players-only detector boxes only the players">
<br><sub>FIBA Istanbul, same court mask and tracker. Left: COCO detector (referees and staff get identities). Right: players-only detector.
▶ <a href="assets/011/compare_coco_vs_player_istanbul.mp4">20-second clip</a></sub>
</div>

### 5 · A ReID model trained on players (012) — better model, same tracking

The appearance model was a ResNet-50 trained on ImageNet; for it, two team-mates are "a person in a white
shirt". An **OSNet ReID model fine-tuned on SportsMOT player crops** (the main ingredient of Deep-EIoU, the
reference tracker on SportsMOT; [notebook](notebooks/colab_train_reid_osnet.ipynb)) finds the same player
among his team-mates twice as well on the unseen games: rank-1 **0.42 → 0.71**, mAP 0.19 → 0.41.
Its distances live on another scale, so the tracker's thresholds were recalibrated on the train split.

| 4 arenas | IDF1 | MOTA | ID switches | Identities per clip |
|---|---|---|---|---|
| ResNet-50 ImageNet (011b) | 74.6 | 96.0 | 49 | 17–21 |
| OSNet fine-tuned, calibrated | 72.9 | 96.1 | 47 | 18–22 |

**Neutral, and the diagnosis says why.** Classifying every identity break: two thirds happen inside
crossings, where by design the tracker uses motion only (the crop mixes two players), and the memory that
should bring lost players back never fires. The bottleneck is how the tracker uses appearance, not the
model. ▶ [012](experiments/012_reid_osnet/)

### 6 · Joining each player's fragments after the video (013)

Seeing the whole video at once, every id is a *tracklet*. Two tracklets that are never on screen together
and look alike (OSNet, median over many clean crops) are the same player and get one id; if a team-mate on
screen at the same time looks almost as alike, nothing is merged. This is the *connector* of GTA (Global
Tracklet Association), the step that lifts Deep-EIoU from 77 to 81 HOTA on SportsMOT. Thresholds come
from the train-split calibration. It post-processes a finished run in minutes, without re-tracking.
▶ [013](experiments/013_tracklet_stitching/)

| 4 arenas | IDF1 | MOTA | ID switches | Identities per clip |
|---|---|---|---|---|
| Online tracker (011b) | 74.6 | 96.0 | 49 | 17–21 |
| + our first, conservative stitching rules | 77.2 | 96.1 | 43 | 16–19 |
| + original GTA code (connector, published settings) | 80.5 | 96.1 | 36 | 11–12 |
| **+ GTA's connector rebuilt here, 12 clean crops per tracklet (013b)** | **81.3** | **96.1** | **34** | **11–12** |

<sub>Per clip IDF1: NCAA 68.9 → 77.2 · Tenerife 69.1 → 73.9 · Istanbul 69.5 → 72.9 · Changsha 85.6 → 95.1.
The rebuilt connector gives exactly the original code's numbers when fed the same embeddings. Describing each
tracklet by clean crops (no other player covering them) instead of every box is better and 10× cheaper.
GTA's splitter added nothing on these clips.</sub>

Our first version refused every doubtful merge, and that was too cautious. The rule that matters is
simpler: two tracklets on screen at the same time are two people. Merging the most similar remaining pair
first, again and again, brings the identities close to one per player.

### 7 · A looser motion match: Expansion IoU (014) — negative

Deep-EIoU, the reference tracker on SportsMOT, matches tracks and detections with boxes grown by 0.7× their
size on each side, so a player who changes direction is not lost. Added to our Kalman tracker with Deep-EIoU's
own settings, it fixes nothing (short losses 18 → 20) and almost doubles the swaps in crossings: grown boxes of
players standing side by side overlap, so neighbours become candidates. It stays in the code, off.
▶ [014](experiments/014_expansion_iou/)

| 4 arenas | IDF1 | MOTA | ID switches |
|---|---|---|---|
| Online tracker (011b) | **74.6** | **96.0** | **49** |
| + Expansion IoU (014) | 68.6 | 95.8 | 81 |
| 011b + stitching (013b) | **81.3** | **96.1** | **34** |
| 014 + stitching | 72.5 | 95.8 | 76 |

### 8 · Swaps in crossings: where the limit is (015–017)

Every identity change that happens while a player stays visible happens in a crossing (015). Cutting the
tracklets at every crossing and re-joining them makes things worse (IDF1 64.3). A local keep-or-swap check
is right only 1 time in 8. A **team-mate benchmark** (016) explains why. With clean ground-truth crops, OSNet
takes the right decision at 97 % of team-mate crossings, but real swaps are only ~4 % of crossings. A 3 % error
rate then creates about as many false swaps as it fixes. Training OSNet on 4× more people, with football and
volleyball added (017), improves retrieval (rank-1 0.71 → 0.75) but not the team-mate separation (AUC 0.71).
The stitching stays at ~81. ▶ [015](experiments/015_crossings/) · [016](experiments/016_teammate_reid_benchmark/) · [017](experiments/017_reid_multisport/)

| team-mate benchmark, 15 val clips | AUC same player vs team-mate | right decision at team-mate crossings |
|---|---|---|
| ResNet-50 ImageNet (online tracker) | 0.558 | 77 % |
| OSNet fine-tuned on basketball (012) | 0.713 | 97 % |
| OSNet fine-tuned on 3 sports (017) | 0.715 | 97 % |

### 9 · Giving stolen ids back (018–019)

A part-based ReID model (one vector per body stripe, 018) is worse than plain OSNet: team-mate AUC 0.713 → 0.668,
and its stripes do not specialise. The real problem turned out to be elsewhere. After stitching, **46 identity
changes are ids stolen in a crossing**: player A's track goes on with player B, and A's id cannot come back while
it is on screen, on B. **36 of the 46 thieves play for the other team**, the case OSNet separates at 100 %. 019 cuts a
tracklet where its team changes for at least 1 s on both sides (team from OSNet, tracklets clustered), then stitches
as before. ▶ [018](experiments/018_reid_parts/) · [019](experiments/019_team_split/)

| 4 arenas | IDF1 | MOTA | ID switches | IDF1 NCAA / Tenerife / Istanbul / Changsha |
|---|---|---|---|---|
| 013b: stitching | 81.3 | 96.1 | **34** | 77.2 / 73.9 / 72.9 / 95.1 |
| **019: team cut + stitching** | **83.5** | 96.0 | 52 | **83.8 / 75.4 / 74.8** / 95.0 |
| ceiling of 019 (ground-truth teams) | 89.1 | 96.4 | 22 | 81.5 / 90.7 / 81.0 / 98.7 |

**On all 15 validation clips the gain disappears** (IDF1 013b 79.9, 019 79.7; 11 clips never used before: 4 better,
5 worse). The unsupervised team labels are not reliable enough everywhere, so 019 stays off and 013b remains the offline
mode. With ground-truth teams the cut would be worth +7.8 on the 4 clips: the idea is right, the labels are the limit.

### 10 · Team labels chosen on train, checked on val (020)

The labeller is now chosen on the **train** clips and checked once on val. A benchmark on ground-truth tracks measures
accuracy, false cuts (a ground-truth track is one player) and how many simulated stolen ids each labeller finds. Two
findings. With the players-only detector there are no referees left, so 019's **third cluster ("other") eats 16–22 % of real
players**: k = 2 is better for every descriptor. And **SigLIP on the torso** (a web-scale image model, never fitted to our
clips) describes the kit, while OSNet describes the player: accuracy 0.975 against 0.56, 2 false cuts in 15 clips, 95 % of
the stolen ids found. ▶ [020](experiments/020_team_labels/)

| 15 val clips | IDF1 | MOTA | ID switches |
|---|---|---|---|
| 013b: stitching | 79.9 | 96.7 | **214** |
| 019: team cut, OSNet labels | 79.7 | 96.6 | 289 |
| **020: team cut, SigLIP torso labels (k = 2)** | **81.8** | 96.6 | 267 |
| ceiling: ground-truth teams, same boxes | 82.0 | 96.6 | 276 |

The labels are now at the oracle's level. **021** splits the errors that are left. The biggest is the connector joining
a clean piece to the wrong player (6.7 % of boxes, 42 of 112 merges wrong, 32 of them between team-mates). Next come swaps
between team-mates that no team cut can see (4.6 %). Where the cut lands costs only 2.1 %. Forbidding joins between two kits
is neutral (81.8 → 81.8): the blocked piece joins the wrong team-mate instead. What is left needs a cue that tells
team-mates apart. ▶ [021](experiments/021_stitch_errors/)

Jersey numbers are that cue, but they must be read first. A generic text recogniser reads nothing (022). Florence-2, a
small vision-language model, reads them with no training: 96 % of the reads along one player agree, and 36 of 40 are right
by eye. Forbidding joins between two different numbers is still neutral (81.8 → 81.7): only 22 % of the pieces get a number,
and only 2 of the 37 wrong joins have a number on both sides. ▶ [022](experiments/022_jersey_feasibility/) · [023](experiments/023_jersey_numbers/)

### 11 · HOTA, and a published tracker on the same boxes (024–026)

From here on, every run is also measured with **HOTA** (official TrackEval), the metric SportsMOT ranks trackers by.
**Deep-EIoU** (WACV-W 2024), the reference tracker on SportsMOT, was run with *our* detections and *our* OSNet:
online it beats our tracker by 5 HOTA, and our offline step on top of it gives the best numbers of the project. The
offline step is not tuned to our tracker's errors: on Deep-EIoU's output it still adds +3.1 HOTA, on 12 of 15 clips.
Deep-EIoU has no licence, so it cannot be shipped here; an ablation (025) found what makes it better (appearance
fused as the *minimum* of two distances, a much larger box expansion, the last detected box instead of a motion
prediction), but the same pieces ported into our tracker one by one make it worse (026): they only work together.
▶ [024](experiments/024_hota_deep_eiou/) · [025](experiments/025_deep_eiou_ablation/) · [026](experiments/026_deep_eiou_port/)

| 15 val clips | HOTA | AssA | IDF1 |
|---|---|---|---|
| 011b online (ResNet-50 ReID) | 68.8 | 55.1 | 72.7 |
| 011b online with OSNet (026) | 69.2 | 55.8 | 73.6 |
| 020: team cut + stitching | 75.3 | 66.1 | 81.8 |
| Deep-EIoU online, same detections and OSNet | 74.3 | 63.5 | 79.6 |
| Deep-EIoU + our offline step (not shippable) | 77.4 | 69.0 | 83.8 |

### 12 · A ReID model that tells team-mates apart (027–029)

The image towers of SigLIP and CLIP, used with no training, are worse than OSNet at telling a player from his
team-mates (027). Fine-tuned in Colab on the same player crops as OSNet (028, the idea of CLIP-ReIdent), SigLIP
becomes far better: on the team-mate benchmark of 016 it beats OSNet on all 15 validation clips, and takes the right
decision at 98.8 % of the crossings between team-mates. It is ~12 times heavier than OSNet, so it is used only
offline, to compare pieces in the stitching, with its threshold transferred on the train split (029). Same online
run, same number of joins: only *which* pieces are joined changes. **Adopted: the best result of the project with
code and models that are ours.** ▶ [027](experiments/027_siglip_clip_reid/) · [028](experiments/028_siglip_reid_finetune/) · [029](experiments/029_stitch_siglip_reid/)

| | AUC same player vs team-mate | right at team-mate crossings | HOTA (15 clips) | IDF1 |
|---|---|---|---|---|
| OSNet (012), stitching of 020 | 0.713 | 96.9 % | 75.3 | 81.8 |
| **SigLIP fine-tuned (028), stitching of 029** | **0.860** | **98.8 %** | **76.9** | **83.9** |

### 13 · The swaps that are left, and why (030–033)

A switch report classifies every identity change against the ground truth, per minute
([`scripts/switch_report.py`](scripts/switch_report.py)). It shows what the metrics hide: **~31 identity switches
per minute are left, as in the online run.** The offline step halves the fragments (a player coming back under a
new id: 101 → 52) but not the swaps; 92 % of the switches happen within a second of two players overlapping, and
most are two players exchanging ids. Two checks at the exit of every crossing with the fine-tuned SigLIP made things
worse (030, 031). A diagnosis (032) found why: the model is right 99.4 % of the time where it can look, but **83 %
of the swaps happen in crowded crossings where no player has a clean view** before or after (only 19 % of the
crossings can be decided at all), so any check around the crossing can reach at most ~15 % of them. The identity has
to come back later, from something that does not change, like the jersey number: a digit-by-digit reader trained on
NBA footage (033, E-BARD dataset) is right 94 % of the time on NBA clips, but on these NCAA and European clips it
reads a number on only 32 % of the pieces, and its reads often contradict each other: neutral (HOTA 76.9 → 76.7).
▶ [030](experiments/030_crossing_check/) · [031](experiments/031_group_crossing_check/) · [032](experiments/032_crossing_check_diagnosis/) · [033](experiments/033_jersey_number_detector/)

| 15 val clips (8.4 min) | HOTA | IDSW | switches / min | fragments | to a team-mate's id | to an opponent's id |
|---|---|---|---|---|---|---|
| 011b online | 68.8 | 278 | 31.9 | 101 | 25 | 141 |
| **029** | **76.9** | 278 | 31.3 | 52 | 34 | 176 |
| 030 / 031: check at crossing exits | 67.8 / 67.2 | 302 / 321 | 34.8 / 37.0 | | | |
| 033: + jersey numbers (NBA reader) | 76.7 | 277 | 31.7 | | | |

### 14 · The players on the court, in metres (034)

The court geometry of chapter 2 now covers **all 15 validation clips** (C09 learned keypoints, no annotation: only
the court type, read by a person from one frame per clip), so the best run can be followed on the court plan:
positions in metres, distance run and speed per identity (`hooptrack map`). Speeds land where they should
(2.1 m/s mean, 5.4 m/s at the 95th percentile). Could the court also fix the swaps? A diagnosis on the ground
truth says mostly not: **62 % of the swaps happen between players less than 0.5 m apart on the court**, in body
contact, where the feet of the player behind are hidden and his position is as uncertain as in the image; where
players are more than 1 m apart, the tracker already keeps them apart in over 97 % of the crossings. So the court
is integrated as an output, not as a second motion model in the tracker. A speed limit on the court in the
stitching (refuse joins that would need more than 9 m/s) is neutral too: the court says where the players are, not
who they are. ▶ [034](experiments/034_court_crossings/) · [035](experiments/035_stitch_speed_rule/)

<div align="center">
<img src="assets/034/map_029_fiba_c003.gif" width="560" alt="FIBA clip: tracked players on the broadcast with the projected court lines, and below the court plan with each identity's position and trail">
<br><sub>029 on a FIBA clip with a median number of switches, broadcast and court plan. ▶ <a href="assets/034/map_029_fiba_c003.mp4">20 s in full quality</a></sub>
</div>

### 15 · Exactly 10 players (036–038)

A game has 10 players on the court, and every clip has exactly 10 ground-truth identities; the best run so far
ended with 11–14. Giving the extra ids back to the 10 longest ones failed (036): after the online swaps a stitched
id holds pieces of several players, so every extra id is on screen together with all of them. Built instead from
pieces cut at the crossings (one player each), with 10 seeds taken from a frame where 10 pieces are on screen
together and every other piece given to the closest player that is not on screen at the same time, the closed set
wins big on the NCAA game and loses on FIBA (037): cutting at every crossing throws away the online tracker's
continuity. Adding it back (a piece stays with the player of its online neighbour unless its appearance clearly
prefers another, 038) gives **the best result of the project, better on 11 of 15 clips**. It makes more switches
than 029, but shorter ones: the identity comes back after a wrong piece.
▶ [036](experiments/036_roster_of_ten/) · [037](experiments/037_closed_set_of_ten/) · [038](experiments/038_closed_set_continuity/)

| 15 val clips | HOTA | AssA | IDF1 | IDSW | switches / min |
|---|---|---|---|---|---|
| 029: team cut + SigLIP stitching | 76.9 | 69.1 | 83.9 | **278** | **31.3** |
| 037b: closed set of 10, appearance only | 74.5 | 65.4 | 81.7 | 612 | 71.6 |
| **038: closed set of 10 + online continuity** | **79.7** | **74.9** | **89.6** | 379 | 43.7 |

### 16 · What limits identity, measured (042–046)

Where do 038's errors come from? Every wrong box was traced back to its piece
([`identity_error_breakdown.py`](scripts/identity_error_breakdown.py)): **7.2 % of the boxes** carry a wrong identity,
and only 27 % of those come from the online tracker (a swap inside a piece). **73 % are clean pieces given to the
wrong player**: 37 % to a team-mate, 36 % to an opponent. Four ways to decide those pieces better, each against
that diagnosis:

| 15 val clips | HOTA | IDF1 | what it says |
|---|---|---|---|
| **038 (adopted)** | **79.7** | **89.6** | |
| 042 soft kit term (kit labels right on 96 % of long pieces) | 77.8 | 86.3 | the opponent errors are not a kit problem |
| 043 every piece re-decided at once (exact integer program) | 78.6 | 88.2 | the exact optimum is no better than the greedy one: **the evidence is the limit, not the algorithm** |
| 046 ceiling: a *perfect* crossing verifier (ground truth) | 77.9 | 84.6 | even perfect links do not help this assignment, so mask propagation (SAM2) was not built |
| 044 jersey numbers: pose torso crop + PARSeq, zero-shot | — | — | 37 % of the players get a number at 91 % (train): too few for pieces; val not used |

The appearance distance between team-mates overlaps the distance between two crops of one player, and no rule,
solver or kit cue on top of it changes that. The next step needs **new evidence**: a jersey reader fine-tuned on
numbers from many games, or a learned association (CAMELTrack-style), both limited by the 3 basketball games of
SportsMOT train.
▶ [042](experiments/042_soft_team_cost/) · [043](experiments/043_global_reassignment/) ·
[044](experiments/044_jersey_parseq/) · [046](experiments/046_crossing_links_ceiling/)

Full log, raw metrics and configs: **[`experiments/`](experiments/)**.

## How it works

```mermaid
flowchart LR
    F["Frame"] --> D["YOLO26m<br/>player detector"]
    D --> C["Court mask<br/>learned court area (C06)"]
    F --> C
    C --> T["Online tracker<br/>Kalman + ReID + memory"]
    T --> S["Offline identities<br/>pieces cut at crossings →<br/>exactly 10 players (038)<br/>+ body smoothing (048)"]
    F --> B["Ball<br/>WASB fine-tuned +<br/>one-ball tracker (B08)"]
    S & B --> P["Possession<br/>who has the ball (P04)"]
    S --> M["Court plan<br/>positions in metres (C08/C09)"]
    S & B & P & M --> V["Demo video<br/>showcase / honest"]
    S --> E["Evaluation<br/>HOTA · IDF1 · switches"]
```

| Stage | What it does | Code |
|---|---|---|
| **Detection** | YOLO26m fine-tuned on SportsMOT players (or COCO `person`), low confidence threshold (0.10) so occluded players are not lost. Run once and cached. | [`detection/detectors.py`](src/hooptrack/detection/detectors.py) |
| **Court mask** | Keeps a box only if its feet are on the court. Three interchangeable sources (`filters.court_source`): floor colour (003), court homography (009), learned court-area segmentation (C06, best). | [`detection/filters.py`](src/hooptrack/detection/filters.py) · [`integrations/court_filter.py`](src/hooptrack/integrations/court_filter.py) |
| **Re-identification** | ResNet-50 / OSNet / colour embeddings, computed only for boxes the tracker can actually use. Offline, a SigLIP fine-tuned for players (028) compares the pieces to join. | [`reid/extractors.py`](src/hooptrack/reid/extractors.py) · [`reid/siglip.py`](src/hooptrack/reid/siglip.py) |
| **Motion model** | Constant-velocity Kalman filter in (cx, cy, w, h), warped by the camera's motion estimated with optical flow on the background. | [`tracking/kalman.py`](src/hooptrack/tracking/kalman.py) · [`gmc.py`](src/hooptrack/tracking/gmc.py) |
| **Association** | Hungarian matching in a cascade: confident boxes with motion + appearance, weak boxes with motion only (ByteTrack idea), then re-identification of lost players. | [`tracking/tracker.py`](src/hooptrack/tracking/tracker.py) |
| **Identity memory** | Players who leave the frame become *dormant* instead of being deleted, and can be revived by appearance, with an ambiguity margin that prefers a new id over a wrong one. | [`tracking/memory.py`](src/hooptrack/tracking/memory.py) |
| **Court plan** | Feet projected to metres with verified court homographies: trajectories, distances, speeds. | [`integrations/topdown.py`](src/hooptrack/integrations/topdown.py) |
| **Offline identities** | Online tracklets cut at every crossing; the pieces are given to exactly 10 players (seeds from a frame with all 10 on screen, then the closest player by fine-tuned SigLIP appearance that is not on screen at the same time, keeping the online tracker's continuity); short P → Q → P excursions on one box chain go back to P. | [`tracking/stitch.py`](src/hooptrack/tracking/stitch.py) · [`stitch_tracklets.py`](scripts/stitch_tracklets.py) · [`smooth_body_labels.py`](scripts/smooth_body_labels.py) |
| **Ball** | WASB (multi-frame heatmap network, BMVC 2023) fine-tuned on 1,245 hand clicks, then one ball through the clip by dynamic programming with physics-limited moves. | [`ball/track.py`](src/hooptrack/ball/track.py) · [`scripts/ball/`](scripts/ball/) |
| **Possession** | The closest player within reach, moving with the ball; short gaps filled, the last state kept while the ball is not seen. | [`ball/possession.py`](src/hooptrack/ball/possession.py) |
| **Demo video** | Broadcast-style drawing (`--mode showcase`) or every box, raw id and identity switch shown (`--mode honest`). | [`p06_render_demo.py`](scripts/possession/p06_render_demo.py) |

The tracker is **not** `model.track()`: it is written from scratch so that every
decision is explicit and can be switched off for an ablation. It borrows ideas
from ByteTrack (two-pass association) and BoT-SORT (Kalman in w/h, camera
compensation, appearance fusion). More: [`docs/architecture.md`](docs/architecture.md) ·
the failures behind each choice: [`docs/design_decisions.md`](docs/design_decisions.md).

## How it is developed

The repository follows a few rules ([`experiments/README.md`](experiments/README.md)):

1. **Diagnose before changing.** [`scripts/analyze_detections.py`](scripts/analyze_detections.py)
   separates detector errors from tracker errors; [`scripts/run_court_source_eval.py`](scripts/run_court_source_eval.py)
   measures one change at a time on several arenas.
2. **Additive changes only.** Every improvement is a new module or config key;
   its default in code reproduces the previous behaviour, so old configs still
   give old numbers.
3. **One experiment = one folder** with the hypothesis, the exact config,
   raw metrics and a conclusion, including what did *not* work.
4. **No leakage.** Training uses SportsMOT *train*; all reported numbers come from the
   *validation* games, which are never used to train or to pick a model.

## Roadmap (chapter 1 · tracking)

- [x] End-to-end pipeline: detection, ReID, tracker, memory, evaluation (IDF1, HOTA, custom re-entry metric)
- [x] **001–007** Baseline, selective ReID, colour court mask, ablation, soft ReID + memory, referee classifier, offline mode
- [x] **008** Court border — no gain (documented)
- [x] **009** Court mask from the court homography — helps only where the automatic homography is right (documented)
- [x] **010** Evaluation on several arenas + learned court mask (C06): IDF1 45.0 → 53.1, MOTA 55.6 → 76.8
- [x] **011** Players-only detector (fine-tuned YOLO26m): IDF1 → 74.6, MOTA → 96.0 on 4 arenas
- [x] **012** Player ReID: OSNet fine-tuned on SportsMOT crops (rank-1 0.42 → 0.71); neutral in the online tracker (diagnosed)
- [x] **013** Offline tracklet stitching by appearance (GTA's connector, rebuilt and verified): IDF1 74.6 → 81.3, 11–12 identities for 10 players
- [x] **014** Expansion IoU (Deep-EIoU) — negative: more swaps in crossings, IDF1 74.6 → 68.6 (documented, off)
- [x] **015** Swap repair at crossings — negative (cut + reconnect 64.3; keep/swap check 1 of 8 right)
- [x] **016** Team-mate ReID benchmark (same player vs team-mate vs opponent, and around crossings)
- [x] **017** OSNet trained on basketball + football + volleyball — neutral (retrieval +3.5, team-mates unchanged)
- [x] **018** Part-based ReID (OSNet + 4 stripes) — negative (team-mate AUC 0.713 → 0.668)
- [x] **019** Give stolen ids back: cut tracklets where they change team — +2.2 on 4 clips, no gain on 15 (79.9 → 79.7)
- [x] **020** Team labels chosen on train: SigLIP on the torso, k = 2 — 15 val clips 79.9 → 81.8 (at the oracle-label ceiling, 82.0)
- [x] **021** Error breakdown after 020 + "never join two kits" rule in the connector — neutral (81.8 → 81.8)
- [x] All 15 validation clips: IDF1 72.7 online, 79.9 with stitching (013b), 81.8 with the team cut (020)
- [x] **022** Jersey numbers, zero-shot (PARSeq) — not readable without a model trained on jersey crops (documented)
- [x] **023** Jersey numbers read zero-shot by Florence-2 — right when it answers (36/40 by eye), but only 22 % of pieces get a number: neutral (81.8 → 81.7)
- [x] **024** HOTA on all runs + Deep-EIoU on the same detections: our offline step adds +3.1 HOTA on its output too (77.4, not shippable)
- [x] **025** Deep-EIoU ablation: min-distance fusion, expansion, last detected box (diagnostic)
- [x] **026** Those pieces ported into our tracker — negative (they only work together; documented, off)
- [x] **027** SigLIP / CLIP as ReID with no training — worse than OSNet (documented)
- [x] **028** SigLIP fine-tuned as player ReID (Colab): team-mate AUC 0.713 → 0.860, better on 15 of 15 clips
- [x] **029** Fine-tuned SigLIP in the stitching: **HOTA 75.3 → 76.9, IDF1 81.8 → 83.9** (adopted)
- [x] **030 / 031** Identity check at crossing exits (pairs / groups) — negative (documented, off)
- [x] **032** Diagnosis: 83 % of the swaps happen where no clean view exists; checks at crossings can reach ~15 %
- [x] **033** Jersey-number reader trained on NBA footage (E-BARD) — 32 % of pieces read on these clips, unreliable: neutral
- [ ] Jersey numbers with labels from many more games
- [x] **034** Players on the court in metres on all 15 clips; diagnosis: 62 % of the swaps are between players < 0.5 m apart, so the court is an output, not a tracker input
- [x] **035** Stitching that refuses joins needing an impossible speed on the court — neutral (documented, off)
- [x] **036** Roster of 10 players (5 per team), leftover ids given back by appearance — negative: stitched ids are not pure enough for the rule (documented, off)
- [x] **037** Closed set of 10 players from pieces cut at crossings — +1 to +15 HOTA on 5 of 6 NCAA clips, worse on FIBA, 74.5 overall (documented, off)
- [x] **038** Closed set + the online tracker's continuity: **HOTA 76.9 → 79.7, IDF1 83.9 → 89.6**, better on 11 of 15 clips (adopted)
- [x] **039** Smoothing single wrong pieces after 038 — nothing to smooth: errors are runs of pieces (diagnosis, off)
- [x] **040** Per-clip embedding adaptation, 5 per team, at most 10 ids on top of 038 — all worse than 038 (documented, off)
- [x] **041** Jersey reader + hand labels of SportsMOT's train clips — 98 % on held-out train clips, unchanged on val games (30 % read): negative
- [x] **042** Error breakdown of 038 (73 % of the errors: clean pieces given to the wrong player) + soft kit term — 77.8: negative (documented, off)
- [x] **043** All pieces re-decided at once with an exact integer program — 78.6: negative, the evidence is the limit (documented, off)
- [x] **044** Jersey numbers, pose-guided torso crop + PARSeq zero-shot — 37 % of the players at 91 % on train, gate out of reach, val not used: negative
- [x] **046** Ceiling of a perfect crossing verifier (ground truth) — 77.9, worse than 038: SAM2 verifier not built
- [x] **B01** COCO detector on the ball: 11 % found — negative
- [x] **B02–B04** Hand clicks (1,245 train / 390 test frames) + our YOLO26s at 1280: **58 % found at 76 % precision** (COCO: 11 % / 29 %) — adopted
- [x] **B05** Offline ball tracking (one ball, physics-limited moves, gaps filled): recall 58 → 73 %, precision 76 → 66 % — mixed
- [x] **B05b/c** Static look-alikes removed on the court plan + "no ball" for weak segments (values from train): **78 % found at 73 % precision**, F1 0.66 → 0.75 (adopted)
- [x] **P01** Possession by fixed rules (closest player within reach): 80 % of the marked catches / releases found within 1 frame, but 3× too many changes — negative per frame
- [x] **P01b** Short gaps filled, values from train — neutral
- [x] **P02** Air from the ball's own motion — negative; with a perfect ball P01 only reaches F1 0.46: the rules, not the ball, are the limit
- [x] **P03** Same possession rules on the B08 ball: event F1 0.41 → **0.50**, better on 4 of 4 clips — the ball was a limit
- [x] **P04** Frames without a ball keep the last state: **86 % of the frames right** (always held: 79 %), event F1 0.50 (adopted)
- [x] **P05** Possession: learned per-frame score + two-state temporal model — 87.9 % of frames but event F1 0.43 (over-smooths); the per-frame score alone: 90.6 %, not adopted (chosen after the test) — mixed
- [x] **B07** WASB (state of the art, NBA weights) zero-shot: 40 % found at **100 % precision** (ours: 78 % / 73 %) — complementary
- [x] **B08** WASB fine-tuned on our hand clicks + our tracker: 71 % found at **85 % precision**, F1 0.75 → **0.78**, better or equal on 4 of 4 clips (adopted)
- [x] **B09** WASB's and our YOLO's candidates in one tracker (weight chosen on train) — F1 0.775 → 0.778: null; the missed balls are on players' bodies (70 % found over one player, 38 % over two)
- [ ] Edge deployment: TensorRT on Jetson Orin Nano

## Quick start

```bash
git clone https://github.com/VaroLG/hoop-reid-tracker && cd hoop-reid-tracker
python -m venv .venv
source .venv/bin/activate              # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt    # pinned versions
pip install -e . --no-deps

python scripts/smoke_test.py           # end-to-end check on a rendered clip: no data, no GPU
pytest                                 # 175 tests
```

**Data.** SportsMOT is released under CC BY-NC 4.0 and is gated: accept the terms at
[huggingface.co/datasets/MCG-NJU/SportsMOT](https://huggingface.co/datasets/MCG-NJU/SportsMOT), then

```bash
hf auth login
python scripts/download_data.py --splits train val     # keeps only the basketball clips
```

**Best configuration** ([`configs/player.yaml`](configs/player.yaml)). It needs two trained models, which are not stored in git:
the player detector (`weights/Player_detector.pt`, [Colab notebook](notebooks/colab_train_player_detector.ipynb)) and the court
segmentation model (`outputs/court_runs/c06_yolo26n_seg/weights/best.pt`, [C06](experiments/court/C06_court_segmentation/)).

```bash
hooptrack track --config configs/player.yaml "data.only=[v_4r8QL_wglzQ_c001]" output.save_video=true --eval
hooptrack map   --run outputs/runs/player --video          # court plan (clips with a verified court)
```

**The final pipeline** (the video above), after the online run of the next block. Models not stored in git: the
fine-tuned SigLIP (`weights/siglip_reid_sportsmot`, [notebook](notebooks/colab_train_reid_siglip.ipynb)) and the fine-tuned
WASB (`weights/wasb_ft_b08.pt`, [notebook](notebooks/colab_finetune_wasb_ball.ipynb); WASB's code in `third_party/WASB-SBDT`).

```bash
# players: exactly 10 identities (038) + body smoothing (048)
python scripts/stitch_tracklets.py --run outputs/runs/player --config configs/player.yaml \
    --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot \
    --split-crossings --closed-set 10 --continuity 0.3 --out outputs/runs/player_038
python scripts/smooth_body_labels.py --run outputs/runs/player_038 --out outputs/runs/player_final
# ball (B08) and possession (P04), on the 4 clips with hand-clicked balls (scored against them)
python scripts/ball/b07_wasb_probe.py --weights weights/wasb_ft_b08.pt --players outputs/runs/player_038 \
    --out outputs/ball/dets_wasb_ft
python scripts/ball/b05_track_ball.py --dets outputs/ball/dets_wasb_ft --name b08 --save-tracks outputs/ball/tracks_b08
python scripts/possession/p01_possession.py --tracks outputs/ball/tracks_b08 --players outputs/runs/player_final \
    --set keep_state=true --out outputs/possession/p04
# metrics and the two videos
python scripts/eval_hota.py --config configs/player.yaml --runs final=outputs/runs/player_final
python scripts/possession/p06_render_demo.py --clip v_00HRwkvvjtQ_c001 --players outputs/runs/player_final \
    --out outputs/videos/showcase_c001.mp4
python scripts/possession/p06_render_demo.py --clip v_00HRwkvvjtQ_c001 --players outputs/runs/player_final \
    --mode honest --out outputs/videos/honest_c001.mp4
```

**Offline tracklet stitching** (013) on a finished run, with the OSNet ReID model (`weights/osnet_x1_0_sportsmot.pth`,
[notebook](notebooks/colab_train_reid_osnet.ipynb)); thresholds from `scripts/calibrate_reid.py` on the train split:

```bash
python scripts/stitch_tracklets.py --run outputs/runs/player --config configs/player.yaml \
    --set reid.type=osnet reid.weights=weights/osnet_x1_0_sportsmot.pth --out outputs/runs/player_stitched
```

**Previous offline mode** (029): cut tracklets where their team changes (SigLIP labels, k = 2) and stitch with the fine-tuned
SigLIP (`weights/siglip_reid_sportsmot`, [notebook](notebooks/colab_train_reid_siglip.ipynb); threshold from
`scripts/calibrate_stitch_threshold.py` on the train split). Measure with HOTA and the switch report:

```bash
python scripts/stitch_tracklets.py --run outputs/runs/player --config configs/player.yaml \
    --set reid.type=siglip reid.weights=weights/siglip_reid_sportsmot --thresh 1.0365 \
    --split-teams --team-source siglip --team-k 2 --out outputs/runs/player_029
python scripts/eval_hota.py --config configs/player.yaml --runs 029=outputs/runs/player_029
python scripts/switch_report.py --config configs/player.yaml --runs 029=outputs/runs/player_029
```

**Compare one change on several arenas** (detections cached once, resumable, per-arena tables, side-by-side videos):

```bash
python scripts/run_court_source_eval.py --config configs/player.yaml --variants colour segmentation --video \
    --clips v_00HRwkvvjtQ_c005 v_5ekaksddqrc_c002 v_BgwzTUxJaeU_c012 v_4r8QL_wglzQ_c001
```

`configs/default.yaml` keeps the configuration of section 1 (COCO detector, colour mask), which needs no training.
Any parameter can be changed from the command line, e.g. `--set filters.court_source=segmentation reid.type=osnet`.

## Repository layout

```
configs/default.yaml          every hyper-parameter, documented (section 1 configuration)
configs/player.yaml           online tracker of the final pipeline (011b)
src/hooptrack/
  detection/                  YOLO wrapper, detection cache, ground-truth oracle, colour court mask, team classifier
  reid/                       histogram, ResNet-50, OSNet, SigLIP / CLIP extractors, embedding cache, jersey readers
  tracking/                   kalman · matching · gmc · track · memory · tracker
  eval/                       motmetrics, TrackEval (HOTA), re-entry metric
  court/                      chapter 2: templates, homographies, clip camera, court segmentation (independent)
  integrations/               the only bridge: court mask filters for the tracker, top-down map, static-ball filter
  ball/                       ball labels, one-ball tracker, possession rules and temporal model
  pipeline.py                 per-video loop with per-stage latency
  cli.py                      hooptrack detect | track | evaluate | map
notebooks/                    Colab training: player detector (011), ReID (012, 028), jersey readers (033, 041), ball (B03, B08)
experiments/                  chapter 1: one folder per experiment: hypothesis, config, metrics, conclusion
  ball/ possession/           ball (B01–B09) and possession (P01–P05) experiments
  court/                      chapter 2: C01 … C09
scripts/                      download, datasets, multi-arena evaluation, diagnosis, ablation, calibration, stitching
  ball/ possession/           ball labelling, detection, tracking; possession, event labelling, demo video
  court/                      chapter 2: datasets, training, evaluation, homographies, overlay videos
tests/                        tracker, filters, court geometry, chapter independence
docs/                         architecture and design decisions
```

## Limitations

- **Swaps in crowded crossings** are the main identity error left. The best offline mode (038) gives 89.6 % of the
  boxes the right identity, but still makes ~44 identity switches per minute, mostly short pieces after a crowded
  crossing given to the wrong player; and the pieces that fit none of the 10 players (referees, false detections)
  keep their own id. 73 % of the remaining errors are clean pieces given to the wrong player, and neither kit
  cues, an exact solver nor perfect crossing links fix them (042–046); jersey numbers would need a reader
  fine-tuned on many games (033, 041, 044).
- **The ball over a player's body** is the ball's main miss (B09), which is also where possession is decided;
  possession still predicts about twice as many catches / releases as there are (P04, P05).
- **The showcase video hides nothing but shows little of the errors**: identity switches are short and happen in
  piles. The honest drawing (`--mode honest`) and the tables above are the reference.
- Results on all 15 validation clips (sections 10–16); the earlier sections use 4 clips, one per arena.
- **Few training games**: the player detector and the ReID model learn from the 3 basketball games of
  SportsMOT train. The detector generalises well to the 4 validation games; crowded rebounds are its
  remaining misses.
- CPU-only inference numbers; training on a free Colab GPU.

## Data and licence

Code: MIT. The SportsMOT data is **not** included or redistributed; it belongs
to its authors (CC BY-NC 4.0). DeepSportRadar images (C06) are CC BY-NC-SA 4.0. The figures and clips in this
README are short excerpts used for non-commercial research illustration.

```bibtex
@inproceedings{cui2023sportsmot,
  title     = {SportsMOT: A Large Multi-Object Tracking Dataset in Multiple Sports Scenes},
  author    = {Cui, Yutao and Zeng, Chenkai and Zhao, Xiaoyu and Yang, Yichun and Wu, Gangshan and Wang, Limin},
  booktitle = {ICCV},
  year      = {2023}
}
```
