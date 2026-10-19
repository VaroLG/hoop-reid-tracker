# Design decisions

Short records of the non-obvious choices, the alternatives considered and why
they were rejected. Each one came from a failure observed while testing.

---

## 1. Write the tracker instead of calling `model.track()`

**Context.** Ultralytics ships ByteTrack and BoT-SORT behind a single call.

**Decision.** Implement the association logic from scratch (Kalman,
cascade, memory), keep YOLO only as the detector.

**Why.** The goal is to own the identity logic: every threshold is
explainable, every stage can be switched off for the ablation, and the
long-term memory does not exist in the off-the-shelf trackers.

---

## 2. An identity switch costs more than a fragmentation

**Context.** When the tracker is unsure, it can either reuse an existing id
(risking a switch) or create a new one (a fragmentation).

**Decision.** Every ambiguous situation resolves towards a new id:
appearance gates, the ambiguity margin in re-identification, refusing to
learn appearance from occluded crops.

**Why.** A switch corrupts *two* trajectories and is invisible to a
downstream consumer (player statistics silently mix two people). A
fragmentation corrupts one and can be merged offline.

---

## 3. Occlusion is measured with *coverage*, not IoU

**Failure observed.** A small player partially hidden behind a large one
has a low IoU with him (the union is huge), so the crop was treated as
"clean". Its embedding was mostly the occluder's jersey, and the tracker
swapped their identities.

**Decision.** `coverage(i) = max_j inter(i, j) / area(i)`. A crop with
coverage ≥ 0.2 is *unreliable*: it never updates an appearance model,
never seeds a new track and never claims a stored identity. In stage 1 it is
matched on motion only.

---

## 4. Lost tracks and dormant identities compete in ONE assignment

**Failure observed.** With a separate "re-acquire lost tracks" stage
before the memory lookup, a player coming back on court was given to a
*lost team-mate* who looked similar and happened to be nearby — even though
his own dormant identity matched much better. His dormant id was later
revived on the wrong person: two stolen identities from one decision.

**Decision.** Stage 4 builds a single cost matrix with every absent
identity (lost + dormant), applies each row's own gate (reachable distance
for lost tracks, stricter threshold for dormant ones) and a margin test
across all of them, then solves one Hungarian assignment.

---

## 5. New tracks wait for a clean look before minting an id

**Failure observed.** A player re-entering next to a team-mate is first
detected with an occluded crop, which stage 4 correctly refuses to trust.
The new track was confirmed two frames later with a brand-new id, and the
correct identity could never be recovered.

**Decision.** When the memory is non-empty, a tentative track without a
clean appearance waits up to `memory_wait` frames; at promotion time it is
checked against the memory once more ("delayed re-identification").

---

## 6. Appearance thresholds are calibrated per extractor, on the train split

**Failure observed.** A colour histogram puts two team-mates at a cosine
distance of ~0.33 (the jersey half of the vector is identical). A gate of
0.35 copied from a deep-ReID setting let team-mates through.

**Decision.** `scripts/calibrate_reid.py` measures the same-player and
closest-other-player distance distributions on the **train** split and
proposes `appearance_gate` (95th percentile of positives) and
`memory_thresh` (1st percentile of hard negatives). Never tuned on val.

---

## 7. Width and height are tracked independently

**Decision.** Kalman state in (cx, cy, w, h) — BoT-SORT style — rather than
(cx, cy, aspect, h) — SORT/ByteTrack style.

**Why.** Players crouch, jump and stretch: the aspect ratio is not constant,
and a constant-aspect prior lags behind those changes.

---

## 8. Camera motion compensation with players masked out

**Decision.** Estimate a similarity transform between consecutive frames
from optical-flow corners *outside* detected boxes, and warp every track's
Kalman state with it before association.

**Why.** Broadcast cameras pan with the ball. Without compensation, a
standing player appears to move by the pan speed and IoU-based association
breaks exactly during fast breaks, when it matters most.

---

## 9. Filter people off the court before tracking, without training

**Failure observed (experiment 001).** A COCO `person` detector found 97 % of
the players, but half of its confident boxes were spectators, bench players
and photographers: 123 identities for 10 players, negative MOTA.

**Decision.** A post-detection `CourtMask` filter estimates the playing
surface from the floor colour (convex hull of the largest floor-coloured
region) and keeps a box only if its feet are on it. It is switchable
(`filters.court_mask`) and never adds or moves boxes; when too little floor is
visible it lets everything through.

**Why not only fine-tune the detector?** It needs a GPU and labelled data,
and it hides the reason for the improvement. The filter is explainable, runs
in ~5 ms on CPU, removed 71 % of false positives with no player lost
(experiment 003), and a fine-tuned detector can still be compared against it.

---

## 10. Spend ReID compute only where the tracker reads it

**Failure observed (experiment 001).** ReID took 2 s/frame — 57× the tracker —
because it embedded every box down to score 0.1.

**Decision.** Embed only boxes with
`score ≥ min(high_thresh, feature_update_min_score)`; lower-score boxes are
matched by IoU only and never update an appearance model. Switch:
`reid.only_usable_boxes`. Experiment 002 verified byte-identical tracks.

---

## 11. Appearance is a cost, not a veto, when motion is clear

**Failure observed (experiment 004a).** Adding ResNet ReID raised ID switches
from 36 to 50, all of them tracks cut and restarted. Calibration on the train
split (005) showed why: half of all same-player pairs one second apart are at
cosine distance > 0.41, while the gate vetoed anything above 0.35.

**Decision.** `tracker.appearance_veto_max_iou`: appearance may forbid a pair
only when the IoU is below 0.5 (motion ambiguous). Above it, appearance only
adds cost, so the Hungarian step still prefers the better-looking candidate
when two players overlap. Result: lost-track switches 29 → 15, and ReID beats
the motion-only tracker for the first time (IDF1 60.5 vs 58.2).

**Lesson.** Thresholds copied from another setting are hypotheses. Measure the
distance distributions of the extractor you actually use, on data you do not
evaluate on.
