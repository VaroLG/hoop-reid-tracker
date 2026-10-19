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

## 12. Team colour: a filter for referees, not a fix for swaps

**Context.** 14 of 36 ID switches are swaps between players of different teams,
and referees are tracked because they stand on the court. A per-video, unsupervised
jersey-colour classifier (k-means, k = 3: two teams + "other") labels boxes
with ~3 % team errors on clean crops.

**What failed.** Forbidding cross-team matches (006a–c) fixed none of the 14
swaps. A diagnosis showed why: every swap happens while the two players overlap
(ground-truth IoU up to 0.89). The crop that would need its colour is the hidden
one, and the motion match is confident, so a gate limited to ambiguous motion
never fires; at any IoU, label noise cuts correct tracks (36 → 149 switches).

**Decision.** Keep the classifier for what it does well: tracks voted "other"
are not output (`tracker.suppress_other`). −83 % false positives, IDF1 58.2 → 64.0
(motion-only). The team gate stays in the code, off, documented as a negative result.

**Lesson.** Check *where* an error happens before choosing the cue that should fix
it: a cue that is unavailable at that moment cannot help, however good it is elsewhere.

## 13. Live by default, offline as an explicit mode

**Context.** After 006, most remaining false positives were referees shown
*before* the colour vote could recognise them: during the 60-frame warm-up and
in the first frames of a re-entry.

**What failed.** Fitting the colour model earlier (007a) never produced a
reliable model — too few clean crops in the first seconds. Holding back new
tracks until they have a team (007b) hid as many player frames as referee frames.

**Decision.** Removing a referee's whole track once the video is over works
(−64 % FP, 007c), but it uses the future. It is a switch
(`teams.offline_suppress_other`) and a separate row in the results, not the
default: the default pipeline stays a live tracker, and the README says which
numbers are which.

**Lesson.** When a fix needs information from the future, make the trade-off
explicit instead of folding it silently into the headline number.

## 14. Calibrate on train before trusting a pattern seen on one clip

**Context.** On the demo clip no player ever stands on the edge of the floor hull,
so a stricter (inward) court margin looked like a free win against people on the
sidelines (008).

**What the train split said.** In 6 of 15 train clips (one arena) players' feet
are regularly *outside* the hull — the floor-colour mask misses part of that court —
and the existing margin already deletes up to 10 % of the players there. A
stricter margin would have made it worse; the 99 %-of-players rule asks for a
*looser* one.

**Decision.** No global margin change. A per-video margin learned from
team-labelled players is implemented (off: no net gain on the demo clip, where
the border people are referees already handled by 006). The next priority is
evaluating on the full validation split, not more tuning on one clip.

**Lesson.** One clip is an anecdote. Every threshold gets a train-split check,
and a filter that "never removed a player" on one video has to prove it on others.

## 15. Court geometry is a separate chapter

**Context.** Fixing the court mask properly (008) needs a court-keypoint
detector, a homography and per-league templates — a training pipeline of its
own, also needed later for the top-down map.

**Decision.** It lives in its own chapter: `src/hooptrack/court/`,
`scripts/court/`, `tests/court/`, `experiments/court/` (steps C01, C02, …).
The court package never imports the tracker and the tracker never imports the
court package; a single adapter will connect them behind a config switch.
`tests/court/test_separation.py` enforces both directions.

**Why.** Each part can be read, tested and reused alone (the court model is
useful without the tracker), and a reviewer sees two clear stories instead
of one tangled one.
