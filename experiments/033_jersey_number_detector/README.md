# 033 · Jersey numbers, done like the working systems: detect the number, then read it

| | |
|---|---|
| **Date** | 2026-11-03 · **negative (gate not passed)**: 32 % of our pieces get a number (target 50 %), and the reads are much less reliable than on E-BARD; HOTA 76.9 → 76.7 |
| **Builds on** | 023 (Florence-2 zero-shot on the whole upper body: right 96 % of the time when it answers, but a number for only 22 % of the val pieces), 030/031 (appearance cannot undo the swaps inside the tracker), the state-of-the-art review in the project docs (`estado-arte-identidad-jugadores`: every working system detects the number region first and votes the read over time) |
| **Question** | With a number DETECTOR in front of the reader, how many pieces get a reliable number? Enough for the (team, number) rule of the stitching to matter? |

> **Data and licences.** Training data: E-BARD jersey numbers (CC BY 4.0) and a Roboflow Universe player dataset (its own licence, printed by the notebook); starting models: SigLIP base (Apache-2.0), YOLO26n (AGPL-3.0); test frames: SportsMOT (CC BY-NC 4.0); weights not redistributed. All sources: [`docs/data_models_licences.md`](../../docs/data_models_licences.md).

## Change of plan (2026-11-02, before any training)

The Roboflow Universe dataset could not be downloaded (no downloadable version from the page). A better source was
found: **E-BARD jersey-number recognition** ([`GabrieleGiudici/E-BARD-JerseyNumberRecognition`](https://huggingface.co/datasets/GabrieleGiudici/E-BARD-JerseyNumberRecognition),
CC BY 4.0, University of Brescia): 3,633 stubs of 16 consecutive **player crops** (NBA 2024-25 broadcast, tracked
with ByteTrack), each labelled with the number or `nan`; author splits 2,906 / 363 / 364 (same games in every split).
It trains a READER directly, so the number-region detector is not needed:
`notebooks/colab_train_jersey_reader.ipynb` and `src/hooptrack/reid/jersey_classifier.py` (the notebook writes an
exact copy of it):

* SigLIP base image tower on the player crop at 256 x 128 + three heads, **digit by digit** (number of digits 0/1/2,
  tens, units): only 52 numbers appear in train, our clips have others, and a per-number classifier could never
  answer those. The 111 answers (no number, 0-9, 00-99) are products of the heads (`number_log_probs`).
* A piece is read by averaging its crops' answers (`read_piece`); the confidence needed is the lowest at which 95 %
  of the answered stubs are right on the dataset's own val split.
* Same optimiser as 028 (AdamW 1e-5 tower / 1e-3 heads, warm-up + cosine), 10 epochs, best epoch on the dataset's
  val; no horizontal flip; blur, down-scaling and small erasing as augmentations. Tests: `tests/test_jersey_classifier.py`.

The gate (step 2 below) and the use in the stitching (step 4) do not change. The Roboflow detector notebook
(`colab_train_number_detector.ipynb`) stays as the fallback.

## Plan, with a gate at each step (fixed before running, first version)

1. **Detector** (`notebooks/colab_train_number_detector.ipynb`, Colab): a public Roboflow Universe basketball
   dataset with a `number` class (the one of Roboflow's "Basketball AI" notebook). Turned into the pipeline's
   situation: one crop per player box (+10 % padding), with the number boxes inside it (IoS >= 0.9); players with
   no visible number kept as negatives. YOLO26n, 320 px, one class, **no horizontal flip** (a mirrored digit is not
   a digit). Reported on the dataset's test split: recall at confidence 0.5, false numbers on crops without one.
2. **Coverage on our val clips** (laptop): the detector's tight crops read by Florence-2 zero-shot, same reading
   rule as 023 (1-2 digits, confidence >= 0.7, at least 2 reads, 2/3 agree). **Gate:** a number for clearly more
   pieces than 023 (target: half of them or more), and more of the 37 wrong merges of 020 numbered on both sides
   than 023's 2. Otherwise the number cannot change the result: stop here.
3. Only if the reads are often wrong: fine-tune the reader (Roboflow: SmolVLM2 56 % → 86 %, ResNet 93 %).
4. The (team, number) rule in the stitching (`--jersey-rule`, already in the repo) on top of 029; measured with
   IDSW and switches per minute by kind (team-mate switches are the target).

The dataset images are not added to the repository: only the trained weights' report (`eval_033.json`, with the
dataset's licence lines) and the results.

## Results

### 1 · The reader on E-BARD (`eval_033.json`, `confidence_curve.png`)

SigLIP base, 10 epochs on a Colab GPU; best epoch on the dataset's val split: 5. The confidence needed was set on
that val split: **0.64** (the lowest with >= 95 % of the answered stubs right).

| E-BARD split | stub accuracy (all answers) | stubs answered at 0.64 | right when answered at 0.64 |
|---|---|---|---|
| val (sets the threshold) | 76.6 % | 57.6 % | 95.2 % |
| test | 70.6 % | 53.6 % | **93.8 %** |

The training loss goes to 0.03 while the val accuracy stops improving after epoch 4-5: the model learns the training
players quickly. The test split shares the games of train, so these numbers are optimistic for other leagues.

### 2 · The gate, on our 15 val clips (`coverage.txt` from [`scripts/jersey_coverage.py`](../../scripts/jersey_coverage.py), `hota.csv`)

Run: the 029 stitching + `--jersey-rule --jersey-reader classifier --jersey-crops 12` (12 tallest clean crops per
piece, threshold 0.64 from E-BARD).

**Coverage: 107 of 334 pieces (32 %)** get a number (023: 22 %), 36 % of the boxes. Per clip 13-45 %. The target
fixed before running was half of the pieces: **not passed.**

**Consistency** (pieces matched to the ground-truth player they cover most; teams from jersey colour):

| pairs of numbered pieces | same number | different numbers |
|---|---|---|
| same player | 22 | 8 (27 % disagree) |
| two different team-mates | **26 (18 %: impossible, numbers are unique in a team)** | 119 |

On E-BARD 94-95 % of the answers above the threshold were right; on our clips the same threshold gives answers
that contradict each other a quarter of the time. Other signs of the change of domain: the reader says "no number"
for only 12 of 321 pieces (8 % of the E-BARD stubs are `nan`, and our clips are full of backs and side views), the
median confidence is 0.49 (below the 0.64 threshold), and the most common answer is "14" (18 pieces).

**Use**: of the 31 wrong joins of 029 that could be judged, the numbers would block **4**; they would also block 2 right
joins. In the run: HOTA 76.93 → 76.73, IDF1 83.90 → 83.62, IDSW 278 → 277, switches per minute 31.3 → 31.7; per clip
HOTA 2 better, 1 worse, 12 unchanged.

## Conclusion

**Negative, not adopted** (the rule stays off; `--jersey-reader classifier` stays available). Trained on NBA footage,
the reader does not transfer well enough to the NCAA and European clips of SportsMOT: it reads a number on a third of
the pieces, and those reads contradict each other often. The jersey route would need **in-domain labels** (numbers of
players of these leagues), which SportsMOT does not have.
