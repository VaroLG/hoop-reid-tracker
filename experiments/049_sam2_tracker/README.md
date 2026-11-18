# 049 · SAM 2 carries each player's identity (mask tracking on the same boxes)

| | |
|---|---|
| **Date** | 2026-11-15 → 2026-11-18 |
| **Builds on** | [046](../046_crossing_links_ceiling/): fixing the crossings after the online tracker does not help 038, so the change goes into the tracker itself · [024–026](../024_hota_deep_eiou/): another tracker compared on the same boxes |
| **Question** | If a SAM 2 mask follows each player frame to frame, and every box goes to the mask it contains, are there fewer identity switches than with the online tracker (011b) on the SAME boxes? And does 038 + 048 on top beat the current pipeline? |
| **Clips** | values chosen on the 3 held-out train clips (`v_-6Os86HzwCs_c009`, `v_2j7kLB-vEEk_c010`, `v_4LXTUim5anY_c013`); val (15 clips) once |

> **Data and licences.** SportsMOT (CC BY-NC 4.0); SAM 2.1 hiera-small (Meta, Apache-2.0) through Hugging Face
> `transformers`; the online tracker's models are this project's own (not redistributed). All sources:
> [`docs/data_models_licences.md`](../../docs/data_models_licences.md).

## Change

[`scripts/sam2_track.py`](../../scripts/sam2_track.py) (tests: [`tests/test_sam2_assoc.py`](../../tests/test_sam2_assoc.py)):

1. the boxes of the online run (011b: players only, on the court), so only the identity mechanism changes;
2. the clip in windows of `--window` frames sharing their first / last frame; at the start of each window every
   followed player is prompted to SAM 2 with his box, plus a new player for every box nobody owns;
3. SAM 2 propagates the masks through the window; each frame's boxes go to the masks they contain (share of the
   mask inside the box ≥ `--min-share`, Hungarian);
4. boxes no mask claims continue short IoU tracks and become SAM 2 players at the next window; a mask that owns no
   box for `--max-lost` frames is dropped;
5. **inertia** (`--keep-bonus`, off by default): the assignment score adds `keep_bonus × IoU(the player's box in the
   last frame, this box)`; the share must still reach `--min-share`. Added after the first grid (see below).

Grid on train: window 15 / 30 / 60 / 120 / 240 frames × min share 0.4 / 0.6, then keep bonus 0 / 0.3 / 0.6 at the
best window; the best HOTA on train is run once on val, and then 038 + 048 on top (also chosen on train).

Speed on a Colab T4 (SAM 2.1 small, ~10 players, profiled on `v_-6Os86HzwCs_c009`): float32 0.4 fps (≈80 s per
30-frame window, almost all in the model); bfloat16 is only emulated on a T4 and was no faster; **float16 1.0 fps**
(≈26 s per window, every mask non-empty): used here (`--dtype float16`). Val (15 clips) takes ≈3.5 h.

## How to run

GPU: [`notebooks/colab_sam2_tracking.ipynb`](../../notebooks/colab_sam2_tracking.ipynb) (Google Colab, a T4 is
enough). It runs 011b on the same clips first, so the baseline and SAM 2 come from the same session, and saves
everything to your Drive.

## Results

### 1 · Train (3 held-out clips): window and min share

| run | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|
| 011b (online, the same boxes) | 72.22 | 59.17 | 74.41 | 97.54 | 35 |
| SAM 2 w15 s0.4 / s0.6 | 71.02 / 70.92 | 57.53 / 57.30 | 69.34 / 69.14 | 96.80 / 96.83 | 139 / 136 |
| SAM 2 w30 s0.4 / s0.6 | 74.90 / 74.89 | 64.25 / 64.16 | 75.31 / 74.54 | 96.83 / 96.81 | 132 / 134 |
| SAM 2 w60 s0.4 | 78.25 | 69.93 | 80.27 | 96.76 | 152 |
| **SAM 2 w120 s0.4** | **80.13** | **72.93** | **83.16** | 96.51 | 189 |
| SAM 2 w240 s0.4 | 78.34 | 69.45 | 82.77 | 96.56 | 188 |

Longer windows help (SAM 2's memory carries a player further) up to 120 frames; min share hardly matters.
But **identity switches rose** (35 → 189) while HOTA, AssA and IDF1 rose: more, but shorter, errors.

### 2 · Why the switches rose: flicker in piles

0 of the switches fall on a window border. Counting switches whose player gets his previous id back within 10
frames (A → B → A):

| clip | switches (w120) | come back within 10 frames | 011b switches |
|---|---|---|---|
| `c009` | 9 | 4 | 4 |
| `c010` | 30 | 15 | 14 |
| `c013` | 121 | 89 | 15 |

When two players overlap, each mask lies inside both boxes, and a fresh assignment every frame flips between them.
Hence the inertia (change 5):

| run (train) | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|
| SAM 2 w120 s0.4, no inertia | 80.13 | 72.93 | 83.16 | 96.51 | 189 |
| **+ keep bonus 0.3** (chosen) | **80.57** | **73.04** | **83.34** | **97.47** | **44** |
| + keep bonus 0.6 | 80.52 | 72.99 | 83.33 | 97.45 | 48 |

Chosen on train: window 120, min share 0.4, keep bonus 0.3.

### 3 · Val (15 clips), once: the online tracker

| run | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|
| 011b online | 68.8 | 55.10 | 72.71 | 96.65 | 279 |
| **SAM 2 online** | **78.1** | **70.39** | **84.34** | 96.67 | 368 |

### 4 · 038 + 048 on top

On train, three options (038 needs its cut at crossings to give pieces to exactly 10 players):

| run (train) | HOTA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|
| 011b + 038 + 048 (the final pipeline) | 84.7 | 81.5 | 94.5 | 97.5 | 36 |
| **SAM 2 + 038 + 048** (chosen) | **88.0** | **87.2** | **97.6** | **97.6** | **19** |
| SAM 2 + 048 (no 038) | 80.6 | 73.2 | 83.4 | 97.6 | 34 |

Val, once:

| run | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|
| 011b + 038 + 048 (reproduces the final pipeline) | 79.8 | 84.9 | 75.0 | 89.6 | 96.6 | 365 |
| **SAM 2 + 038 + 048** | **84.6** | **86.4** | **82.8** | **95.1** | **96.8** | **236** |

HOTA per val clip (011b + 038 + 048 → SAM 2 + 038 + 048): 13 of 15 clips improve, `v_5ekaksddqrc_c003` is level
(77.2 → 77.3) and one drops, `v_00HRwkvvjtQ_c001` (86.4 → 82.4). The largest gains are on the hardest clips:
`v_00HRwkvvjtQ_c011` 66.9 → 85.8, `v_BgwzTUxJaeU_c014` 73.1 → 87.4, `v_BgwzTUxJaeU_c008` 65.0 → 79.6.

Files: `hota_*.csv` and `chosen_on_train.json` from `MyDrive/hooptrack/runs/049/`.

## Conclusion

**Adopted.** SAM 2 carrying each player's mask is the first tracker change that improves the whole pipeline, not
only the online stage: on val **+4.8 HOTA (79.8 → 84.6), +7.8 AssA, +5.5 IDF1 and 129 fewer identity switches**
(365 → 236), with every value chosen on train. The two parts are complementary: SAM 2 keeps identities through
crossings, so 038's pieces between crossings are cleaner; 038 still adds the closed set of 10 (SAM 2 alone: 78.1
on val). This is what 046 pointed to: correcting crossings afterwards could not help, a tracker that does not
break them did.

Two lessons on the way: IDSW alone misled (it rose while every identity measure improved, because of one-frame
flicker in piles), and the fix came from a diagnostic, not from the grid.

Cost: a GPU. ≈1 fps on a Colab T4 in float16 (SAM 2.1 small), against real time for the online tracker on the
laptop's GPU. The CPU pipeline stays 011b + 038 + 048.

**In the pipeline.** `scripts/run_pipeline.py --tracker auto` (the default) uses SAM 2 when PyTorch sees an NVIDIA
GPU and 011b on a CPU; `--tracker sam2` / `online` force one. Without a GPU, the identities can be computed in Colab
(`--players-only`, part B of the notebook) and the rest drawn on the laptop (`--players-from`).
