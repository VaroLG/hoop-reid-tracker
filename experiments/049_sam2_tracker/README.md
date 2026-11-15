# 049 · SAM 2 carries each player's identity (mask tracking on the same boxes)

| | |
|---|---|
| **Date** | 2026-11-15 (prepared; results pending) |
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
   box for `--max-lost` frames is dropped.

Grid on train: window 15 / 30 frames × min share 0.4 / 0.6; the best HOTA on train is run once on val.

## How to run

GPU: [`notebooks/colab_sam2_tracking.ipynb`](../../notebooks/colab_sam2_tracking.ipynb) (Google Colab, a T4 is
enough). It runs 011b on the same clips first, so the baseline and SAM 2 come from the same session, and saves
everything to your Drive.

## Results

*Pending: copy `hota_train.csv`, `chosen_on_train.json`, `hota_val.csv` (and `hota_val_final.csv`) here.*

## Conclusion

*Pending.*
