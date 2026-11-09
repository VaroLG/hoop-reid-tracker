# 028 · SigLIP fine-tuned as a player ReID model (CLIP-ReIdent idea)

| | |
|---|---|
| **Date** | 2026-10-31 · **positive on the benchmark**: fine-tuned SigLIP beats OSNet on all 15 val clips (AUC mate 0.860 vs 0.713). Not yet tried in the tracker |
| **Builds on** | 027 (zero-shot SigLIP / CLIP: AUC mate 0.590 / 0.553 vs OSNet 0.713), 012 (data, sampler, losses, test protocol) |
| **Question** | CLIP-ReIdent's gain comes from fine-tuning the CLIP tower on player crops. Does a fine-tuned SigLIP tell team-mates apart better than OSNet? |

> **Data and licences.** Training data: player crops of SportsMOT (CC BY-NC 4.0); starting model: SigLIP base (Apache-2.0); the fine-tuned model (used by 029–048) is not redistributed (CC BY-NC 4.0 if it were). All sources: [`docs/data_models_licences.md`](../../docs/data_models_licences.md).

## Method (fixed before training)

Notebook [`notebooks/colab_train_reid_siglip.ipynb`](../../notebooks/colab_train_reid_siglip.ipynb). Everything that
can stay as in 012 stays: same `reid_players.zip` (15 basketball train clips to learn, 15 val clips to test), P × K =
16 × 4 identities from one clip, identity loss + batch-hard triplet (margin 0.3), same augmentations, same retrieval
test (query vs same clip ≥ 2 s away). What changes, and why:

| | 012 OSNet | 028 |
|---|---|---|
| backbone | OSNet-x1.0 (2.2 M parameters, ImageNet) | SigLIP base ViT-B/16 vision tower (~86 M, web image-text); `MODEL = "clip"` for CLIP ViT-B/16 |
| input | 256 × 128 | the same 256 × 128 (height × width), not deformed: the position vectors learned on a 14 × 14 grid (224 × 224) are interpolated to 16 × 8 (`interpolate_pos_encoding`), as CLIP-ReID does; normalisation with the tower's own mean / std |
| optimiser | Adam 3e-4, 60 epochs, step decay | AdamW, 1e-5 for the tower, 1e-3 for the new head, 1 epoch warm-up + cosine, 20 epochs, gradient clipping 1.0 |
| head | torchreid classifier | BatchNorm neck + linear classifier, training only; the embedding is the tower's own output |

A pretrained transformer is damaged by large steps, hence the low tower learning rate; 20 epochs because a ViT-B/16
costs ~18× OSNet's compute per crop. No validation split, no best-epoch selection, one run.

The towers' own processors are not used (CLIP's would cut off the head and feet, SigLIP's squeezes the player, see
027). The output is a plain Hugging Face vision model folder plus `hooptrack_reid.json` (input size), loaded on the
laptop by the existing `reid.type=siglip` (or `clip`) extractor, whose preprocessing (`reid_transform`) is the
notebook's test transform. The notebook re-loads the saved folder before the final test. Checked on tiny random
towers: same input pixels and same embedding in the notebook and on the laptop (max difference 0.0), for SigLIP and
CLIP; tests in `tests/test_siglip_reid.py`.

## How to run

1. `data/reid_players.zip` in `MyDrive/hooptrack/` (already there from 012).
2. Colab, T4 GPU or better, run all cells (~1.5–2 h on a T4; resumes from Drive if disconnected).
3. Download `MyDrive/hooptrack/runs/reid_siglip_reid_sportsmot/`: `siglip_reid_sportsmot.zip`, `eval_028.json`,
   `training_curves.png`, `distances.png`. Unzip the model into `weights/siglip_reid_sportsmot/`.
4. The team-mate benchmark decides (016 protocol):

```powershell
python scripts/reid_teammate_benchmark.py --config configs/player.yaml --model `
    osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth osnet_multisport:osnet:weights/osnet_x1_0_multisport.pth `
    siglip_256x128:siglip:weights/siglip-base-patch16-224 siglip_ft:siglip:weights/siglip_reid_sportsmot
```

**Decision rule (set now):** worth trying in the offline stitching only if it beats OSNet 012 on the benchmark's
AUC mate on most of the 15 clips. It is too heavy for the online tracker on CPU.

## Results

Both towers trained once each, same recipe (`MODEL = "siglip"` and `MODEL = "clip"`), Colab, 20 epochs.
Training reports: `eval_028_siglip.json`, `eval_028_clip.json` (recipe, losses per epoch, test, distances).
Benchmark summary: `benchmark_val.json`. Train set 36,311 crops / 150 identities; test 23,797 crops / 150 identities
(4 games never seen in training, as in 012).

### Retrieval among all players of a clip (012 protocol, in the notebook)

| model | rank-1 | mAP |
|---|---|---|
| SigLIP zero-shot | 0.510 | 0.223 |
| CLIP zero-shot | 0.560 | 0.267 |
| OSNet 012 | 0.710 | 0.409 |
| OSNet 017 (multi-sport) | 0.745 | 0.448 |
| **SigLIP fine-tuned** | **0.852** | **0.622** |
| CLIP fine-tuned | 0.833 | 0.598 |

### Team-mate benchmark (016 protocol, laptop, 15 val clips) — the decision

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates | AUC mate (2 s piece) | team-mate crossings: AUC | keep < swap |
|---|---|---|---|---|---|---|
| OSNet 012 | 0.713 | 0.839 | 0.757 | 0.756 | 0.795 | 0.969 |
| OSNet 017 | 0.715 | 0.851 | 0.791 | 0.737 | 0.810 | 0.965 |
| SigLIP zero-shot, 256 × 128 | 0.579 | 0.749 | 0.571 | 0.602 | 0.649 | 0.903 |
| CLIP zero-shot, 256 × 128 | 0.632 | 0.789 | 0.670 | 0.675 | 0.717 | 0.957 |
| **SigLIP fine-tuned** | **0.860** | **0.972** | **0.895** | **0.909** | **0.931** | **0.988** |
| CLIP fine-tuned | 0.835 | 0.951 | 0.874 | 0.874 | 0.919 | 0.988 |

Per clip, `AUC mate` (crop), fine-tuned vs OSNet 012: **SigLIP better on 15 of 15** (+0.04 to +0.17, median +0.14),
CLIP better on 15 of 15 (+0.07 to +0.16). Against OSNet 017: SigLIP 14 of 15, CLIP 15 of 15. The same holds for
2-second pieces and for rank-1 among team-mates. 258 team-mate and 445 opponent crossings, as before.

## Reading

1. **The decision rule is met, clearly.** Fine-tuned, both towers beat OSNet on every val clip. The largest gain
   is where it was needed: telling team-mates apart (AUC 0.71 → 0.86) and pieces of 2 s (0.76 → 0.91), which is what
   the offline stitching compares.
2. **Pretraining makes the difference, not the architecture alone.** Zero-shot the towers were below OSNet
   (027); after the same 20 epochs on the same 150 players they are far above. The web image-text pretraining gives
   them a much richer description of a person, and the fine-tuning selects the parts of it that identify one player.
3. SigLIP ends slightly above CLIP, although CLIP was the better zero-shot tower (027). Both are one run each; the
   difference between the two (0.025 AUC) is small next to their gain over OSNet.
4. **The base-rate argument of 016 still applies.** At team-mate crossings the right decision is now taken 98.8 % of
   the time (OSNet 96.9 %). 016 estimated that an online swap check needs ~99 %+ to pay off: close, not yet there.
5. **Cost.** A ViT-B/16 at 256 × 128 needs ~12 times OSNet's compute per crop (~11.5 vs ~1.0 GFLOPs): not for the online tracker on CPU, but
   cheap for the offline stitching, which embeds a dozen clean crops per piece.

## Next

Use the fine-tuned SigLIP in the offline stitching (013b/020), the step that joins and cuts pieces. Its distance scale
is different from OSNet's (equal-error threshold 0.58 vs 0.16 in the notebook), so the stitching threshold must be
recalibrated **on the train split**, not chosen on val.
