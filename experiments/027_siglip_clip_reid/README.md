# 027 · SigLIP and CLIP, zero-shot, as ReID models: they do not tell team-mates apart

| | |
|---|---|
| **Date** | 2026-10-31 · **negative**: zero-shot, both towers stay below OSNet on all 15 clips (re-run at 256 × 128 after the correction below) |

> **Correction (2026-10-31).** In the first run each tower prepared the crop with its own image processor. SigLIP's
> squeezes the player into a 224 × 224 square; **CLIP's resizes the short side to 224 and keeps the central square, so
> a 1:2 player crop loses its top and bottom quarters (head and feet)**. The CLIP numbers below therefore describe the
> torso only and underrate it. Both towers now use our own preprocessing as ReID models: crop at 256 × 128, not
> deformed, position vectors interpolated (`hooptrack.reid.siglip.reid_transform`, `reid.input_size`), the same as
> the 028 training. The 020 team labels keep the old path (checked: identical output). Re-run below ("256 × 128"
> rows), with new model names so the old embedding caches are not reused.
| **Builds on** | 016 (team-mate benchmark), 017 (OSNet multi-sport), 020 (SigLIP used for TEAM labels, where it is the best labeller) |
| **Question** | CLIP-ReIdent (Habel et al. 2022) re-identifies players with a CLIP image tower. Before fine-tuning anything: how well do the generic image-text towers separate a player from his **team-mates**, with no training? |

> **Data and licences.** Crops: SportsMOT (CC BY-NC 4.0); models used as is: SigLIP base (Google, Apache-2.0), CLIP ViT-B/16 (OpenAI, MIT). All sources: [`docs/data_models_licences.md`](../../docs/data_models_licences.md).

## Setup

Same benchmark and crops as 016/017 (`scripts/reid_teammate_benchmark.py`, 15 val clips, ground-truth crops, clean
only). Two new types in `build_reid` (`hooptrack.reid.siglip`), weights read from local folders:

* `siglip`: `google/siglip-base-patch16-224`, pooled output of the vision tower (768-d), the model of 020;
* `clip`: `openai/clip-vit-base-patch16`, projected image embedding (512-d).

First run: each tower's own processor (see the correction above). No fine-tuning: the CLIP-ReIdent weights are not
used.

## Results, first run (15 val clips, each tower's own processor)

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates | AUC mate (2 s piece) | team-mate crossings: AUC | keep < swap |
|---|---|---|---|---|---|---|
| ResNet-50 ImageNet | 0.558 | 0.674 | 0.492 | 0.568 | 0.600 | 0.767 |
| colour histogram | 0.574 | 0.841 | 0.483 | 0.595 | 0.624 | 0.748 |
| **CLIP ViT-B/16** | 0.553 | 0.697 | 0.455 | 0.579 | 0.607 | 0.783 |
| **SigLIP base** | 0.590 | 0.762 | 0.624 | 0.608 | 0.650 | 0.911 |
| OSNet part-based (018) | 0.668 | 0.739 | 0.761 | 0.701 | 0.720 | 0.957 |
| OSNet 012 (basketball) | 0.713 | 0.839 | 0.757 | 0.756 | 0.795 | 0.969 |
| OSNet 017 (multi-sport) | 0.715 | 0.851 | 0.791 | 0.737 | 0.810 | 0.965 |

Per clip (`AUC mate`, crop): SigLIP is below OSNet 012 on **all 15 clips** (−0.07 to −0.17), CLIP on all 15
(−0.11 to −0.25). SigLIP beats the colour histogram on 12 of 15, CLIP on 2.

## Results, re-run at 256 × 128 (crop not deformed, our preprocessing)

| model | AUC mate (crop) | AUC opp (crop) | rank-1 among team-mates | AUC mate (2 s piece) | team-mate crossings: AUC | keep < swap |
|---|---|---|---|---|---|---|
| CLIP, own processor (torso only) | 0.553 | 0.697 | 0.455 | 0.579 | 0.607 | 0.783 |
| **CLIP, 256 × 128** | **0.632** | 0.789 | **0.670** | **0.675** | **0.717** | **0.957** |
| SigLIP, own processor (squeezed) | 0.590 | 0.762 | 0.624 | 0.608 | 0.650 | 0.911 |
| SigLIP, 256 × 128 | 0.579 | 0.749 | 0.571 | 0.602 | 0.649 | 0.903 |
| OSNet 012 | 0.713 | 0.839 | 0.757 | 0.756 | 0.795 | 0.969 |

Per clip (`AUC mate`, crop) at 256 × 128: CLIP below OSNet 012 on all 15 clips (−0.05 to −0.12), SigLIP on all 15
(−0.09 to −0.19).

* **The CLIP crop was the problem, not CLIP**: seeing the whole player, CLIP goes from 0.553 to 0.632 and from the
  level of the ImageNet ResNet to clearly above colour, with 95.7 % right decisions at team-mate crossings.
* SigLIP does not gain from keeping the proportions (0.590 → 0.579): its attention-pooling head may rely more on
  the square layout it was trained on. Not investigated.
* Zero-shot, neither reaches OSNet. The readings below (written for the first run) still hold, except that CLIP is
  now the better of the two towers.

Per clip and per model: [`summary_val.json`](summary_val.json).

## Reading

1. **Zero-shot, neither tower is a ReID model.** CLIP is at the level of the ImageNet ResNet-50 (0.553 vs 0.558);
   SigLIP is a little better than colour (0.590 vs 0.574) and far from OSNet (0.713).
2. **This is consistent with 020.** SigLIP is the best TEAM labeller because it describes the kit; telling two
   players in the same kit apart needs the details a ReID model is trained for (build, shoes, hair, how a shirt
   hangs), and these generic towers were never asked to separate two people in identical clothes.
3. **CLIP-ReIdent's gain comes from its fine-tuning** (contrastive training on player crops), not from the CLIP
   tower itself. Testing that path means training CLIP/SigLIP on our player crops in Colab, the same data and
   evaluation as 012/017: a ViT-B/16 needs ~18× the compute of OSNet per crop (~17.6 vs ~1.0 GFLOPs) and ~40× the parameters, so it would only be worth it offline
   (stitching), and only if it clearly beats OSNet on this benchmark.

## Conclusion

**Negative, not adopted.** `reid.type=siglip|clip` stays available. Next step on this line: fine-tune SigLIP (or
CLIP) on the 012 crops in Colab and compare on this benchmark before touching the tracker, prepared in
[028](../028_siglip_reid_finetune/).

```powershell
python scripts/reid_teammate_benchmark.py --config configs/player.yaml --model `
    resnet50_imagenet:resnet histogram:histogram osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth `
    osnet_multisport:osnet:weights/osnet_x1_0_multisport.pth `
    siglip:siglip:weights/siglip-base-patch16-224 clip:clip:weights/clip-vit-base-patch16
```

Re-run with the 256 × 128 preprocessing (the default since the correction), under new names:

```powershell
python scripts/reid_teammate_benchmark.py --config configs/player.yaml --model `
    resnet50_imagenet:resnet histogram:histogram osnet_sportsmot:osnet:weights/osnet_x1_0_sportsmot.pth `
    osnet_multisport:osnet:weights/osnet_x1_0_multisport.pth `
    siglip_256x128:siglip:weights/siglip-base-patch16-224 clip_256x128:clip:weights/clip-vit-base-patch16
```
