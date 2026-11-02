# 024 · HOTA on the 15 val clips, and Deep-EIoU as an external baseline

| | |
|---|---|
| **Date** | 2026-10-30 · **positive**: Deep-EIoU online + our offline team cut (020) = **HOTA 77.4, IDF1 83.8** (020: 75.3 / 81.8) |
| **Builds on** | 011b (online tracker), 013b / 019 / 020 (offline stitching and team cut), 012 (OSNet) |
| **Question** | Where does the project stand on the metric SportsMOT ranks by (HOTA)? And is our online tracker as good as the reference sports tracker when both get the same detections? |

> **Correction (2026-10-30, found while preparing 026).** The online run 011b uses the ImageNet ResNet-50 for
> appearance, not OSNet: OSNet (012) is only used by the offline stitching (013b, 019, 020). So "Deep-EIoU online vs
> 011b" below compares **tracker + ReID**, not the tracker alone. The fair comparison (our online tracker with OSNet on
> the 15 clips) is the first run of [026](../026_deep_eiou_port/). The rows with our offline step are not affected:
> stitching uses OSNet in every run.

## Setup

- **HOTA / DetA / AssA**: official TrackEval, `scripts/eval_hota.py`, same 15 basketball val clips for every run.
- **Deep-EIoU** (Huang et al., WACV-W 2024): official code, unchanged, cloned to `third_party/` (no licence file in
  their repository, so it is not copied into this one). `scripts/run_deep_eiou.py` feeds it **our** cached
  detections (011, YOLO26m players only + C06 court mask) and **our** OSNet (012). Only the tracker changes.
  Their default arguments (high 0.6, low 0.1, new 0.7, buffer 60, match 0.8, proximity 0.5, appearance 0.25),
  **nothing tuned on val**. Reported raw and with their interpolation (`tools/sport_interpolation.py`).
- **deep_eiou_020**: the offline step of 020 (team cut with SigLIP labels, k=2, then GTA stitching with OSNet),
  run unchanged on Deep-EIoU's online output.

## Results (15 val clips)

| run | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|---|
| 011b: our online | 68.8 | 86.0 | 55.1 | 72.7 | 96.7 | 278 |
| 013b: + offline stitching | 73.0 | 85.7 | 62.3 | 79.9 | 96.7 | 225 |
| 019: + team cut (colour labels) | 73.7 | 85.8 | 63.3 | 79.7 | 96.6 | 301 |
| 020: + team cut (SigLIP labels) | 75.3 | 85.7 | 66.1 | 81.8 | 96.6 | 283 |
| Deep-EIoU online | 74.3 | 87.0 | 63.5 | 79.6 | 95.8 | 859 |
| Deep-EIoU + interpolation | 74.4 | 86.8 | 63.8 | 79.6 | 95.6 | 731 |
| **Deep-EIoU + our offline (020)** | **77.4** | 86.9 | **69.0** | **83.8** | 95.8 | 853 |

Per clip (HOTA, `hota.csv`):

| comparison | better | worse | median Δ HOTA |
|---|---|---|---|
| Deep-EIoU online vs our online (011b) | 10 | 5 | +5.9 |
| Deep-EIoU online vs our full pipeline (020) | 7 | 8 | −0.2 |
| **Deep-EIoU + our offline vs 020** | **10** | 5 | **+1.9** |
| **Deep-EIoU + our offline vs Deep-EIoU alone** | **12** | 2 (1 equal) | **+3.3** |
| interpolation vs none | 9 | 2 (4 equal) | +0.1 |

## Reading

1. **Our online tracker is the weak link.** With the same detections, Deep-EIoU's online tracker with OSNet
   beats 011b (ResNet-50 ImageNet, see the correction above) by +5.5 HOTA (AssA 55.1 → 63.5), on 10 of 15 clips. Alone, it already matches our whole offline
   pipeline (74.3 vs 75.3, 7 vs 8 clips).
2. **Our offline step generalises.** It was designed and tuned on our tracker's output. On another tracker's output,
   unchanged, it still adds +3.1 HOTA / +4.3 IDF1, on 12 of 15 clips. The team cut is not fitted to 011b's errors.
3. **The two combine.** Their online + our offline is the best result of the project: 77.4 HOTA, better than 020 on
   10 of 15 clips. The 5 losses are clips where our online tracker was already strong (two NCAA clips c001, c003, and
   three clips of `v_5ekaksddqrc`).
4. **Part of the gain is detection, not association.** DetA 87.0 vs 85.7: Deep-EIoU keeps low-score boxes in a
   second association stage (BYTE, low threshold 0.1), so it finds more players, with more false positives
   (MOTA 95.8 vs 96.6).
5. **Interpolation is neutral** (+0.1).
6. **Open:** Deep-EIoU has 3× more ID switches (859 vs 283) while its AssA and IDF1 are higher. That suggests many
   short flickers (an id lost and recovered within a few frames), which cost little in HOTA. Not checked yet.

## Conclusion

Positive, but **not adoptable as is**: Deep-EIoU's code has no licence, so the repository cannot ship it. What it shows
is where to work next: the online association. Next experiment: find which part of Deep-EIoU makes the difference
(second low-score stage, iterative box expansion, the ReID/IoU gating) by ablation, and reimplement that part in our
tracker. Plain EIoU alone was already tried in 014 and did not help, so the expansion is probably not the whole story.
