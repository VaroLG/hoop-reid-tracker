# B07 · WASB, the published state of the art for sports balls, on our test clips (no training)

| | |
|---|---|
| **Date** | 2026-11-05 |
| **Builds on** | [B05b/c](../B05b_static_and_weak/): our detector + tracker, 78 % of the balls found at 73 % precision |
| **Question** | How far is our approach from the state of the art? WASB (Tarashima et al., BMVC 2023; code MIT) predicts a heatmap from 3 consecutive frames at high resolution; its basketball weights were trained on ~275k NBA frames. |

> **Data and licences.** Frames: SportsMOT (CC BY-NC 4.0); model used as is: WASB (github.com/nttcom/WASB-SBDT, MIT) with its basketball weights (trained by its authors on the NBA dataset of Yan et al., ECCV 2020); WASB's code stays in `third_party/`, not committed. All sources: [`docs/data_models_licences.md`](../../../docs/data_models_licences.md).

## Method

[`scripts/ball/b07_wasb_probe.py`](../../../scripts/ball/b07_wasb_probe.py): WASB's own network and checkpoint
(`third_party/WASB-SBDT`, weights from its model zoo; neither committed), its input (3 frames at 512 × 288,
ImageNet normalisation) and its post-processing (heatmap blobs above 0.5); each frame's heatmaps from the 3 windows
that contain it are averaged. Scored exactly as B04 / B05 on the 390 hand-clicked test frames
([`scripts/ball/b05_track_ball.py`](../../../scripts/ball/b05_track_ball.py), [`per_clip.csv`](per_clip.csv)).

## Results (4 test clips, 334 visible balls)

| | found (recall) | precision | elsewhere | answer without a ball |
|---|---|---|---|---|
| our detector alone (B04) | 193 (58 %) | 76 % | 45 | 15 |
| our detector + tracker (B05c) | **261 (78 %)** | 73 % | 62 | 36 |
| **WASB alone** | 134 (40 %) | **100 %** | **0** | **0** |
| WASB + our tracker (B05) | 162 (49 %) | 85 % | 19 | 9 |

Per clip, WASB alone, recall: NCAA 75 %, FIBA Changsha 34 %, FIBA women's WC 20 %, FIBA 2010 25 % (precision 100 % on
all four).

## Reading

1. **WASB never answers on anything but the ball**: not one false alarm in 390 frames, where our single-frame
   detector puts the ball on a painted circle, a table or a yellow shoe. Seeing three frames at once (a still
   look-alike does not move) and predicting a point instead of a box is what our detector lacks.
2. **But it misses most of the balls outside the NBA style**: 75 % found on the NCAA clip (a broadcast close to its NBA
   training footage), 20–34 % on the FIBA clips (other courts, balls, kits). The same change of domain as the jersey
   reader of 033, here for the ball.
3. The two are complementary: ours finds many balls with many false alarms, WASB few balls with none.

## Conclusion

**A clear direction, not a replacement**: WASB as is finds half of our balls; its design removes the false alarms
we struggle with. Next (B08): WASB **fine-tuned** on our hand clicks (the train clips: other leagues, our resolution),
starting from its basketball weights, in Colab; scored on the same test frames.
