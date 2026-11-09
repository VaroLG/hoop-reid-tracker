# B08 · WASB fine-tuned on our hand clicks

| | |
|---|---|
| **Date** | 2026-11-05 |
| **Builds on** | [B07](../B07_wasb_zero_shot/): WASB with its NBA weights, 40 % of the balls at 100 % precision · [B05b/c](../B05b_static_and_weak/): ours, 78 % at 73 % |
| **Question** | Starting from WASB's basketball weights, do our 1,245 clicks from other leagues keep its precision and recover the balls it misses? |

> **Data and licences.** Training data: SportsMOT (CC BY-NC 4.0) train frames with our hand clicks (kept local); starting model: WASB basketball weights (MIT repository); the fine-tuned weights are not redistributed (CC BY-NC 4.0 if they were). All sources: [`docs/data_models_licences.md`](../../../docs/data_models_licences.md).

## Method

* **Data** ([`scripts/ball/b08_export_wasb_data.py`](../../../scripts/ball/b08_export_wasb_data.py)): every clicked train
  frame with two neighbours on each side, at WASB's input size (512 × 288); each click gives 3 windows of 3 frames in
  which only the clicked frame has a target (the other two outputs are left out of the loss); frames with no visible
  ball are kept as empty targets. The last clip of each train game is held out.
* **Training** ([`notebooks/colab_finetune_wasb_ball.ipynb`](../../../notebooks/colab_finetune_wasb_ball.ipynb)): WASB's
  HRNet, target (disk of radius 2.5 px) and quality focal loss, Adam 1e-4 with cosine decay, flips and mild colour
  changes, Colab T4 (~4 min per epoch). Stopped by hand after epoch 22 (plateau); best epoch on the held-out clicks:
  16 ([`history.json`](history.json)): F1 0.855 → **0.922** (recall 77 → 87 %, precision 97 → 98.5 %).
* **Test**: [`scripts/ball/b07_wasb_probe.py`](../../../scripts/ball/b07_wasb_probe.py) with the new weights, then
  [`scripts/ball/b05_track_ball.py`](../../../scripts/ball/b05_track_ball.py) (B05's tracker as is), scored as B04–B07
  on the 390 clicked test frames ([`per_clip.csv`](per_clip.csv)).

## Results (4 test clips, 334 visible balls)

| | found (recall) | precision | F1 | elsewhere | answer without a ball |
|---|---|---|---|---|---|
| our YOLO + tracker (B05c, until now) | **261 (78 %)** | 73 % | 0.75 | 62 | 36 |
| WASB, NBA weights (B07) | 134 (40 %) | 100 % | 0.57 | 0 | 0 |
| **WASB fine-tuned (B08), alone** | 176 (53 %) | **99.4 %** | 0.69 | **0** | **1** |
| **WASB fine-tuned + tracker** | 237 (71 %) | 85 % | **0.78** | 24 | 17 |

F1 per clip, B05c → WASB fine-tuned + tracker: NCAA 0.87 → 0.87, FIBA Changsha 0.79 → 0.79, FIBA women's WC
0.62 → **0.66**, FIBA 2010 0.74 → **0.76**: equal or better on all four.

## Reading

1. **Fine-tuning keeps WASB's precision and adds recall where it was missing**: alone, 40 → 53 % found with still
   almost no false alarm; on the FIBA clips 20–34 % → 37–53 %.
2. **With the tracker it is the best ball localisation of the project**: F1 0.78, with 85 % of the answers right
   (73 % for our YOLO pipeline) and far fewer answers on still look-alikes (elsewhere 62 → 24). It finds a few fewer
   balls than B05c (71 vs 78 %): the trade is fewer, surer answers, which is what possession needs.
3. Our tracker was designed for a noisy detector; with WASB's clean candidates its static filter is not needed.
   The remaining misses are mostly on the FIBA women's World Cup clip (55 % found).

## Conclusion

**Positive: adopted as the ball localisation** (WASB fine-tuned + B05 tracker). Code: WASB is MIT (cited, kept in
`third_party/`, not committed); the fine-tuned weights stay out of the repo (`weights/`). Next: possession (P01)
again with this ball, which is right 85 % of the time instead of 73 %.
