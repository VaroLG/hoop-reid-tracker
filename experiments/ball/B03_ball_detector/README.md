# B02–B04 · Hand-labelled ball, a ball detector of our own, and its test

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Builds on** | [B01](../B01_coco_probe/): the COCO detector is not good enough to pseudo-label the ball |
| **Question** | With a few hundred clicks, can a detector of our own find the ball on games it has never seen? |

## What was done

* **B02 · labels** ([`scripts/ball/b02_label_ball.py`](../../../scripts/ball/b02_label_ball.py)): one click on the
  ball's centre (or "not visible") in one frame out of ten. **Train**: the 15 basketball train clips, 1,245 frames,
  ball visible in 84 %. **Test**: the 4 validation clips of B01 (one per arena), 390 frames, visible in 86 %. No box
  is drawn: the diameter is 0.12 × the height of the nearest players
  ([`hooptrack/ball/labels.py`](../../../src/hooptrack/ball/labels.py); checked by eye on 16 random clicks).
* **B03 · detector** ([`scripts/ball/b03_build_ball_dataset.py`](../../../scripts/ball/b03_build_ball_dataset.py),
  [`notebooks/colab_train_ball_detector.ipynb`](../../../notebooks/colab_train_ball_detector.ipynb)): YOLO26s,
  COCO-pretrained, fine-tuned at **1280 px** on the train clicks (1,056 frames; the last clip of each train game for
  the best epoch; frames without the ball as negatives). A separate model: the player detector and every tracking
  result stay as they are.
* **B04 · test** ([`scripts/ball/b04_eval_ball.py`](../../../scripts/ball/b04_eval_ball.py)): on the validation
  clicks, the most confident box of each frame; **found** = within one ball diameter of the click.

## Results (4 validation clips, 390 frames, 334 with the ball; confidence 0.25)

| | found (recall) | right when it answers (precision) | elsewhere | missed | false alarm without a ball | median error |
|---|---|---|---|---|---|---|
| COCO YOLO26m, `sports ball` | 37 (11 %) | 29 % | 76 | 221 | 13 | 2.5 px |
| **ours, YOLO26s 1280** | **193 (58 %)** | **76 %** | 45 | 96 | 15 | 2.5 px |

Per clip, recall / precision, COCO → ours ([`per_clip_coco_1280.csv`](per_clip_coco_1280.csv),
[`per_clip_ball_b03.csv`](per_clip_ball_b03.csv)):

| NCAA c001 | FIBA Changsha | FIBA women's WC c003 | FIBA 2010 c012 |
|---|---|---|---|
| 23 / 24 % → **76 / 92 %** | 0 / – → **57 / 80 %** | 6 / 23 % → **44 / 54 %** | 16 / 100 % → **53 / 88 %** |

Trade-off over confidences ([`sweep_ball_b03.csv`](sweep_ball_b03.csv)): 64 % found at 65 % precision (0.05), 51 % at
85 % (0.40), 30 % at 94 % (0.50). No answer above 0.60: confidences are low but well ordered.

## Reading

1. **Five times more balls found, and far fewer wrong answers**, on all four arenas, none of them seen in training.
   1,245 clicks (about 40 minutes) were enough.
2. The hardest clip is the FIBA women's World Cup (44 % found, 27 answers elsewhere): the yellow kits and neon
   shoes look like the ball's colour, and players handle it close to the body.
3. Most misses are a ball in the hands, half hidden or blurred in a pass: one frame alone cannot tell. That is the
   tracker's job: a ball follows physics between two detections.

## Conclusion

**Positive: adopted as the ball detector** (`weights/ball_yolo26s_1280.pt`, not committed). Next: a ball tracker that
fills the gaps between detections and drops the isolated false alarms (B05), then possession (nearest player).
