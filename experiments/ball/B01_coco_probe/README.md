# B01 · Does a COCO detector already see the basketball?

| | |
|---|---|
| **Date** | 2026-11-04 |
| **Question** | SportsMOT has no ball labels. Can the COCO-pretrained YOLO26m (class `sports ball`, our 001 baseline) find the ball well enough to pseudo-label the training frames? |
| **Clips** | 4 validation clips, one per arena; every 10th frame; input 640 and 1280 px |
| **Script** | [`scripts/ball/b01_probe_coco_ball.py`](../../../scripts/ball/b01_probe_coco_ball.py) |

> **Data and licences.** Frames: SportsMOT (CC BY-NC 4.0); model used as is: YOLO26m COCO weights (Ultralytics, AGPL-3.0). All sources: [`docs/data_models_licences.md`](../../../docs/data_models_licences.md).

## Results ([`probe.csv`](probe.csv))

| clip | frames with a detection, 640 | dets / frame, 640 | frames with a detection, 1280 | dets / frame, 1280 |
|---|---|---|---|---|
| NCAA c001 | 44 % | 0.62 | 89 % | **3.44** |
| FIBA Changsha c001 | 8 % | 0.08 | 17 % | 0.18 |
| FIBA women's WC c003 | 43 % | 0.57 | 42 % | 0.59 |
| FIBA 2010 c012 | 16 % | 0.16 | 26 % | 0.29 |

There is one ball: more than one detection per frame means false alarms. Precision from the sheets (best detection
of random frames): [`sheet_640.jpg`](sheet_640.jpg), [`sheet_1280.jpg`](sheet_1280.jpg).

## Reading

1. **Recall is low**: on three of four arenas the ball is found in 8–43 % of the sampled frames (it is visible in
   most of them).
2. **At 1280 px the false alarms are confident**: neon-yellow objects in the stands (cameras, signs, the Australian
   team's shoes) score 0.5–0.7, as high as real balls; on NCAA c001, 3.4 detections per frame.
3. At 640 px the confident detections (≳ 0.45) are mostly real balls, but they are few.

## Conclusion

**Negative: not good enough to pseudo-label.** The ball needs labels from a person: a click tool on the train
frames, with this detector's suggestions (restricted to the court) only as a shortcut to accept.
