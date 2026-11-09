# Chapter 3 · Ball

Find the ball and follow it through the clip. All scores on the 4 test clips (SportsMOT validation, never used for
training or choosing values), against 390 hand-clicked frames.

| step | what | result |
|---|---|---|
| [B01](B01_coco_probe/) | COCO detector, "sports ball" | 11 % found, 29 % precision: negative |
| [B02–B04](B03_ball_detector/) | hand clicks + our YOLO26s at 1280 px | 58 % at 76 %: adopted then |
| [B05](B05_ball_tracker/) | one ball through the clip (dynamic programming) | 73 % at 66 %: mixed |
| [B05b / B05c](B05b_static_and_weak/) | still look-alikes removed, weak segments dropped | 78 % at 73 %, F1 0.75 |
| [B07](B07_wasb_zero_shot/) | WASB as published (NBA weights) | 40 % at 100 % |
| [B08](B08_wasb_finetuned/) | **WASB fine-tuned on our clicks + the tracker** | **71 % at 85 %, F1 0.78: adopted** |
| [B09](B09_fused_candidates/) | WASB + YOLO candidates in one tracker | F1 0.778: null |

**Data and licences.** Frames: SportsMOT (CC BY-NC 4.0). The hand clicks are ours and stay local (`data/ball_labels/`);
the click tool is `scripts/ball/b02_label_ball.py`. Models: YOLO26 (Ultralytics, AGPL-3.0), WASB
(github.com/nttcom/WASB-SBDT, MIT, kept in `third_party/`, not committed). No weights are stored here. All sources:
[`docs/data_models_licences.md`](../../docs/data_models_licences.md).
