# Chapter 3 · Possession

Who has the ball, frame by frame, from the tracked ball (B08) and the tracked players (038). Scored against 100
catches and releases marked by hand on the 4 test clips; values are chosen on 3 held-out train clips (37 marks).

| step | what | result |
|---|---|---|
| [P01](P01_rules/) | rules: the closest player within reach, moving with the ball | 80 % of the events found, 3× too many: negative per frame |
| [P01b](P01b_tuned_gaps/) | short gaps filled, values from train | neutral |
| [P02](P02_ball_motion/) | "air" from the ball's own motion | negative; with a perfect ball the rules reach F1 0.46 |
| [P03](P03_better_ball/) | the rules on the B08 ball | event F1 0.41 → 0.50 |
| [P04](P04_keep_state/) | **the last state kept while the ball is not seen** | **86 % of the frames right: adopted** |
| [P05](P05_temporal_model/) | learned score + two-state temporal model | 87.9 % of frames, event F1 0.43: mixed |

**In the final pipeline:** P04. See [the final selection](../../README.md#the-final-selection).

**Data and licences.** Frames and player tracks: SportsMOT (CC BY-NC 4.0). The marked events are ours and stay local
(`data/possession_labels/`); the marking tool is `scripts/possession/p02_label_events.py`. P05 uses scikit-learn
(BSD-3-Clause). All sources: [`docs/data_models_licences.md`](../../docs/data_models_licences.md).
