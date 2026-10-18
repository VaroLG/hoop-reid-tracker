# Detection vs tracking diagnosis

Detections: `outputs/detections/yolo26m` · split: `val` · sequences: 1 · filters: `court_mask`

## v_00HRwkvvjtQ_c001

- Ground truth: 9.7 players/frame, 10 identities, 1162 frames
- Recall at the lowest kept score: **97.4 %**
- Median box height — players: 144 px · false positives (score ≥ 0.5): 79 px

| score ≥ | detections/frame | true positives | false positives | precision |
|---|---|---|---|---|
| 0.1 | 57.8 | 10982 | 56204 | 16.3 % |
| 0.3 | 29.7 | 10625 | 23938 | 30.7 % |
| 0.5 | 17.6 | 10305 | 10169 | 50.3 % |
| 0.6 | 13.9 | 10011 | 6154 | 61.9 % |

### After filters: CourtMask

- Players kept (score ≥ 0.5): **100.0 %** · false positives removed (score ≥ 0.5): **71.0 %**
- Recall at the lowest kept score: 97.4 %

| score ≥ | detections/frame | true positives | false positives | precision |
|---|---|---|---|---|
| 0.1 | 16.5 | 10982 | 8246 | 57.1 % |
| 0.3 | 13.3 | 10625 | 4811 | 68.8 % |
| 0.5 | 11.4 | 10305 | 2944 | 77.8 % |
| 0.6 | 10.6 | 10011 | 2337 | 81.1 % |

Filter stats: court_frames_trusted = 1162, court_frames_passthrough = 0, court_boxes_removed_% = 71.4

Motion-only tracker (no ReID, no GMC, no memory):

| input boxes | IDF1 | MOTA | IDSW | ids created | re-entries |
|---|---|---|---|---|---|
| A · detector as is | 36.1 | -25.0 | 44 | 132 | 0/4 |
| A+F · detector + filters | 51.5 | 69.1 | 44 | 54 | 0/4 |
| B · only boxes that match a player | 64.4 | 94.3 | 42 | 24 | 0/4 |
| C · ground-truth boxes | 80.3 | 99.8 | 9 | 17 | 0/6 |
