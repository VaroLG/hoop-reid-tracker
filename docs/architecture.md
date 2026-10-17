# Architecture

## Data flow

```mermaid
flowchart LR
    F["Frame t"] --> D["Detector<br/>YOLO / cached / oracle"]
    D -->|boxes, scores| R["ReID extractor<br/>OSNet / ResNet / histogram"]
    F --> G["Camera motion<br/>sparse optical flow + RANSAC"]
    D --> T
    R -->|embeddings| T["ReIDTracker"]
    G -->|2x3 affine| T
    T --> O["MOT file<br/>frame,id,x,y,w,h,score"]
    O --> E["Evaluation<br/>IDF1 · IDSW · HOTA/AssA · re-entry recovery"]
```

## Per-frame association cascade

Each stage only sees what the previous stages could not explain, and each one
is stricter about what it accepts.

```mermaid
flowchart TD
    P["0 · Predict every track<br/>Kalman + camera warp"] --> S1
    S1["1 · HIGH dets vs TRACKED+LOST<br/>cost = λ·appearance + (1-λ)·(1-IoU)<br/>spatial gate + appearance gate"] --> S2
    S2["2 · LOW dets vs still-unmatched TRACKED<br/>IoU only — occluded players"] --> S3
    S3["3 · HIGH dets vs TENTATIVE<br/>IoU — confirm or kill<br/>on promotion: one more memory check"] --> S4
    S4["4 · Re-identification<br/>clean HIGH dets vs ALL absent identities<br/>LOST: appearance + reachable-distance gate<br/>DORMANT: appearance only<br/>joint Hungarian + ambiguity margin"] --> S5
    S5["5 · Birth<br/>confident leftovers → TENTATIVE"] --> S6
    S6["6 · Bookkeeping<br/>TRACKED→LOST · LOST→DORMANT · prune memory"]
```

## Track life cycle

```mermaid
stateDiagram-v2
    [*] --> Tentative: confident unexplained detection
    Tentative --> Tracked: min_hits matches (+ memory check)
    Tentative --> Removed: missed once
    Tracked --> Lost: no match this frame
    Lost --> Tracked: stage 1 or stage 4 match
    Lost --> Dormant: lost > max_time_lost
    Dormant --> Tracked: revived by appearance (stage 4 / promotion)
    Dormant --> Removed: older than memory_max_age
```

## Module map

| Module | Responsibility |
|---|---|
| `tracking/kalman.py` | constant-velocity Kalman filter in (cx, cy, w, h), camera-motion warp |
| `tracking/matching.py` | IoU, coverage, cosine distance, gated Hungarian, motion/appearance fusion |
| `tracking/gmc.py` | global motion compensation (players masked out) |
| `tracking/track.py` | single-track state machine and appearance model (EMA + gallery) |
| `tracking/memory.py` | dormant-identity gallery, nearest-neighbour distance, ambiguity margin |
| `tracking/tracker.py` | the association cascade above |
| `detection/` | YOLO wrapper, detection cache, ground-truth oracle |
| `reid/` | colour-histogram, ResNet-50 and OSNet extractors |
| `eval/` | motmetrics (IDF1, IDSW…), TrackEval (HOTA/AssA), re-entry recovery metric |
| `pipeline.py` | per-video loop with per-stage latency |
| `cli.py` | `hooptrack detect / track / evaluate` |
