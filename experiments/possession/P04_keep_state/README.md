# P04 · Frames without a ball keep the last state

> **In the final pipeline:** possession. See [the final selection](../../../README.md#the-final-selection).

| | |
|---|---|
| **Date** | 2026-11-05 |
| **Builds on** | [P03](../P03_better_ball/): P01's rules on the B08 ball, event F1 0.50; 88 % of the frames right where the system has a ball, but frames without a ball counted as "air" |
| **Question** | If a ball we do not see keeps its last state (held by the same player, or still in the air), is possession right on the whole clip? |

## Change (one value, off by default; nothing to tune)

`PossessionConfig.keep_state` ([`hooptrack/ball/possession.py`](../../../src/hooptrack/ball/possession.py), test in
`tests/test_possession.py`): frames without a ball keep the last state, however long the ball stays unseen (P01:
the holder for 10 frames at most, then "no ball"); frames before the first ball take the first state.
Same B08 ball, same 038 player tracks, same rules otherwise ([`per_clip.csv`](per_clip.csv)).

## Results (4 test clips, 3,883 frames; "always held" gets 78.6 % right)

| | frames right (held / air) | event F1 |
|---|---|---|
| P03 (no ball = air) | 73.7 % | 0.50 |
| **P04 (keep the state)** | **86.1 %** | 0.50 |

Per clip, frames right, P03 → P04 (always held): NCAA 76 → **93 %** (86), Changsha 82 → **86 %** (64), FIBA women's WC
62 → 79 % (81), FIBA 2010 82 → **89 %** (81). The events do not change (no change is created or removed: the state
is only carried through frames where nothing was seen).

## Reading

1. **For the first time possession beats the trivial answer frame by frame** (86 % vs 79 %), on three of four clips;
   on the FIBA women's World Cup it stays just under it (79 vs 81 %), the clip where the ball is hardest to find.
2. The events are where the work is left: 78 % of the catches and releases found within 2 frames, but about two
   invented for each real one.

## Conclusion

**Positive: adopted** (`keep_state=True` in the possession reference). Possession now: right in 86 % of the frames,
event F1 0.50. Next: a temporal model learned from examples to remove the invented changes.
