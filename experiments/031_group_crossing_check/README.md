# 031 · Identity check for crossings of any number of players

| | |
|---|---|
| **Date** | 2026-11-02 · **negative**: online HOTA 68.8 → 67.2, IDSW 278 → 321; of the identity changes it made, 17 right and 25 wrong |
| **Builds on** | 030 (negative: switches are reciprocal swaps, 81 % within 1 s, but they happen in crowded crossings the pair check refused to decide; train cannot calibrate a margin, natural value 0) |
| **Question** | Deciding every crossing, crowded ones included, how many swaps are undone and how many are created? |

## Method (fixed before running)

`GroupCrossingCheck` in `hooptrack.tracking.crossing_check` (`crossing_check.mode: group`; the comment block before
the class explains it step by step). Differences with 030:

* **groups** of any size: tracks linked by overlaps; a member that disappears stays in the group (its identity can
  be the answer for a visible track);
* **looks before kept per identity** (track id), the last 8 clean crops of the last 4 s, moved with the id when an
  identity is reassigned (030 dropped them after every crossing: in crowded play nobody had two clean crops);
* **looks after per member**, as soon as that member overlaps no other (a group does not wait for every pair in a
  crowd to separate); a group older than 5 s is decided with the members that are complete;
* **decision**: Hungarian assignment of tracks (look after) to identities (look before), mean cosine distance;
  applied only if cheaper than "nothing happened". **No margin** (030: train cannot calibrate one);
* every group is logged with its outcome and, when not decided, why.

Only one change from 011b: the check (same detections, same tracker, ResNet-50 for the tracker's own appearance,
fine-tuned SigLIP of 028 for the check). Then the 029 offline step on top.

**Measured with** IDSW and switches per minute by kind (`switch_report.py`), HOTA / IDF1, the time per frame, and the
group log. Adopted if switches per minute fall on most clips without a loss in HOTA, online and with the offline
step. Team-mate swaps reported separately (the risky case of 015).

Tests: `tests/test_crossing_check.py` (a three-player rotation is undone; nothing changes when the tracker was
right; a lost player's id is given back to the track that took his place).

## Results (15 val clips, `hota.csv`, `switches_per_clip.csv`)

| run | HOTA | AssA | IDF1 | IDSW | switches / min | team-mate / opponent switches |
|---|---|---|---|---|---|---|
| 011b | 68.8 | 55.1 | 72.7 | 278 | 31.9 | 25 / 141 |
| **031** (011b + group check) | 67.2 | 52.7 | 70.8 | 321 | 37.0 | 52 / 157 |
| 029 | 76.9 | 69.1 | 83.9 | 278 | 31.3 | 34 / 176 |
| 031 + 029 offline | 74.6 | 65.1 | 81.0 | 325 | 36.9 | 71 / 189 |

Per clip: HOTA worse on 8 of 15 (better on 1), switches per minute higher on 9 (lower on none). Speed: 1.26 vs
1.82 frames per second on the laptop CPU (the check embeds the crops of every decided group).

**Group log** (`<run>/<clip>/crossings/*.csv`): 292 groups, 2 to 12 players (median 2), 146 kept, **21 reassigned**,
125 not decided (`few_looks`: fewer than two members with a look before, or nobody complete after). The 030 problem
of not deciding is mostly solved.

**Were the reassignments right?** Each reassignment changes the id of some players; a RIGHT one gives a player back
the id he carried before the crossing (and the switch metric counts that as a switch too, so the count alone does
not say). Of the 42 player-level id changes made by the 21 reassignments: **17 back to the id the player had before
the crossing, 25 to an id he never had in it.** More wrong than right; team-mate switches doubled (25 → 52).

## Reading

1. The check now decides most crowded crossings, and still does harm: 60 % of its corrections are wrong, against
   the 98.8 % right decisions of the same model in the 016 benchmark.
2. The benchmark decides with **ground-truth** crops: right identity before, right identity after, boxes that fit
   the player. Online, the "look before" of an identity is whatever crops that id had: after any earlier swap that
   was not undone, the look of an id IS another player, and the check then "corrects" towards the wrong one. The
   boxes come from the detector (partial players, merged players at the edge of an overlap) and only 3 crops after.
   These are plausible causes, not measured ones.

## Conclusion

**Negative, not adopted** (`mode: group` stays available, off). Two online attempts (030, 031) at fixing swaps at
crossing exits with the fine-tuned SigLIP both made things worse. The gap between the benchmark (98.8 %) and the
decisions in the tracker (~40 % right) is the open question; without explaining it, another variant of the check
is a guess.
