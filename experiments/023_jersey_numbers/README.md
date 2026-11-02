# 023 · Jersey numbers read zero-shot (Florence-2): right, but too rare

| | |
|---|---|
| **Date** | 2026-10-28 · **neutral**: 15 val clips 81.8 → 81.5 (023), 81.7 (023b) |
| **Builds on** | 021 (what is left after 020 is the team-mate problem) and 022 (a generic text recogniser cannot read jersey numbers) |
| **Question** | Can a small vision-language model read the numbers with no training, and does "never join two different numbers" fix the wrong joins between team-mates? |

## The reader, measured on train

Florence-2-base (Xiao et al. 2024, 0.23 B parameters, many vision tasks including OCR), task `<OCR>` on the upper body
(5–55 % of the box height), greedy decoding. A read counts when the whole answer is 1–2 digits; confidence = product of
the token probabilities. `hooptrack.reid.jersey`. Train split, 6 tallest clean crops per ground-truth player (900 crops,
150 players), `outputs/_review/jersey/florence_reads.py`:

| confidence ≥ | crops read | players with a number | reads agreeing along a player | same-team clashes |
|---|---|---|---|---|
| 0.5 | 20 % | 55 % | 92 % | 2 |
| **0.7** | 15 % | 44 % | **96 %** | **0** |
| 0.85 | 11 % | 34 % | 99 % | 0 |

40 random reads at ≥ 0.7, checked by eye ([`reads_checked.jpg`](reads_checked.jpg)): **36 right** (#12, #22, #31, #37 wrong). Two of the 4 wrong ones are partial reads of a 2-digit number
("17" → "7", "31" → "3": a fold or an arm hides one digit), two are misreads ("21" → "27", "5" → "3"). Cost: ~3 s per crop on 2 CPU cores.

A piece gets a number when it has at least 2 reads ≥ 0.7 and 2/3 of them agree (`tracklet_number`), fixed from these train numbers.

## In the pipeline (15 val clips)

020 (team cut, SigLIP labels) + `stitch_gta(..., numbers=...)`: two groups with different known numbers are never merged.
`--jersey-rule` in `scripts/stitch_tracklets.py`; reads cached per box in `<out>/<clip>/jersey_reads.csv`.

| | IDF1 | MOTA | ID switches |
|---|---|---|---|
| 020: team cut, SigLIP | **81.8** | 96.6 | 267 |
| 023: + jersey rule | 81.5 | 96.6 | 272 |
| 023b: + jersey rule, partial reads compatible ("3" ~ "23") | 81.7 | 96.6 | 269 |

023 changes 2 clips, both NCAA, both worse (c011 67.2 → 62.9, c007 73.9 → 73.0). In both, one player was read twice with
different numbers: "3" and "23" in c011, "0" and "23" in c007. The rule then refused right merges. 023b treats a number
contained in the other as a partial read. It was added **after** seeing these val clips, so it is a post-hoc fix, but partial
reads were already among the errors seen on train. It recovers c011 and not c007.

## Why it cannot help much: coverage

`outputs/_review/jersey/merges023.py`, the merges of 020 by whether the pieces have a number:

| merges of 020 | no number | one side numbered | both sides numbered |
|---|---|---|---|
| right | 28 | 35 | 7 |
| **wrong** | 23 | 12 | **2** |

Only 74 of 334 pieces (22 %) get a number on val, fewer than the 44 % of players on train: pieces are shorter than whole
players, and broadcast val clips show fewer backs in close-up. **A veto can only act when both sides have a number: 2 of the
37 wrong merges.** That is the ceiling of this rule with this reader.

## Conclusion

**Neutral, not adopted.** The reader is right when it answers (96 % agreement, 36/40 by eye), but it answers too rarely for
the pieces that go wrong. `--jersey-rule` and `--jersey-substring` stay available, off. To make numbers count, the
coverage has to rise a lot. That means more crops per piece (cost), or a reader trained on jersey crops (pseudo-labels from
these confident reads on train, fine-tuned in Colab), so that it reads the small, blurred and side-on numbers too.
