# 041 · Jersey-number reader with labels from SportsMOT's own train clips

| | |
|---|---|
| **Date** | 2026-11-03 |
| **Builds on** | [033](../033_jersey_number_detector/) reader trained on NBA footage (E-BARD): 32 % of our pieces read, reads often contradictory |
| **Question** | Do in-domain labels (SportsMOT's resolution and blur, one FIBA game) make the reader work on the validation clips? |
| **Gate** (fixed before running) | on the 15 val clips: a number on **≥ 50 %** of the pieces, two pieces of one player with different numbers **≤ 10 %**, two team-mates with the same number **≤ 5 %** |

> **Data and licences.** Training data: E-BARD (CC BY 4.0) + crops of SportsMOT (CC BY-NC 4.0) with our typed numbers (kept local); starting model: SigLIP base (Apache-2.0); weights not redistributed. All sources: [`docs/data_models_licences.md`](../../docs/data_models_licences.md).

## What was done

1. **Labels** ([`scripts/label_jersey_numbers.py`](../../scripts/label_jersey_numbers.py)): one number typed per
   ground-truth player of the 15 **train** clips (12 crops shown per player): 150 players, 144 with a readable number,
   22 different numbers. Train = 1 FIBA game (women's Asia Cup 2019) + 2 NBA games; no NCAA.
2. **Stubs** ([`scripts/export_jersey_stubs.py`](../../scripts/export_jersey_stubs.py)): up to 4 stubs of 16 clean, tall
   crops per player, E-BARD's format; the last clip of each game held out (65 stubs) to set the confidence threshold.
3. **Training** ([`notebooks/colab_train_jersey_reader_041.ipynb`](../../notebooks/colab_train_jersey_reader_041.ipynb)):
   033's model and recipe from scratch (SigLIP tower, 3 digit heads, 10 epochs), E-BARD + our 305 stubs repeated 3
   times; threshold on E-BARD val + our held-out clips: **0.70** ([`eval_041.json`](eval_041.json),
   [`confidence_curve.png`](confidence_curve.png)).
4. **Gate**: 029's stitching with `--jersey-rule --jersey-reader classifier --jersey-crops 12`, read by
   [`scripts/jersey_coverage.py`](../../scripts/jersey_coverage.py) ([`coverage.txt`](coverage.txt)).

## Results

| | answered | right when answered |
|---|---|---|
| our held-out train clips (same 3 games) | **93.8 %** | **98.4 %** |
| E-BARD test (NBA) | 56.9 % | 90.3 % (033: 53.6 % / 93.8 %) |

On the 15 **validation** clips (other games), against the gate:

| | 033 | **041** | gate |
|---|---|---|---|
| pieces with a number | 32 % | **29.9 %** | ≥ 50 % |
| same player, different numbers | 8 of 30 (27 %) | **8 of 22 (36 %)** | ≤ 10 % |
| two team-mates, same number | 26 of 145 (18 %) | **19 of 103 (18 %)** | ≤ 5 % |
| wrong joins of 029 it would block | 4 of 31 | 3 of 31 (and 1 right join) | |

**Gate not passed on any of the three conditions**: no better than 033.

## Reading

1. **Near-perfect on the held-out train clips, unchanged on new games.** The held-out clips are other moments of the
   same 3 games: same players, same kits. With ~2 stubs per player, the reader most likely learned to recognise
   *these players* (and their kits) rather than to read digits better: what it learned does not carry to the four
   validation games.
2. Making the reader transfer needs labels from **many** games (fonts, kit colours, number positions), not many
   crops of three; SportsMOT's train split has only three basketball games.

## Conclusion

**Negative, not used in the tracker** (the rule stays off). The labels and both scripts are kept: with more games
labelled the same way (about 8 minutes per 150 players), the experiment can be re-run as is.
