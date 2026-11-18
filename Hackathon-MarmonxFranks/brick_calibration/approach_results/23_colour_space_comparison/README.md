# Approach 23: colour-space comparison for Brick 6

Script: `../../evaluate_colour_spaces.py`. 30,000 masked brick pixels per image
(Approach 5 mask; holes and clipped pixels removed), described per channel by a
16-bin histogram, 5 percentiles and the standard deviation (66 features per space).
Classification is leave-one-physical-brick-out with nested C selection, as in all
other approaches.

| Colour space | Brick 6 distance to nearest good / nearest bad | Colour-only image acc. | Bricks | Brick 6 images | Colour + texture image acc. | Bricks | Brick 6 images |
|---|---|---:|---:|---:|---:|---:|---:|
| RGB | 1.67 / 4.88 | 57.4% | 6/10 | 2/24 | 55.0% | 5/10 | 2/24 |
| HSV | 2.45 / 5.04 | 57.8% | 6/10 | 3/24 | 73.3% | 9/10 | 3/24 |
| HLS | 2.40 / 5.45 | 63.2% | 7/10 | 1/24 | 72.5% | 8/10 | 3/24 |
| LAB | 1.90 / 6.39 | 74.0% | 9/10 | 2/24 | 76.4% | 9/10 | 3/24 |
| Luv | 1.83 / 5.40 | 74.4% | 9/10 | 2/24 | 77.9% | 9/10 | 1/24 |
| YCrCb | 1.89 / 5.90 | 80.6% | 9/10 | 3/24 | 78.7% | 9/10 | 2/24 |
| XYZ | 1.63 / 4.66 | 64.0% | 7/10 | 4/24 | 62.4% | 7/10 | 4/24 |
| rgb chromaticity | 2.24 / 5.19 | 73.6% | 8/10 | 7/24 | 79.8% | 8/10 | 4/24 |
| Opponent | 1.84 / 5.95 | 77.9% | 9/10 | 3/24 | 79.8% | 9/10 | 2/24 |
| Log chromaticity | 1.94 / 6.32 | 77.5% | 8/10 | 3/24 | 79.8% | 9/10 | 3/24 |

## Conclusion

- In every colour space, Brick 6 is 2.2-3.4 times closer to the good bricks than
  to the bad bricks, and its mean classifier score is on the good side.
  No colour space detects Brick 6 at brick level; the best single-image result is
  7/24, from a model that also misclassifies two other bricks.
- Single-statistic screen (`brick6_single_feature_screen.csv`): of 180 channel
  percentiles/spreads, Brick 6 is inside the good-brick range on 148 and outside
  by at most 0.26 times ordinary photo-to-photo variation on the rest. Several of
  those point the opposite way to the other bad bricks (Brick 6 is slightly redder).
- Colour spaces are re-arrangements of the same RGB values, so they cannot add
  information that is not in the pixels. Brick 6's photographed colour matches
  the good bricks.
- Side finding: colour-only features in YCrCb, opponent, LAB, Luv and log
  chromaticity reach 74-81% image accuracy with 9/10 or 8/10 bricks correct,
  close to Approach 5 without any texture features.
- Masks use randomly initialised GrabCut, so rerunning the script changes these
  accuracies by up to about 4 points; differences between the better spaces are
  within that noise.
