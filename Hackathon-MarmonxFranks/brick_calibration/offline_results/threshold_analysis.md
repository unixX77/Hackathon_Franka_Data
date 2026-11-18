# Offline SAM3-mask score threshold analysis

Source: `all_predictions.csv`, 258 photos of 10 physical bricks. The frozen
Approach 17 `score_bad` was used. A binary policy assigns **good** when
`score_bad <= threshold` and **bad** otherwise. These images are from the same
physical bricks used to train Approach 17, so all percentages below are
in-sample, and the threshold is optimized on the same images. They are not
estimates of new-brick or live-camera accuracy.

There are 109 good photos from four bricks and 149 bad photos from six bricks.
An exhaustive sweep over every observed score found that **no single binary
threshold exceeds 90% accuracy on all 258 photos**. The maximum is 221/258,
or 85.66%, for any threshold in `[-1.442229, -1.420629)`. A practical value
is `-1.44`.

| Binary threshold | Correct | Overall | Good correct | Bad correct | Bad called good |
|---:|---:|---:|---:|---:|---:|
| -2.00 (current) | 207/258 | 80.23% | 67/109 | 140/149 | 9 |
| -1.50 | 217/258 | 84.11% | 88/109 | 129/149 | 20 |
| -1.44 (best total) | 221/258 | 85.66% | 92/109 | 129/149 | 20 |
| -1.34 (balanced recalls) | 220/258 | 85.27% | 93/109 | 127/149 | 22 |
| -1.00 | 215/258 | 83.33% | 100/109 | 115/149 | 34 |
| 0.00 | 190/258 | 73.64% | 108/109 | 82/149 | 67 |

At -1.44, good recall is 84.40% and bad recall is 86.58%. At -2.0, good
recall is 61.47% and bad recall is 93.96%. The best overall threshold thus
raises the number of bad photos called good from nine to 20.

| Physical brick | Photos | Correct at -2.0 | Correct at -1.44 |
|---|---:|---:|---:|
| Brick 1 good | 26 | 15 | 22 |
| Brick 2 good | 30 | 18 | 26 |
| Brick 3 good | 28 | 13 | 20 |
| Brick 4 good | 25 | 21 | 24 |
| Brick 5 bad | 25 | 25 | 24 |
| Brick 6 bad | 24 | 23 | 20 |
| Brick 7 bad | 24 | 24 | 24 |
| Brick 8 bad | 24 | 19 | 16 |
| Brick 9 bad | 24 | 23 | 21 |
| Brick 10 bad | 28 | 26 | 24 |

## If over 90% accuracy is required

Two thresholds can produce good and bad labels only on more confident scores.
Scores in the middle receive **review/uncertain**, so accuracy is calculated
only on decided photos. This does not meet a requirement to label *every*
photo good or bad.

| Good if score <= | Bad if score > | Correct / decided | Accuracy on decided | Coverage | Review |
|---:|---:|---:|---:|---:|---:|
| -1.75 | -1.20 | 194/215 | 90.23% | 83.33% | 43 |
| -2.00 | -1.00 | 182/200 | 91.00% | 77.52% | 58 |
| -1.95 | -0.88 | 179/194 | 92.27% | 75.19% | 64 |
| -2.50 | -0.50 | 144/152 | 94.74% | 58.91% | 106 |

The score ranges overlap: the maximum good score is 2.150073 and the minimum
bad score is -3.646347. Threshold tuning cannot separate them perfectly.
If all available orientations of each familiar brick are aggregated by the
median score, -1.44 classifies all 10/10 physical bricks correctly. That is a
different task from classifying each of the 258 individual photos and uses
multiple views of bricks already seen during model fitting.
The Approach 5 component alone peaks at 223/258 (86.43%) with its own score
threshold -0.505996; the DINOv2 component peaks at 204/258 (79.07%) with
threshold -1.525663. Neither reaches 90% on every photo.

Recommendation: retain -2.0 when missing a bad brick is the primary concern.
For a demonstration where total photo accuracy matters more, test -1.44 as a
separate policy. Use a two-threshold policy only if the live workflow can
handle review/uncertain predictions. Calibrate any selected policy on new,
independently labeled camera frames before interpreting these percentages as
live performance.
