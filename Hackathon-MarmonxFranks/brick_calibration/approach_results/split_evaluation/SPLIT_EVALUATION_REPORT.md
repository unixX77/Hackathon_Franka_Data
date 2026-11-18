# Approaches 5 and 17 on brick-disjoint train / validation / test splits

Splits: `/cache/s95vkond/brick_splits` (see `SPLIT_PROTOCOL.md` there).
Script: `evaluate_on_splits.py`. Per split: SVM C is chosen on the validation
bricks, the model is refit on train + validation, and the test bricks are
scored once. Approach 17 keeps its fixed 50/50 score average; the decision
threshold stays at 0.

Training uses 7 bricks (6 without Brick 6), against 9 in leave-one-brick-out
(LOBO), so somewhat lower numbers are expected.

## Headline results (pooled over all test sets)

| | LOBO | Splits, Brick 6 set aside | Splits, Brick 6 included |
|---|---:|---:|---:|
| **Approach 5** image accuracy | 79.5% | 85.1% | 80.7% |
| Approach 5 balanced accuracy | 81.4% | 84.6% | 80.3% |
| Approach 5 brick accuracy | 90% | 90% | 92% |
| Approach 5 Brick 6 images correct | 4% (1/24) | 9% (set aside, 20 models) | 6% (in test 4 times) |
| **Approach 17** image accuracy | 82.2% | 85.5% | 81.9% |
| Approach 17 balanced accuracy | 83.5% | 85.0% | 81.6% |
| Approach 17 brick accuracy | 90% | 90% | 90% |
| Approach 17 Brick 6 images correct | 4% (1/24) | 11% (set aside, 20 models) | 7% (in test 4 times) |

Single-split accuracy ranges from about 52% to 100% depending on which two bricks
are in test, so a single split is not a reliable estimate.

No model in either family called Brick 6 bad at brick level (0/20 set-aside
models; 0/4 test appearances).

## Like-for-like comparison (Brick 6 removed from every test set)

The "set aside" column above looks best partly because Brick 6, which every model
misses, is never tested. Matched numbers (`matched_comparison.csv`):

| Balanced accuracy | LOBO excl. Brick 6 | 20 shared pairs, Brick 6 in training | Same pairs, Brick 6 set aside |
|---|---:|---:|---:|
| Approach 5 | 87.6% | 85.9% | 84.6% |
| Approach 17 | 90.3% | 87.5% | 85.0% |

Approach 17 is ahead of Approach 5 in every matched comparison.

## Effect of keeping Brick 6 in training (same test and validation bricks)

| Test-image accuracy | A5: B6 in training | A5: B6 set aside | A17: B6 in training | A17: B6 set aside |
|---|---:|---:|---:|---:|
| Brick_8_bad | 59% | 30% | 91% | 39% |
| Brick_10_bad | 70% | 64% | 90% | 80% |
| Brick_5_bad | 87% | 81% | 62% | 59% |
| Brick_2_good | 81% | 91% | 77% | 93% |
| Brick_3_good | 84% | 95% | 79% | 93% |

Brick 8 is a dark, low-redness bad brick. Without Brick 6 every bad training brick
is pale or sandy, so the models learn "dark = good" and miss Brick 8. Brick 6, a
dark bad brick, prevents that shortcut and pushes the models towards redness. It
also makes them stricter, which costs some good-brick photos. The net effect is
+1.3 (Approach 5) and +2.5 (Approach 17) points of balanced accuracy.

## Files

- `comparison.csv`, `matched_comparison.csv`, `summary.json`
- `<family>/per_split_results.csv`: chosen C, validation accuracy and test result per split
- `<family>/test_image_predictions.csv`: every test (and set-aside) image score per split

## Train, validation and test accuracy (`score_train_val_test.py`)

Image accuracy pooled over all splits. Train and validation scores come from the
model fitted on the training bricks; test scores come from the model refitted
on train + validation.

| Family | Approach | Train | Validation | Test |
|---|---|---:|---:|---:|
| Brick 6 set aside | 5 | 99.2% | 89.1% | 85.1% |
| Brick 6 set aside | 17 | 99.8% | 89.1% | 85.5% |
| Brick 6 included | 5 | 90.2% | 87.3% | 80.7% |
| Brick 6 included | 17 | 94.5% | 86.1% | 81.9% |

- The train-to-test gap of 14-15 points without Brick 6 shows the models largely
  memorise the 5-6 training bricks; more physical bricks matter more than model changes.
- With Brick 6 included, training accuracy falls to 90-95% because Brick 6 cannot
  be fitted: it carries a "bad" label but has good-brick colour.
- Validation is above test because the validation bricks were used to choose C.
- Approach 17 has the best test score in both families, by 0.4 and 1.2 points.

Per-split values: `<family>/train_val_test_scores.csv`; totals: `train_val_test_summary.json`.
