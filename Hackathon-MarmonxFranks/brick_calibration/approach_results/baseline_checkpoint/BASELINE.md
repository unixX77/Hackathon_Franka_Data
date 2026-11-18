# Official live baseline: SAM3 + Approach 17 with review band

Equal-weight late fusion of two linear SVMs:

- **Approach 5 branch:** LAB colour histograms + texture statistics (52 features).
- **DINOv2 branch:** frozen `facebook/dinov2-small` (revision
  `ed25f3a31f01632728cabb09d1542f84ab7b0056`, bundled in `dinov2_small/`);
  patch-token mean/std/P90 → PCA 32 → SVM.

`score_bad` is the mean of the two SVM decision values. The live baseline
uses a SAM3 brick mask and `review_band_175_120`: good at score <= -1.75,
uncertain at -1.75 < score <= -1.20, and bad at score > -1.20. Its settings
are recorded in `live_baseline.json`. SAM3 selects up to ten brick masks per
frame. The live baseline classifies each frame independently.

The lower-level `FusionClassifier.classify()` API still returns the historical
zero-centered margin label. Use `score_bad` with `threshold_policies.policy`
and `review_band_175_120` to obtain the live baseline label. The historical
`classify_burst` method uses a median score; the SAM3 viewer does not perform
burst aggregation.

Replaces Approach 5 (`../05_approach_checkpoint/`, kept unchanged).

## Historical performance on unseen bricks

These figures use the original automatic masks and decision rule; they do not
measure the live baseline's SAM3 masks or review threshold.

| Evaluation | Approach 17 | Approach 5 |
|---|---:|---:|
| Leave-one-brick-out image accuracy | 82.2% (212/258) | 79.5% (205/258) |
| Leave-one-brick-out bad-image recall | 75.2% | 69.1% |
| Bricks correct (leave-one-brick-out) | 9/10 | 9/10 |
| Train/val/test splits, all 10 bricks: test accuracy | 81.9% | 80.7% |

Training accuracy on the splits is 94.5-99.8%, indicating a substantial gap to
the held-out results. `Brick_6_bad` was misclassified after brick-level score
aggregation in the held-out test. The lead over Approach 5 is not
statistically significant on ten bricks.

For the SAM3 masks and review policy, 194 of 215 decided offline photos were
correct and 43 of 258 were uncertain. Those photos are from the ten bricks
used to fit the saved model. The policy was selected using the same scores,
so these figures do not measure accuracy on new bricks or on the live camera.

## Usage

```python
from fusion_checkpoint import FusionClassifier
classifier = FusionClassifier.load("approach_results/baseline_checkpoint")
classifier.classify(rgb)               # {"label": "good" | "bad" | "uncertain", "score_bad": ...}
classifier.classify_burst([rgb, ...])  # median over stationary frames
```

For the baseline label from a SAM3 mask, apply the saved policy to the score:

```python
from threshold_policies.policy import label_for, load_policy
result = classifier.classify(rgb, mask=sam3_mask)
label = label_for(result["score_bad"], load_policy("review_band_175_120"))
```

No network access or Hugging Face cache is needed. Run from `brick_calibration/`
(the loader imports the feature code in `evaluate_approaches.py` and
`evaluate_dino_approaches.py`).

## Rebuild

```bash
HF_HOME=/cache/s95vkond/huggingface python fusion_checkpoint.py \
  --data /cache/s95vkond/test_data/rgb_bricks --results approach_results \
  --out approach_results/baseline_checkpoint \
  --dino-revision ed25f3a31f01632728cabb09d1542f84ab7b0056 --bundle-encoder \
  --extra-metadata <baseline metadata json>
```

Before deployment, re-select the uncertainty margin and confirm accuracy on
RealSense top-surface crops of new physical bricks.
