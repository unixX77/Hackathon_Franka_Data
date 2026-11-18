# Approach 5 orientation follow-up

## Outcome

The canonical-crop variant does not replace the frozen Approach 5 baseline.

| Metric | Frozen A5 held-out | Canonical A5 held-out |
|---|---:|---:|
| Image accuracy | 79.5% (205/258) | 76.4% (197/258) |
| Image bad recall | 69.1% (103/149) | 69.1% (103/149) |
| Brick accuracy | 90.0% (9/10) | 90.0% (9/10) |
| Brick bad recall | 83.3% (5/6) | 83.3% (5/6) |

`Brick_6_bad` is still the single missed physical bad brick. Its canonical
mean bad-decision score is -0.832 (negative means good), so canonical alignment
does not supply the missing visual defect evidence.

## What was evaluated

1. The original frozen checkpoint was tested at 0, 15, 30, 45, 60, 75, and
   90 degrees. Accuracy ranged from 89.9% to 93.0%. This is an in-sample
   robustness test, not a generalization estimate.
2. Each foreground mask was aligned by its minimum-area rectangle, cropped,
   resized to 256 x 128, and evaluated using leave-one-physical-brick-out folds.
3. Mean decision-score aggregation was simulated with 1, 3, 5, and 9 images.
   Mean brick accuracy increased from 76.1% to 86.8%.
4. Symmetric reject/uncertain margins were evaluated. Margin 0.5 retained
   79.8% of images; accuracy on retained images was 82.5%.

## Interpretation

Rotation itself is not the main Folder 6 problem. The current global color and
texture representation sees Brick 6 as resembling good bricks. Canonicalization
also loses some image-level performance. Preserve the original A5 checkpoint,
use canonical top-surface warping as a robot-input normalization step only after
it is retrained and validated on robot crops, and gather targeted images where
the Brick 6 defect is visible.

Detailed artifacts are in `rotation_metrics.csv`, `burst_aggregation_metrics.csv`,
`uncertainty_metrics.csv`, and `16_canonical_surface_approach5/`.
