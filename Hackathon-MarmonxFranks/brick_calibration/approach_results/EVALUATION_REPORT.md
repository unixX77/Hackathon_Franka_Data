# Brick classification approach comparison

## Evaluation protocol

- Dataset: 258 photographs of 10 physical bricks (4 good, 6 bad).
- Outer validation: leave one complete physical brick out.
- Inner model selection: leave one physical training brick out; mean per-brick
  image accuracy selects regularization.
- No photograph of an outer held-out brick is used for feature scaling,
  classifier fitting, or hyperparameter selection.
- Brick decisions average the held-out image decision scores for that physical
  brick. Image decisions describe single-view performance.

## Results matrix

| Approach | Image accuracy | Image bad recall | Brick accuracy | Brick bad recall | Bad bricks accepted | Good bricks rejected |
|---|---:|---:|---:|---:|---:|---:|
| Historical evaluator: centre ROI, fixed logistic regression | 63.6% | not saved in summary | 80.0% | 66.7% | 2/6 | 0/4 |
| 01 Centre ROI summaries + nested logistic regression | 58.9% | 57.7% | 70.0% | 66.7% | 2/6 | 1/4 |
| 02 Automatic mask, summaries + logistic regression | 61.2% | 55.7% | 70.0% | 50.0% | 3/6 | 0/4 |
| 03 Automatic mask, robust percentiles + logistic regression | 55.8% | 47.7% | 70.0% | 50.0% | 3/6 | 0/4 |
| 04 Automatic mask, LAB histograms + linear SVM | 66.7% | 59.7% | 70.0% | 50.0% | 3/6 | 0/4 |
| **05 Automatic mask, LAB histograms + texture + linear SVM** | **79.5%** | **69.1%** | **90.0%** | **83.3%** | **1/6** | **0/4** |
| 06 Approach 5 + localized crack/patch geometry | 79.5% | 73.2% | 90.0% | 83.3% | 1/6 | 0/4 |
| 07 Frozen DINOv2 global token + linear SVM | 69.8% | 70.5% | 70.0% | 66.7% | 2/6 | 1/4 |
| 08 Frozen DINOv2 patch distribution + linear SVM | 77.1% | 72.5% | 80.0% | 66.7% | 2/6 | 0/4 |
| 09 Approach 5 + compressed DINOv2 global token | 69.4% | 63.8% | 90.0% | 83.3% | 1/6 | 0/4 |
| 10 Approach 5 + compressed DINOv2 patch distribution | 69.4% | 62.4% | 80.0% | 66.7% | 2/6 | 0/4 |
| 11 DINOv2 nearest-good-reference anomaly | 52.3% | 49.0% | 60.0% | 50.0% | 3/6 | 1/4 |
| 12 DINOv2 true per-patch nearest-good anomaly | 57.0% | 49.7% | 70.0% | 66.7% | 2/6 | 1/4 |
| 13 DINOv3 true per-patch nearest-good anomaly | blocked | blocked | blocked | blocked | official model access required | official model access required |

The historical result is included for context, but it used fixed logistic
regression settings and a different LAB conversion implementation. The five
numbered approaches form the directly controlled comparison.

## What worked

The combined colour-distribution and texture representation performed best.
It correctly classified all four good physical bricks and five of six bad
physical bricks. It also recovered `Brick_8_bad`, which the historical model
missed. This suggests that defects are not represented by one global colour
shift alone; colour distribution and surface structure both carry useful
information.

LAB histograms alone improved single-image accuracy over summary statistics,
but did not improve physical-brick recall. Therefore, richer colour without
surface information is insufficient for this dataset.

## What did not work

Automatic masking followed by global summary statistics or additional
percentiles did not improve generalization. Those approaches accepted three of
the six bad physical bricks. More global percentiles also increased feature
dimension without providing new separability.

`Brick_6_bad` remains the failure of the best approach. Its mean held-out score
is strongly on the good side, so this is not merely a threshold-edge error.
Its defect needs to be inspected spatially and compared with the features of
the good bricks.

Approach 6 calculated localized dark contrast, gradient tails, connected dark
component size and elongation, long-line geometry, and maximum patch anomalies
for every image in all ten folders. It moved Brick 6's mean score from -1.038
to -0.630, but did not cross the bad decision boundary. At image level it
recovered six additional bad photographs but increased false rejection of good
photographs from 7 to 13. Because brick accuracy and bad-brick recall remained
unchanged, Approach 6 does not replace Approach 5.

Frozen DINOv2 results support one useful conclusion: patch-distribution
features are materially stronger than the global CLS token for this local
surface-defect task. Approach 08 reached 77.1% image accuracy versus 69.8% for
Approach 07, but neither matched Approach 5's brick-level result. The hybrid
representations also did not help: Approach 09 merely tied the brick result
while substantially reducing image accuracy, and Approach 10 introduced an
additional missed bad brick.

The nearest-good-reference anomaly model performed poorly. Four good physical
bricks do not define the normal appearance distribution sufficiently, and
viewpoint/material variation can be larger than the feature distance between
quality classes. It is not recommended with the current dataset.

Approach 12 retained every spatial DINOv2 patch and used the mean of the top 5%
nearest-good cosine distances. It produced all 258 held-out anomaly heatmaps,
but still missed `Brick_5_bad` and `Brick_6_bad` and falsely rejected
`Brick_2_good`. Brick 6 heatmaps respond broadly to ordinary surface and
boundary variation rather than concentrating on the crack. The failure of
Approach 11 was therefore not only caused by early patch aggregation: the
current good reference set and automatic material mask do not define a clean
local anomaly distribution.

Approach 13 is implemented but not evaluated. The configured Hugging Face
account has not been granted access to Meta's gated
`facebook/dinov3-vits16-pretrain-lvd1689m` weights. No unofficial mirror was
used to bypass the model license. Resume instructions are saved in
`13_dinov3_true_patch_anomaly/BLOCKED.md`.

## Important limitations

The automatic mask removes the white background and isolates the brick
silhouette well, but some dark pixels inside the perforations remain within the
silhouette. Inspect the 20 images in `mask_overlays/` before relying on the
texture gain. A future top-surface classifier should receive the surface mask
from the RGB-D segmentation pipeline rather than reproduce this photographic
foreground heuristic.

Only ten independent physical objects are available. The 258 images improve
repeatability estimates but do not create 258 independent examples. Trying
additional models against these same ten outer folds will gradually tune the
research process to this evaluation set.

Single-image bad recall for the best method is only 69.1% (46 bad photographs
were predicted good). The 90% brick result relies on aggregating many views of
each brick and should not be interpreted as a one-frame robot result.

## Recommendation

Use approach 05 as the current research baseline, not as the deployed robot
classifier. Next:

1. Inspect and annotate the actual defect on `Brick_6_bad`.
2. Replace the photographic automatic mask with the segmentation pipeline's
   exact top-surface mask.
3. Evaluate short stationary frame-burst aggregation, because the robot can
   capture several frames before choosing a brick.
4. Validate on RealSense top-surface crops captured at the robot's working
   distance and illumination.
5. Prioritize bad recall and introduce an uncertain/manual-review outcome
   instead of forcing every low-confidence sample into good or bad.

The DINO experiments do not replace this recommendation. Approach 5 remains
the frozen baseline. If pretrained dense features are revisited, they should be
computed from the robot segmentation's top-surface crop and retain spatial
patch positions rather than reducing patches to orderless mean/std/P90 values.
Approach 12 performed that spatial test and did not improve classification, so
future anomaly work should first use exact RGB-D top-surface masks and more
independent good bricks rather than only changing the pretrained encoder.

## Frozen Approach 5 checkpoint

The final Approach 5 pipeline was refit on all 258 images only after evaluation.
Grouped cross-validation selected a linear SVM with `C=0.1`. The serialized
pipeline and its fitted `StandardScaler` are stored in
`05_approach_checkpoint/model.joblib`; the exact feature schema, label mapping,
training counts and limitations are in `checkpoint_metadata.json`.

The checkpoint is for reproducibility and subsequent robot-domain validation.
Its held-out performance is the Approach 5 row above; training-set performance
must not be reported as an estimate of generalization.

## Orientation and canonical-surface follow-up

Approach 16 applies the unchanged 52-feature Approach 5 representation after
rotating and resizing each masked brick into a canonical horizontal crop. Its
grouped held-out result was 76.4% image accuracy, 69.1% image bad recall, 90%
brick accuracy, and 83.3% brick bad recall. It therefore does **not** replace
Approach 5 (79.5%, 69.1%, 90%, and 83.3%, respectively).

Most importantly, canonicalization did not solve `Brick_6_bad`: its aggregated
bad decision score remained negative (-0.832), so the physical brick was still
classified good. This indicates that orientation normalization is useful for
deployment consistency but does not create the missing defect signal.

The frozen checkpoint retained 89.9--93.0% in-sample accuracy across synthetic
rotations from 0 through 90 degrees. This is only a stress/stability test because
that checkpoint was fitted on all photographs. It must not be compared directly
with the held-out 79.5% figure.

Simulated score aggregation improved mean brick accuracy from 76.1% using one
random frame to 86.8% using nine frames, but it sampled the existing varied
photographs rather than consecutive RealSense frames. At a symmetric uncertainty
margin of 0.5, the canonical evaluator retained 79.8% coverage and achieved
82.5% accuracy on retained images. Both policies require calibration on new
robot-domain sequences before they can control a robot.

The saved evaluation is in `rotation_canonical_validation`. Keep Approach 5 as
the frozen baseline. Use exact top-surface corners from RGB-D segmentation for
the robot-facing perspective warp, collect the missing real angled/upright
validation images, and treat low-confidence predictions as `uncertain` rather
than automatically `good`.


## Accuracy follow-up: Approaches 17-22 and the Brick 6 label audit

Script: `evaluate_improvements.py` (features from `extract_improvement_features.py`
and `extract_rotated_dino_features.py`). Same nested leave-one-physical-brick-out
protocol as Approaches 01-16. Running Approach 5 through the new harness
reproduces 205/258 and confusion matrix `[[102, 7], [46, 103]]` exactly.

| Approach | Images correct | Image acc. | Image bad recall | Brick acc. | Brick 6 images |
|---|---:|---:|---:|---:|---:|
| 05 baseline (unchanged) | 205/258 | 79.5% | 69.1% | 90% | 1/24 |
| 08 DINOv2 patch distribution | 199/258 | 77.1% | 72.5% | 80% | 8/24 |
| **17 Equal-weight late fusion of 05 + 08** | **212/258** | **82.2%** | **75.2%** | **90%** | 1/24 |
| 18 Approach 5 + rotation-invariant LBP | 195/258 | 75.6% | 69.1% | 90% | 2/24 |
| 19 Approach 5 after paper white balance | 177/258 | 68.6% | 57.0% | 70% | 3/24 |
| 20 Late fusion of 18 + 08 | 206/258 | 79.8% | 74.5% | 90% | 1/24 |
| 21 DINOv2 patch, rotation-augmented (0/90/180/270) | 205/258 | 79.5% | 76.5% | 80% | 11/24 |
| 22 Late fusion of 05 + 21 | 206/258 | 79.8% | 73.8% | 90% | 3/24 |

Per-brick counts are in `improvement_comparison/per_brick_comparison.csv`.

### What improved

Approach 17 averages the Approach 5 and Approach 8 linear-SVM decision values with
a fixed weight of 0.5, so it adds no parameter tuned on held-out bricks. It was
motivated by the two models' complementary per-brick errors: Approach 5 is weak on
Bricks 8 and 10, Approach 8 on Bricks 2 and 5. Fusion keeps all four good bricks
correct and raises Brick 8 from 13/24 to 21/24 and Brick 10 from 19/28 to 23/28,
at a cost of 3 images on Brick 5 and 2 on Brick 2.

Relative to Approach 5, fusion fixes 15 images and breaks 8. An exact McNemar test
gives p = 0.21, and images of one physical brick are not independent, so the gain
is promising but not conclusive. The comparison design also used these same ten
bricks, so a new physical test brick set is needed to confirm it.

Deployment-style metrics on the same held-out scores
(`17_.../burst_aggregation_metrics.csv`, `uncertainty_metrics.csv`, baseline in
`improvement_comparison/approach5_baseline_deployment_metrics/`):

| | Approach 5 | Approach 17 |
|---|---:|---:|
| Brick accuracy, 1 random frame | 78.9% | 81.9% |
| Brick accuracy, 3-frame mean | 83.3% | 85.2% |
| Brick accuracy, 9-frame mean | 88.5% | 88.5% |
| Accuracy on confident images, margin 0.3 | 81.9% (87.6% coverage) | 86.1% (81.0% coverage) |

Fusion helps most for single frames and short bursts. With nine frames both models
reach the same brick accuracy.

Rotation augmentation (Approach 21) improved the DINOv2 branch from 77.1% to 79.5%
and is the only model that moved Brick 6 near the boundary (11/24, mean score
-0.10). It also collapsed on Brick 5 (3/25) and lost Brick 2 images, and fusing it
with Approach 5 (22) diluted the Brick 6 signal. LBP texture (18) and paper white
balance (19) reduced accuracy. White balance appears to remove useful absolute
colour information under this fixed lighting.

### Rejected shortcut: silhouette shape

Outline-damage features (solidity, convexity defects) separated the classes
strongly: good-brick solidity 0.969-0.973, bad 0.914-0.966, Brick 6 0.956. They
were **not used**. Visual inspection (`brick6_label_audit/rejected_silhouette_shortcut_*.jpg`)
shows the deep defects on bad bricks come from the automatic mask leaking into red
brick dust lying on the paper. Dust is present in most bad-brick photographs and
absent from good-brick photographs, and it will not exist on the robot. The same
leak affects under 0.5% of mask pixels for every brick (0.29-0.33% good,
0.33-0.48% bad, measured by large-kernel opening), so it does not materially
inflate the colour and texture histograms of Approach 5 or 17.

### Brick 6: why no model separates it

`brick6_label_audit/` contains side-by-side crops. `Brick_2_good` (e.g. b034, b032)
and `Brick_4_good` (b085) both show a full-depth transverse crack through the brick
that is at least as visible as the crack on `Brick_6_bad` (b030, b042). Brick 6's
colour also lies in the good-brick range (dark red-brown with white patches, like
Brick 2), while the other bad bricks are lighter, sandier orange.

When Brick 6 is held out, the training set teaches every model that a crack on a
dark red-brown brick is acceptable. Any feature that would flag Brick 6's crack
would equally flag Bricks 2 and 4. With the current labels, Brick 6 cannot be
learned from visible appearance without overfitting to the held-out brick itself.

The labelling rule needs to be confirmed by the person who assigned the labels:

1. If a through-crack makes a brick bad, `Brick_2_good` and `Brick_4_good` are
   mislabelled and should be relabelled before further modelling.
2. If those cracks are acceptable, record what makes Brick 6 bad (for example
   strength, sound, a defect on a face that was not photographed, or crack width).
   If it is not visible on the photographed faces, an RGB classifier cannot be
   expected to detect it.

### Recommendation

Use Approach 17 as the research candidate and keep Approach 5 as the official
baseline until the gain is confirmed on new physical bricks. The frozen checkpoint
is `17_approach_checkpoint/` (`fusion_checkpoint.py` loads it; `classify` returns
`good`, `bad` or `uncertain`, and `classify_burst` uses the median frame score).
Its default uncertainty margin of 0.3 was read from this held-out sweep and must be
re-selected on RealSense top-surface validation bricks. Resolve the Brick 2, 4 and
6 labelling question before any further model search; more experiments on the same
ten folds now risk tuning to the evaluation set.

Masks are produced by OpenCV GrabCut, which has a random initialisation, so
recomputed masks differ slightly between runs. Checkpoint scores on raw images
match cached-feature scores to within about 0.01 (Approach 5 branch) and 0.2
(DINOv2 branch) in spot checks; no predicted labels changed.

## Baseline change (2026-10-25)

Approach 17 is now the official baseline, replacing Approach 5. The
self-contained checkpoint (SVMs plus pinned, bundled DINOv2 encoder) is
`baseline_checkpoint/`; see `baseline_checkpoint/BASELINE.md`. Its SVMs give
identical scores to `17_approach_checkpoint/`. `05_approach_checkpoint/` is kept
unchanged as the previous baseline for comparison. Earlier sections that call
Approach 5 the baseline describe the state before this change.
