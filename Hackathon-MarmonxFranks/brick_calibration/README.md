# Brick color calibration evaluation

This tests color capture repeatability and good/bad separation of **ten physical bricks** (4 good, 6 bad), with images grouped by physical brick ID. It **does not determine the correct labels**, which you must obtain from your hackathon ground truth.

## Install

```bash
python -m pip install -r requirements.txt
```

## Data layout

```text
data/
  good/
    b01_good/01.png 02.png ...
    b02_good/...
    b03_good/...
    b04_good/...
  bad/
    b05_bad/01.png 02.png ...
    ...
    b10_bad/...
```

The evaluator also accepts the captured-data layout directly, without copying
or relabelling images:

```text
data/
  Brick_1_good/*.png
  Brick_2_good/*.png
  ...
  Brick_10_bad/*.png
```

## Brick material sampling IMPORTANT

Preferred: create white-on-black **surface masks** (255=brick material, 0=background, holes, deep shadows), at `masks/good/b01_good/01.png`, etc. All masks must match image size. Then run:

```bash
python evaluate_calibration.py --data data --masks masks --out results
```

If your images are cropped so the **central 40% width and height contains only brick material**, you can instead run:

```bash
python evaluate_calibration.py --data data --out results --roi-fraction 0.4
```

This center-region option is **not automatic brick detection**. If the center includes holes or background, measured LAB values are unreliable. Prefer masked brick face patches and look at several locations on variegated bricks.

## Outputs

- `image_quality_and_features.csv`: image brightness, fraction of near-black/near-white pixels, and LAB features.
- `within_brick_consistency.csv`: within-object ΔE00 mean/max/p90. Smaller is generally more repeatable (compare matched views if faces differ).
- `pairwise_brick_distances.csv`: pairwise median ΔE00 among different physical bricks.
- `nearest_opposite_class.csv`: nearest different-label brick and rough separation/noise ratio.
- `color_distance_heatmap.png`: brick-wise ΔE00 visualization.
- `lab_chromaticity.png`: a*/b* scatter with 1 point per image and emphasized per-brick medians.
- `held_out_image_predictions.csv`, `held_out_brick_predictions.csv`: logistic regression predictions using **leave-one-physical-brick-out** validation.
- `summary.json`: evaluation counts, bad recall/precision/F1, confusion matrix.

## Interpreting results

1. Check images for visible clipping, blur, glare, shadow, and color cast before trusting calculations. The script only reports a crude near-black and near-white pixel test, not perfect exposure assessment.
2. For each brick, evaluate consistency across repeated shots **of the same face**, then optionally different faces. Different surfaces on clay bricks can have real intrinsic color variation.
3. Look for bricks that are close to the opposite label in `nearest_opposite_class.csv`; inspect these manually.
4. Held-out-brick performance measures generalization to unseen physical specimens, but is very uncertain with only 10 objects; **do not read image-level scores as 50 independent examples**.
5. `ΔE00` uses image colors as measured, not certified physical color. Stable CCT alone does not ensure accurate color rendering; ideally use a camera color card or fixed manual white balance/exposure.
6. Adjusting lighting, feature choices or thresholds after looking at hold-out results makes them no longer an unbiased final estimate. Record initial results, and reserve future physical bricks for a final test when possible.

## Compare multiple approaches

Run the controlled grouped-validation comparison on the captured flat folder
layout with:

```bash
python evaluate_approaches.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --out approach_results
```

Each approach receives its own result directory. The combined comparison is in
`approach_results/evaluation_matrix.csv`, and the interpretation is recorded
in `approach_results/EVALUATION_REPORT.md`.

**Current live baseline: SAM3 masks + Approach 17 + the
`review_band_175_120` policy.** The frozen Approach 17 model is stored at
`approach_results/baseline_checkpoint/` (load with
`fusion_checkpoint.FusionClassifier.load`; see `BASELINE.md` there).
For live use, classify its score with the saved policy: good at <= -1.75,
uncertain through -1.20, bad above -1.20. The Python `classify` method itself
still returns the historical zero-centered margin label.

For a USB RGB camera, `live_rgb_classifier.py` displays a live frame next to
Approach 17's most recent good/bad/uncertain prediction. Setup, camera-index
options, transfer instructions, and output details are in `LIVE_RGB.md`.
When the camera is attached to the Mac and the model must stay on photolab,
use `remote_rgb_client.py` and `remote_rgb_server.py` as described in
`REMOTE_RGB.md`.
The separate `live_video/` variant adds a selected single-brick crop, visible
material mask, fresh-frame bursts, quality checks, labelled Mac capture/upload,
and a camera-domain evaluation script. See `live_video/README.md`; its
checkpoint weights remain the frozen Approach 17 baseline.

An offline SAM3-mask experiment on all 258 labelled photos is implemented in
`offline_sam3_classifier.py`; folder-wise masks, soft red/green outlines,
scores, and a matched GrabCut comparison are in `offline_results/`.
The same SAM3-mask flow is the default in the Mac two-panel live viewer; see
`sam3_live/README.md` for the baseline command and evaluation limitations.
The separate experimental `local_defect_candidate.py` adds local defect
features and grouped training. Its checkpoint and evaluation are in
`approach_results/local_defect_candidate/`. It scored 158/258 in leave-one-
physical-brick-out testing, so Approach 17 remains the recommended baseline.
The live option is `--pipeline sam3_local`; see the candidate README for its
limits and the face-label review template.

The previous baseline, the frozen Approach 5 model, is stored at
`approach_results/05_approach_checkpoint/model.joblib`. Always use it with the
feature extractor in `evaluate_approaches.py`; the model expects the exact
52-dimensional Approach 5 feature vector.

Frozen DINOv2 and hybrid experiments can be reproduced with:

```bash
HF_HOME=/cache/s95vkond/huggingface python evaluate_dino_approaches.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --results approach_results
```

After the first run, add `--reuse-features` to reuse
`approach_results/dino_feature_cache/dinov2_features.npz` without loading the
encoder again. Approaches 07–11 use a frozen `facebook/dinov2-small`; PCA and
the classifier are fitted only inside their grouped training folds.

True per-patch anomaly evaluation is implemented in
`evaluate_patch_anomaly.py`. Dense feature caches are stored under
`/cache/s95vkond/brick_calibration_features`, while metrics and heatmaps remain
under `approach_results/12_dinov2_true_patch_anomaly` and
`approach_results/13_dinov3_true_patch_anomaly`.

## Orientation and deployment validation

Run the frozen Approach 5 rotation stress test, canonical surface experiment,
frame-burst simulation, and uncertainty sweep with:

```bash
python evaluate_canonical_a5.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --checkpoint approach_results/05_approach_checkpoint/model.joblib \
  --results approach_results/rotation_canonical_validation
```

`canonical_surface.py` contains two separate operations:

- `canonicalize_masked_object` is the photographic-dataset evaluator. It
  estimates orientation from the automatically generated foreground mask.
- `canonicalize_detected_surface` is the robot-facing helper. It perspective
  warps the four ordered top-surface corners delivered by segmentation.

The rotation table is a robustness stress test of a checkpoint fitted to all
images; it is **not held-out accuracy**. The canonical experiment uses the same
leave-one-physical-brick-out evaluation as the main comparison. Burst results
simulate combining existing photographs and must be confirmed with consecutive
RealSense frames. See `rotation_canonical_validation/RESULTS.md` and
`REAL_ROBOT_VALIDATION.md` before deployment.

## Accuracy follow-up (Approaches 17-22)

Extract the additional features, then evaluate score fusion, LBP texture, paper
white balance and rotation augmentation under the same grouped protocol:

```bash
python extract_improvement_features.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --out approach_results/improvement_feature_cache/improvement_features.npz
HF_HOME=/cache/s95vkond/huggingface python extract_rotated_dino_features.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --out approach_results/improvement_feature_cache/dinov2_rotated_patch_features.npz
python evaluate_improvements.py \
  --data /cache/s95vkond/test_data/rgb_bricks --results approach_results
```

`evaluate_improvements.py` also appends rows 17-22 to `evaluation_matrix.csv` and
writes `approach_results/improvement_comparison/`. The best candidate, Approach 17
(equal-weight fusion of Approach 5 and DINOv2 patch-distribution scores), is
frozen with:

```bash
HF_HOME=/cache/s95vkond/huggingface python fusion_checkpoint.py \
  --data /cache/s95vkond/test_data/rgb_bricks \
  --results approach_results --out approach_results/17_approach_checkpoint
```

Read "Accuracy follow-up" in `approach_results/EVALUATION_REPORT.md`, including the
Brick 6 label audit in `approach_results/brick6_label_audit/`, before using it.

## Brick-disjoint train / validation / test splits

```bash
python make_brick_splits.py --data /cache/s95vkond/test_data/rgb_bricks \
  --out /cache/s95vkond/brick_splits
python evaluate_on_splits.py --data /cache/s95vkond/test_data/rgb_bricks \
  --splits /cache/s95vkond/brick_splits --results approach_results
```

Two families of hard-linked splits (with and without Brick 6) are written to the
cache and verified for non-overlap. A local copy (images plus symbolic-link split
folders, identical assignments) is in `brick_splits/`; see
`brick_splits/SPLIT_PROTOCOL.md`. Results and the comparison with
leave-one-brick-out are in `approach_results/split_evaluation/SPLIT_EVALUATION_REPORT.md`.
