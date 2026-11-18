# Face, color, and rotation candidate

This is a separate experiment using the existing SAM3 masks. It does not change
the official live baseline, which remains SAM3 + Approach 17 +
`review_band_175_120`.

## Four feature groups

1. **View/orientation proxy:** the mask's axis angle, aspect ratio,
   rectangularity, and dark-region statistics. There are no verified face
   labels; dark regions can be holes, shadows, or marks. This is not a
   reliable physical-face classifier or 3-D pose.
2. **Visible material color:** canonicalize the brick's long axis and remove
   perimeter pixels, very dark pixels, highlights, and low-saturation pixels
   heuristically. Extract a LAB histogram and material color percentiles.
   Examine `material_mask_examples/`: the left image is the canonical crop,
   and green tint on the right shows selected material. Holes and shadows are
   not guaranteed to be excluded; real dark material can also be excluded.
3. **Local color variation:** summarize LAB medians across a 2 x 4 grid on
   that mask, then compute percentiles and variation across the regions.
4. **Rotation-aware DINOv2:** extract patch-distribution features at 0, 90,
   180, and 270 degrees in the image plane and average them. A 90-degree
   test on a fixed saved mask gave a maximum feature difference below 1e-6.
   These synthetic rotations do not reveal the brick's hidden faces.

The candidate also includes the original Approach 5 color/texture features.
The feature groups are fed to a standardized linear SVM; rotation-aware DINOv2
features are reduced to 16 PCA components. The raw SVM threshold and SVM C
are selected using only group-stratified inner training folds. The reported
`score_bad` is raw SVM score minus that threshold; positive means bad. This
score and threshold are not interchangeable with Approach 17's review policy.

## Results

All 258 photos received a SAM3 mask. In the outer evaluation, each physical
brick was held out in turn. PCA, scaling, model fitting, and inner threshold
selection used only the remaining bricks.

| Feature setup | Correct photos | Good recall | Bad recall | Bricks correct |
|---|---:|---:|---:|---:|
| Single-SVM control, original A5 + original DINO | 167/258 (64.7%) | 82/109 (75.2%) | 85/149 (57.0%) | 6/10 |
| Add view/orientation proxy | 175/258 (67.8%) | 86/109 (78.9%) | 89/149 (59.7%) | 8/10 |
| Add material color | 178/258 (69.0%) | 87/109 (79.8%) | 91/149 (61.1%) | 7/10 |
| Add local color | 173/258 (67.1%) | 86/109 (78.9%) | 87/149 (58.4%) | 7/10 |
| Use rotation-averaged DINO | 176/258 (68.2%) | 83/109 (76.1%) | 93/149 (62.4%) | 7/10 |
| All four additions | 179/258 (69.4%) | 92/109 (84.4%) | 87/149 (58.4%) | 7/10 |

Each row uses the same masks, grouping, single-SVM architecture, and nested
threshold selection. Rows with a single addition use the original DINO
features except the rotation row. Brick-level results use the median of all
held-out photo margins per physical brick; they are not one-frame results.
These ablations are exploratory comparisons on ten bricks, not independent
confirmation. Selecting a winner after examining these results would make this
same set an optimistic final test.

The official Approach 17's historical result is 212/258 (82.2%), but it used
different automatic masks and a two-SVM fusion model, so the comparison is not
a controlled feature ablation. The combined candidate detected **2/24**
photos each for Brick 6 and Brick 8. It has not shown a reason to replace the
existing live baseline, and it has not been validated on the Mac or robot
camera.

`held_out_image_predictions.csv`, `held_out_brick_predictions.csv`, and
`held_out_fold_parameters.csv` contain the main candidate results. The
`feature_ablation_matrix.csv` and `ablations/` folders contain per-feature
comparisons. `view_diagnostics.json` contains per-image proxy information.

## Experimental live test

The separate Mac viewer option is `--pipeline sam3_face_color`. Copy the
updated client first. It segments up to ten bricks and sends each through this
candidate, drawing green/red contours. Its per-detection JSON includes the
raw score, threshold and view proxy; no physical face identity is asserted.

```bash
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --pipeline sam3_face_color --interval 3 --save-frames
```

The policy argument for Approach 17 does not apply to this candidate. A local
protocol smoke test of two saved bricks ran in about 1.1 seconds after model
loading; this does not include Mac network or camera delays. Four DINO passes
are batched for every detected brick; processing time can increase with more
bricks. The second camera and cross-camera matching are not implemented here.

## Reproduce

```bash
HF_HOME=/cache/s95vkond/huggingface \
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/face_color_rotation_candidate.py

HF_HOME=/cache/s95vkond/huggingface \
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/evaluate_face_color_ablations.py
```

Results and the small checkpoint are in this working-directory folder. The
1.2 MB feature cache is at
`/cache/s95vkond/brick_feature_cache/face_color_rotation_features.npz`. To
recover workspace quota, other reusable feature caches were copied and
verified in the same cache directory, with symlinks at their original project
paths. Those symlinks work on this lab machine; copying the project elsewhere
requires copying their target cache files as well.
