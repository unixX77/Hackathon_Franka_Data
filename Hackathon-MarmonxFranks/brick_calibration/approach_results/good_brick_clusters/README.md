# Good-brick reference clusters

This experiment keeps the existing Approach 17 checkpoint and its policies
separate. It asks whether a brick is close to one of the known good-brick
appearance clusters. It is a binary classifier: the good-reference prototypes
use only good bricks, while known bad labels are used when selecting a
threshold and configuration on training/validation data.

## Method and threshold

1. Start with the saved SAM3 brick mask. Extract LAB histogram and texture
   features, and optionally frozen DINOv2 patch-distribution features.
2. Fit feature scaling on the training **good** bricks only, giving each
   physical good brick equal total weight. Build one or two K-means prototypes
   per good physical brick; these are appearance clusters, not verified face
   identities.
3. Compute the distance to the nearest good prototype. Normalize it by the
   median distance of the good training photos to their nearest prototype.
4. Select features, prototype count, and distance threshold using only inner
   physical-brick validation. Optimize class-balanced mean image accuracy,
   giving each physical brick equal weight; use bad-class accuracy to break
   ties. No outer held-out brick is used for any of these choices.
5. Predict bad when normalized distance > threshold, good otherwise. The
   displayed `score_bad` is `distance - threshold`; it is not a probability and
   is on a different scale from Approach 17 scores.

The final `reference_model.joblib` is fitted to the four good bricks with two
prototypes per brick. Inner grouped selection chose **color plus texture** and
threshold **1.179216**; DINOv2 was among the candidate representations but was
not selected for this checkpoint. A saved nearest-reference name means visual
similarity to that training group; it does not identify the physical brick.

## Evaluation: unseen physical brick

Each of the ten physical bricks was held out in turn. Its photos were not used
to fit the scaler, prototypes, feature choice or threshold in that fold.
Results: **130/258 photos correct (50.4%)**, good recall 35/109 (32.1%), bad
recall 95/149 (63.8%), and 4/10 physical bricks correct using median photo
scores. Brick 6 had **2/24** correct photos. `held_out_image_predictions.csv`,
`held_out_brick_predictions.csv`, `held_out_fold_parameters.csv`, and
`held_out_per_brick.png` show the details.

The earlier Approach 17 held-out evaluation reported 212/258 (82.2%) with a
different automatic mask. It therefore is not a controlled same-mask ablation.
This reference experiment does not provide evidence that it outperforms
Approach 17 or is reliable enough for automatic sorting.

## Evaluation: same ten bricks, different ordered photos

The separate known-brick experiment uses ordered filenames per brick: 60% of
photos for prototype training, 20% for selecting representation and threshold,
and the remaining 20% for testing. The test photos are not used in model or
threshold selection. The SHA-256 of all 258 image files was distinct; the
manifest lists the exact split. Adjacent images may still be near duplicates,
and filename order is not proof of distinct sessions or faces.

On the 51 test photos, the selected model correctly classified **33/51
(64.7%)**: 8/21 good photos and 25/30 bad photos. These are the same physical
bricks as in training, so this is not a test of new bricks. The experimental
model for this split is stored as `known_brick_demo_model.joblib`; the live
mode described below uses the full-data `reference_model.joblib` instead.

## Live test

Copy the updated Mac `remote_rgb_client.py`, then use
`--pipeline sam3_reference` to run this reference classifier. SAM3 and the
model run on photolab; the Mac shows individual green/red brick contours. The
per-frame `detections_*.json` records normalized distance, threshold, and the
closest good reference. `--policy` values for Approach 17 do not apply.

```bash
cd /Users/varunteja/Downloads/brick_live_runtime
ssh s95vkond@131.220.238.95 \
  "cd '/ipb70/home_domuser2/s95vkond/Desktop/Franks robotics/Hackathon-MarmonxFranks' && tar -cf - brick_calibration/remote_rgb_client.py" \
  | tar -xf -
python brick_calibration/remote_rgb_client.py \
  --host 131.220.238.95 --camera 1 --backend avfoundation \
  --pipeline sam3_reference --interval 3 --save-frames
```

The default `--pipeline sam3 --policy review_band_175_120` continues to select
Approach 17. Its score threshold is unrelated to the prototype distance.

To reproduce both evaluations and the full-data checkpoint from the repository
root:

```bash
HF_HOME=/cache/s95vkond/huggingface \
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/good_brick_clusters.py
```
