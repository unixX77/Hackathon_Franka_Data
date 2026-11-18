# Approach 17 score distributions

`folder_score_distributions.png` shows the 258 saved SAM3-mask photos in
`../offline_results/all_predictions.csv`. Every dot is one photo. Green/red
indicates the brick's **known folder label**, and each black diamond is the
folder mean. The amber band on the combined score marks the current review
range: good if score <= -1.75, review if -1.75 < score <= -1.20, and bad if
score > -1.20.

The model has one Approach 5 linear SVM for color **and** texture; these are
not separate classifiers. The first two panels show how the two groups of
standardized features contribute additively to that SVM's decision score:

```
Approach 5 score = color contribution + texture contribution + intercept
Approach 17 score = (Approach 5 score + DINOv2 SVM score) / 2
```

The intercept is saved in `metadata.json`; the current value is approximately
0.19258. Contributions are relative to the scaler's training mean. They are
not independently calibrated good/bad scores, and should not be thresholded
as individual color or texture classifiers. The DINOv2 column is the SVM
decision value from DINOv2 patch features, not a raw DINO feature or a
probability. Higher scores tend toward bad under this model; the review
threshold applies only to the **combined Approach 17 score**.

`folder_score_statistics.csv` contains mean, population standard deviation,
minimum and maximum for each group and score. `per_image_scores.csv` contains
the contributing values and original scores for each photo. Reconstruction of
the Approach 5 score is checked against each saved result; the maximum error
is written to `metadata.json`.

These photos belong to the ten physical bricks used to fit the saved model.
The plots describe this offline run and do not measure held-out or live-camera
accuracy. Differences between folder means also do not by themselves identify
the physical cause of a classification.

To regenerate from the repository root:

```bash
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/visualize_offline_scores.py
```
