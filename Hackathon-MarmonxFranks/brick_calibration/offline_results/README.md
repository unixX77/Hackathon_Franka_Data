# Offline SAM3 → Approach 17 results

Input: `../brick_splits/images/Brick_N_good|bad/*.png` (258 labelled photos).
The input images are preserved. Each output folder mirrors an input folder.

Pipeline for each photo:

1. Load the cached SAM3 model once and prompt it with `brick`.
2. Keep the dominant central whole-brick instance. Small border detections
   are ignored; a large, high-confidence brick partly cut off by the frame
   can be retained and marked `sam_touches_border`. Save its binary pixel
   mask in `masks/`.
3. Pass the original RGB frame **and that SAM3 mask** to the frozen Approach 17
   `FusionClassifier`. This replaces the automatic GrabCut mask for this run.
4. Apply the saved `minus_two_binary` policy: score ≤ −2.0 is good; score > −2.0
   is bad. Colour the mask contour green or red using a soft 3-pixel boundary.

Each folder contains the annotated image, a same-stem JSON record, a `masks/`
subfolder, and `predictions.csv`. The root contains `all_predictions.csv`,
`summary.json`, a matched `mask_input_comparison.csv/json`, and run logs. The
comparison uses cached GrabCut features from the same photos and the same
frozen checkpoint and −2 policy. The JSON records include the tight mask bounding
box, SAM3 score, Approach 5 score, DINOv2 score, and labels from all three
saved threshold policies.

To reproduce from the repository root:

```bash
HF_HOME=/cache/s95vkond/huggingface \
/home_domuser2/s95vkond/miniconda3/envs/sam3/bin/python \
  brick_calibration/offline_sam3_classifier.py
```

To redraw only the saved contours at another width without rerunning either
model, use `python brick_calibration/rerender_offline_sam3.py --outline-width 3`
inside the `sam3` environment.

The SAM3 checkpoint is read from the cached local snapshot, so this command
does not need to fetch the gated model online. The Approach 17 checkpoint is
read from `../approach_results/baseline_checkpoint/`.

These photos are from the same ten physical bricks used to fit the frozen
classifier. Folder accuracy here measures this mask change on familiar bricks;
it is **not** a leave-one-brick-out or new-brick generalisation estimate. The
new mask can change score calibration, so the −2 threshold should be checked
on separately labelled live-camera frames before deployment.

## Observed results with the −2.0 policy

SAM3 produced a selected brick mask for all **258/258** photos. One Brick 7
photo has a brick partly outside the frame and is marked
`sam_touches_border: true` in its JSON metadata. The classifier matched the
folder label on **207/258 photos (80.2%)**.

| Folder | Correct / photos |
|---|---:|
| Brick_1_good | 15 / 26 |
| Brick_2_good | 18 / 30 |
| Brick_3_good | 13 / 28 |
| Brick_4_good | 21 / 25 |
| Brick_5_bad | 25 / 25 |
| Brick_6_bad | 23 / 24 |
| Brick_7_bad | 24 / 24 |
| Brick_8_bad | 19 / 24 |
| Brick_9_bad | 23 / 24 |
| Brick_10_bad | 26 / 28 |

With the same checkpoint and −2 policy, the previous cached GrabCut features
gave **206/258** correct on these same familiar photos. The segmentation mask
changed 31 individual labels; its outline is visually tighter, but the overall
classification gain is only one photo in this in-sample comparison. The
SAM3-mask pipeline incorrectly labels many good-brick photos as bad, especially
Brick 3. Inspect [the matched comparison](mask_input_comparison.csv) and the
folder images before using it as the live default.
